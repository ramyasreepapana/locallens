"""
OCR engine: runs RapidOCR (PaddleOCR PP-OCRv4 models, exported to ONNX) through ONNX Runtime.

The three ONNX model files (text detection, text direction, text recognition) are shipped
INSIDE the `rapidocr-onnxruntime` pip package, so there is no model download and the app
works fully offline after `pip install`.

Backends
    "cpu" : ONNX Runtime CPUExecutionProvider. Implemented and tested.
    "qnn" : registers Qualcomm's `onnxruntime-qnn` plugin execution provider with ONNX Runtime
            and asks for the NPU device. It refuses to start if no NPU device is found.
            NOT tested on Snapdragon hardware by the author. These models have dynamic input
            shapes; the NPU often needs fixed-shape or quantized models, so ONNX Runtime may
            keep some or all of the work on the CPU. The app therefore reports what the
            sessions REALLY ended up using.
"""

import os
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np

QNN_PROVIDER = "QNNExecutionProvider"


class OCRError(Exception):
    """Message is safe to show to a normal user."""


@dataclass
class OCRLine:
    text: str
    confidence: float
    box: List[List[float]]          # four [x, y] corner points


@dataclass
class OCRResult:
    lines: List[OCRLine] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


def _find_qnn_devices(ort, allow_any_device: bool):
    """
    Register the onnxruntime-qnn plugin EP and return its hardware devices.
    Only NPU devices are accepted, unless allow_any_device is set (developer testing on
    a non-Snapdragon machine, where the QNN plugin exposes a CPU-type device only).
    """
    try:
        import onnxruntime_qnn
    except ImportError as err:
        raise OCRError("onnxruntime-qnn is not installed. Run: pip install onnxruntime-qnn") from err
    try:
        ort.register_execution_provider_library(onnxruntime_qnn.get_ep_name(), onnxruntime_qnn.get_library_path())
    except Exception as err:  # noqa: BLE001
        # Registering twice raises an error in some versions; that is harmless if devices exist.
        if not any(d.ep_name == QNN_PROVIDER for d in ort.get_ep_devices()):
            raise OCRError(f"The QNN plugin could not be registered ({type(err).__name__}: {err}).") from err
    devices = [d for d in ort.get_ep_devices() if d.ep_name == QNN_PROVIDER]
    if not allow_any_device:
        devices = [d for d in devices if "NPU" in str(d.device.type)]
    if not devices:
        raise OCRError(
            "No Snapdragon NPU device was found by ONNX Runtime's QNN plugin on this machine. "
            "Use the CPU backend, or run on a Snapdragon X PC with onnxruntime-qnn installed (see README)."
        )
    return devices


def _patch_onnxruntime_for_qnn(devices) -> None:
    """
    RapidOCR only knows CPU/CUDA/DirectML. To use the QNN plugin we wrap the InferenceSession
    that RapidOCR creates, so the QNN devices are attached to its session options first.
    """
    from rapidocr_onnxruntime.utils import infer_engine

    original_session = infer_engine.InferenceSession
    if getattr(original_session, "_locallens_qnn", False):
        return

    def qnn_session(model_path, sess_options=None, providers=None, **kwargs):
        sess_options.add_provider_for_devices(devices, {})   # QNN first; ORT keeps CPU as fallback
        return original_session(model_path, sess_options=sess_options, **kwargs)

    qnn_session._locallens_qnn = True
    infer_engine.InferenceSession = qnn_session


class LocalOCR:
    def __init__(self, backend: str = "cpu"):
        if backend not in ("cpu", "qnn"):
            raise OCRError(f"Unknown backend '{backend}'. Use 'cpu' or 'qnn'.")
        self.backend = backend
        self._engine = None
        self._qnn_device_types = []
        self.load_seconds = 0.0

    # ---------- loading ----------

    def load(self) -> None:
        if self._engine is not None:
            return
        try:
            import onnxruntime
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as err:
            raise OCRError("Required packages are missing. Run: pip install -r requirements.txt") from err

        if self.backend == "qnn":
            devices = _find_qnn_devices(onnxruntime, allow_any_device=bool(os.environ.get("LOCALLENS_QNN_ANY_DEVICE")))
            self._qnn_device_types = [str(d.device.type) for d in devices]
            _patch_onnxruntime_for_qnn(devices)

        started = time.perf_counter()
        try:
            self._engine = RapidOCR()
        except Exception as err:  # noqa: BLE001 - shown to the user as a friendly message
            raise OCRError(f"The OCR models could not be loaded ({type(err).__name__}: {err}).") from err
        self.load_seconds = time.perf_counter() - started

    # ---------- inference ----------

    def run(self, image) -> OCRResult:
        """image: a PIL image or a numpy array (RGB)."""
        self.load()
        array = np.array(image.convert("RGB")) if hasattr(image, "convert") else np.asarray(image)
        started = time.perf_counter()
        try:
            raw_result, _ = self._engine(array)
        except Exception as err:  # noqa: BLE001
            raise OCRError(f"Text recognition failed ({type(err).__name__}: {err}).") from err
        seconds = time.perf_counter() - started

        lines = [
            OCRLine(text=text, confidence=float(conf), box=[[float(x), float(y)] for x, y in box])
            for box, text, conf in (raw_result or [])
        ]
        return OCRResult(lines=lines, seconds=seconds)

    # ---------- honest reporting ----------

    def describe(self) -> dict:
        """What is REALLY in use, read back from the ONNX Runtime sessions."""
        info = {
            "model": "PaddleOCR PP-OCRv4 (ONNX) via RapidOCR",
            "runtime": "ONNX Runtime",
            "requested_backend": self.backend,
            "network_at_runtime": False,
        }
        if self._engine is None:
            return info
        sessions = {
            "detection": self._engine.text_det.infer.session,
            "recognition": self._engine.text_rec.session.session,
        }
        used = {name: s.get_providers() for name, s in sessions.items()}
        info["providers_per_model"] = used
        info["qnn_device_types"] = self._qnn_device_types
        # True only if a QNN session was built on an NPU-type device. Confirm speed/placement with profiling.
        info["runs_on_npu"] = any(p and p[0] == QNN_PROVIDER for p in used.values()) and any(
            "NPU" in t for t in self._qnn_device_types
        )
        return info
