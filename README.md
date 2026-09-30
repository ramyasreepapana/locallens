# LocalLens

**Read documents with AI, privately. Your images never leave your device.**

LocalLens turns a photo or screenshot of a document (invoice, receipt, notice, form) into text. A neural-network OCR model runs **locally** through ONNX Runtime, then simple pattern matching pulls out emails, phone numbers, dates, amounts and links. No cloud API, no account, no model download.

> **Honesty box - read this first**
>
> | Status | What |
> |---|---|
> | **Implemented and tested** (x86-64 Linux, Python 3.11) | Real OCR on the sample image with the real model (12 automated tests, including an end-to-end UI test); OCR runs with **all network sockets blocked**; the server binds to localhost only; refusal of the QNN backend when no NPU device exists |
> | **Not tested** | Windows, any Snapdragon device, the NPU. The QNN backend is written against the real `onnxruntime-qnn` plugin API, and in a dry run on a non-NPU machine it attached the plugin, failed to create a QNN session, fell back to CPU and reported that truthfully. Whether it succeeds on a real Snapdragon NPU is **unknown** |
> | **Planned** | Fixed-shape / quantized OCR models for the NPU, measured CPU vs NPU comparison, batch processing |
>
> **The default backend runs on the CPU.** The app shows which ONNX Runtime providers actually ran each model and only says "Snapdragon NPU" if a QNN session was really built on an NPU device.
> **No performance numbers are claimed.** Run `python benchmark.py` on your machine.

## 1. Problem
People photograph or screenshot documents that contain personal details: invoices, IDs, letters, prescriptions. Online OCR and cloud AI tools upload those images to third-party servers and need internet access.

## 2. Solution
Run the OCR model on the user's own PC. The model files ship inside the `rapidocr-onnxruntime` pip package, so after `pip install` the app works fully offline.

## 3. Key features
- Upload PNG/JPG/BMP/WEBP or use the built-in sample image
- Text extraction with boxes drawn on the image (green = confident, red = check this line)
- Low-confidence warning (lines below 0.85 confidence) - a hint only, not a guarantee of correctness
- Quick fields (emails, phones, dates, amounts, links) via regular expressions
- Download extracted text as .txt
- Friendly errors (no file, wrong file type, file too big, no text found, missing packages, unsupported hardware)
- CPU backend plus an opt-in Snapdragon/QNN backend; the UI reports what really ran

## 4. Why on-device AI?
Privacy (documents are never uploaded), works offline, no per-request cost. The trade-off is a smaller model than the largest cloud services.

## 5. Why Snapdragon?
Snapdragon X PCs include an NPU for efficient on-device AI, a natural fit for always-available private tools. This project has **not** measured any NPU benefit; the QNN backend exists so that test can be done honestly.

## 6. AI model used
**PaddleOCR PP-OCRv4** text detection + text recognition + direction classifier (about 16 MB total, ONNX format) via the **RapidOCR** project (`rapidocr-onnxruntime==1.4.4`), run with **ONNX Runtime**. It is not a Qualcomm AI Hub model. It was chosen because the models ship inside the pip package (nothing to download), it is an ONNX model (the format Qualcomm's ONNX Runtime QNN plugin consumes), and it gives a clear, visual demo. Check the package pages for license terms before redistribution.

## 7. Architecture
```
User -> Streamlit UI (app.py)
        -> image validation and resize (size limit, format check)
        -> LocalOCR (ocr_engine.py)
             |- cpu : ONNX Runtime CPUExecutionProvider          (default, tested)
             |- qnn : onnxruntime-qnn plugin, NPU device only    (untested on Snapdragon)
        -> PP-OCRv4 ONNX models (inside the pip package, local)
        -> text lines + boxes + confidence
        -> utils/extract.py (regex fields, not AI)
        -> results displayed in the browser on the same machine
```

## 8. Technology stack
Python 3.11, Streamlit, ONNX Runtime, RapidOCR (PP-OCRv4), Pillow; optional `onnxruntime-qnn`.

## 9. Installation (Windows)
**Use 64-bit (x64) Python 3.11 from python.org, on every laptop, including Snapdragon ones.** On a Snapdragon PC, x64 Python runs under Windows' emulation layer. Reason: on PyPI, `opencv-python`, `shapely`, `pyclipper` and `pyarrow` (needed by RapidOCR and Streamlit) currently publish **no Windows-ARM64 wheels**, so installing with native ARM64 Python would likely fail or need a compiler. (Checked on PyPI on 2026-09-30; check again, this changes.) Running under emulation has **not been tested** by the author.

```bat
git clone <your-repo-url> locallens
cd locallens
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 10. Running
```bat
streamlit run app.py
```
Open http://localhost:8501, click **Use sample image**, or upload your own picture.

## 11. Verify on your machine (2 minutes)
1. `pip install pytest` then `python -m pytest -q tests` (12 tests should pass)
2. `python benchmark.py` (prints real timings for your machine)
3. Run the app and click **Use sample image**

## 12. Snapdragon/QNN setup (optional, untested path)
The NPU path needs a Python that can load Qualcomm's ARM64 libraries, which normally means **native ARM64 Python**, and that conflicts with the dependency problem in section 9 (no ARM64 wheels for OpenCV/shapely/pyclipper/pyarrow). So treat the NPU as a stretch goal: the CPU demo under x64 Python is the safe submission, and any NPU result must be reported exactly as the app shows it. If you try it, on a Snapdragon X Windows-on-ARM PC with ARM64 Python 3.11 (`onnxruntime-qnn` publishes `win_arm64` wheels for Python 3.11):
```bat
pip install -r requirements.txt
pip install -r requirements-snapdragon.txt
python -c "import onnxruntime, onnxruntime_qnn; onnxruntime.register_execution_provider_library(onnxruntime_qnn.get_ep_name(), onnxruntime_qnn.get_library_path()); print([(d.ep_name, str(d.device.type)) for d in onnxruntime.get_ep_devices()])"
```
You want to see a `QNNExecutionProvider` entry with an **NPU** device type. Then:
```bat
python benchmark.py --backend qnn
python benchmark.py --backend cpu
```
and compare your own numbers. Notes:
- `onnxruntime-qnn` 2.x is a **plugin for onnxruntime**; do not uninstall onnxruntime.
- PP-OCRv4 models use dynamic input shapes. NPUs generally prefer fixed shapes or quantized models, so ONNX Runtime may keep some or all work on CPU. The app's "Execution" line and `describe()` output tell you which.
- If nothing runs on the NPU, say so in your submission; it is still a legitimate on-device, offline result.
- `LOCALLENS_QNN_ANY_DEVICE=1` is a developer switch that lets the QNN plugin attach to a non-NPU device for testing. Do not use it for claims.

## 13. Benchmarking
```bat
python benchmark.py                               # CPU, 5 timed runs after a warm-up
python benchmark.py --backend qnn                 # Snapdragon only
python benchmark.py --image my.png --runs 10 --output results.json
```

## 14. Privacy considerations
- Images are processed in memory; the app writes nothing to disk except when you click Download.
- The app makes no network requests while reading a document (verified by blocking sockets in a test run). Install-time package downloads do use the internet.
- Streamlit usage statistics are disabled and the server listens on localhost only (`.streamlit/config.toml`).
- These statements are about this app, not about other software on your PC.

## 15. Limitations
- OCR is not perfect. On the clean sample invoice all text and all field types were found, but the model dropped some spaces (for example `Phone:+919876543210`). On an earlier test image drawn in a stylized font it misread the letter O as a zero (`INV0ICE`, `30 0ct 2026`) **with confidence between 0.93 and 0.99**, so the confidence score does not reliably catch mistakes. Always review the output.
- Mainly printed text; handwriting and heavily skewed or blurry photos work poorly.
- Quick fields use simple patterns and will miss unusual formats.
- Not tested on Windows or Snapdragon by the author; NPU use is unproven.

## 16. Future improvements
Quantized/fixed-shape models for the NPU, measured CPU vs NPU comparison, PDF support, redaction of detected fields, batch mode.

## 17. Demo instructions
See `docs/DEMO_SCRIPT.md`.

## 18. Troubleshooting
| Problem | Fix |
|---|---|
| `pip install` fails building opencv/shapely/pyclipper/pyarrow | You are on ARM64 Python. Install x64 Python 3.11 instead (section 9) |
| "Required packages are missing" | Activate the venv, `pip install -r requirements.txt` |
| `streamlit` not found | Activate the venv or run `python -m streamlit run app.py` |
| "No text was found" | Use a sharper, well-lit, higher-resolution image |
| QNN: "onnxruntime-qnn is not installed" | `pip install -r requirements-snapdragon.txt` |
| QNN: "No Snapdragon NPU device was found" | Expected on non-Snapdragon PCs; use the CPU backend |
| UI says CPU although QNN was selected | ONNX Runtime fell back to CPU (see section 12) - report it honestly |
| Port in use | `streamlit run app.py --server.port 8502` |

## License
MIT (code). See the license terms of RapidOCR / PaddleOCR models you redistribute.
