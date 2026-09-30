"""
LocalLens - Streamlit UI.

Flow: upload image -> local OCR model (ONNX Runtime) -> text + boxes + extracted fields.
Run with:  streamlit run app.py
"""

import io
import os
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, UnidentifiedImageError

from ocr_engine import LocalOCR, OCRError
from utils.extract import extract_fields

SAMPLE_PATH = Path(__file__).parent / "sample_data" / "sample_invoice.png"
MAX_FILE_MB = 10
MAX_SIDE_PX = 2400          # larger images are shrunk first so processing stays fast
LOW_CONFIDENCE = 0.85

st.set_page_config(page_title="LocalLens", page_icon="🔒", layout="wide")

st.markdown(
    """
    <style>
      .block-container { max-width: 1150px; padding-top: 2rem; }
      .subtitle { color: #5b6472; font-size: 1.1rem; margin-top: -0.6rem; margin-bottom: 1.2rem; }
      .card { border: 1px solid #e3e7ee; border-radius: 12px; padding: 1rem 1.2rem;
              background: #ffffff; margin-bottom: 1rem; color: #1c2430; }
      .card h3 { margin: 0 0 0.6rem 0; font-size: 0.95rem; letter-spacing: 0.06em; color: #2b5fd9; }
      .badge { display: inline-block; background: #e8f5ec; color: #1e6b3a; border-radius: 999px;
               padding: 0.2rem 0.75rem; margin: 0 0.4rem 0.4rem 0; font-size: 0.85rem; }
      .chip { display: inline-block; background: #eef2fb; color: #1f3f99; border-radius: 6px;
              padding: 0.15rem 0.55rem; margin: 0 0.35rem 0.35rem 0; font-size: 0.9rem; }
      .muted { color: #5b6472; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


@st.cache_resource(show_spinner=False)
def get_engine(backend: str) -> LocalOCR:
    """Create the OCR engine once; the loaded models stay in memory between reruns."""
    engine = LocalOCR(backend)
    engine.load()
    return engine


def prepare_image(data: bytes) -> Image.Image:
    """Validate and open the uploaded bytes. Raises ValueError with a friendly message."""
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise ValueError(f"The file is larger than {MAX_FILE_MB} MB. Please use a smaller image.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise ValueError("That file could not be read as an image. Use PNG, JPG, BMP or WEBP.")
    image = image.convert("RGB")
    if max(image.size) > MAX_SIDE_PX:
        image.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX))
    return image


def draw_boxes(image: Image.Image, lines) -> Image.Image:
    annotated = image.copy()
    drawer = ImageDraw.Draw(annotated)
    for line in lines:
        points = [tuple(p) for p in line.box]
        colour = (220, 60, 60) if line.confidence < LOW_CONFIDENCE else (30, 140, 70)
        drawer.polygon(points, outline=colour)
    return annotated


def clear_all() -> None:
    for key in ("result", "error", "image_bytes"):
        st.session_state.pop(key, None)
    st.session_state["upload_version"] = st.session_state.get("upload_version", 0) + 1


def run_ocr(data: bytes, backend: str) -> None:
    st.session_state.pop("result", None)
    st.session_state.pop("error", None)
    try:
        image = prepare_image(data)
        with st.spinner("Loading the local OCR model (first run only) and reading the document..."):
            engine = get_engine(backend)
            result = engine.run(image)
        if not result.lines:
            st.session_state["error"] = "No text was found in this image. Try a sharper, well-lit picture."
            return
        st.session_state["result"] = {
            "image": image, "ocr": result, "info": engine.describe(), "load_seconds": engine.load_seconds,
        }
    except (ValueError, OCRError) as err:
        st.session_state["error"] = str(err)
    except Exception:  # noqa: BLE001 - last-resort guard: users never see a stack trace
        st.session_state["error"] = "Something unexpected went wrong while reading the image. Please try again."


def render_badges(info: dict) -> None:
    badges = [
        "✓ Processed locally on this device",
        "✓ No cloud API used",
        "✓ Your document was not uploaded anywhere by this app",
    ]
    st.markdown("".join(f'<span class="badge">{b}</span>' for b in badges), unsafe_allow_html=True)
    providers = info.get("providers_per_model", {})
    where = "; ".join(f"{name}: {', '.join(p)}" for name, p in providers.items())
    npu = "Snapdragon NPU requested and used by ONNX Runtime" if info.get("runs_on_npu") else "CPU (no NPU in use)"
    st.markdown(
        f'<div class="muted">Model: {escape(info["model"])} &middot; Runtime: {info["runtime"]} &middot; '
        f"Execution: {escape(npu)} &middot; ONNX Runtime providers &rarr; {escape(where)}</div>",
        unsafe_allow_html=True,
    )


def render_results(result: dict) -> None:
    ocr = result["ocr"]
    st.markdown("### Results")
    render_badges(result["info"])

    left, right = st.columns(2)
    with left:
        st.markdown("**Detected text regions** (green = confident, red = check this line)")
        st.image(draw_boxes(result["image"], ocr.lines), use_container_width=True)
    with right:
        st.markdown("**Extracted text**")
        st.text_area("Text", ocr.text, height=260, label_visibility="collapsed")
        st.download_button("Download as .txt", ocr.text, file_name="locallens_text.txt")

    fields = extract_fields(ocr.text)
    body = "".join(
        f"<div><b>{escape(label)}:</b> " + "".join(f'<span class="chip">{escape(v)}</span>' for v in values) + "</div>"
        for label, values in fields.items()
    ) or "No emails, phones, dates, amounts or links found."
    st.markdown(
        f'<div class="card"><h3>QUICK FIELDS (pattern matching, not AI)</h3>{body}</div>',
        unsafe_allow_html=True,
    )

    doubtful = [line for line in ocr.lines if line.confidence < LOW_CONFIDENCE]
    if doubtful:
        st.warning("Low-confidence lines, please double-check: " + " | ".join(l.text for l in doubtful))

    st.caption(
        f"Read {len(ocr.lines)} lines in {ocr.seconds:.2f} s on this machine "
        f"(model load {result['load_seconds']:.2f} s, first run only)."
    )


# ---------- Page ----------

st.title("LocalLens")
st.markdown(
    '<div class="subtitle">Read documents with AI, privately. Your images never leave this device.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings")
    choices = ["cpu", "qnn"]
    default = os.environ.get("LOCALLENS_BACKEND", "cpu")
    backend = st.selectbox(
        "Inference backend", choices, index=choices.index(default) if default in choices else 0,
        help="cpu: works anywhere. qnn: Snapdragon NPU via ONNX Runtime (needs onnxruntime-qnn).",
    )
    st.caption("The page always reports what ONNX Runtime actually used. It never claims NPU use on CPU.")

upload = st.file_uploader(
    "Upload a photo or screenshot of a document",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    key=f"uploader_{st.session_state.get('upload_version', 0)}",
)

col_a, col_b, col_c, _ = st.columns([1.2, 1.3, 1, 3])
read_clicked = col_a.button("Read document", type="primary", use_container_width=True)
sample_clicked = col_b.button("Use sample image", use_container_width=True)
col_c.button("Clear", on_click=clear_all, use_container_width=True)

if sample_clicked:
    st.session_state["image_bytes"] = SAMPLE_PATH.read_bytes()
    run_ocr(st.session_state["image_bytes"], backend)
elif read_clicked:
    if upload is None:
        st.session_state.pop("result", None)
        st.session_state["error"] = "Please upload an image first, or click 'Use sample image'."
    else:
        run_ocr(upload.getvalue(), backend)

if "error" in st.session_state:
    st.error(st.session_state["error"])
if "result" in st.session_state:
    render_results(st.session_state["result"])
