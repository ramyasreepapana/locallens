"""
Tests. The OCR tests run the REAL model on the sample image (no stubs).
Run:  python -m pytest -q tests
"""

from pathlib import Path

import pytest
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

from ocr_engine import LocalOCR, OCRError
from utils.extract import extract_fields

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "sample_data" / "sample_invoice.png"


# ---------- field extraction (pure Python) ----------

def test_extract_basic_fields():
    text = "Mail a.b@x.org or call +91 98765 43210 on 14/10/2026. Total Rs 12,450.00 see https://x.org/p"
    fields = extract_fields(text)
    assert fields["Emails"] == ["a.b@x.org"]
    assert fields["Phone numbers"] == ["+91 98765 43210"]
    assert fields["Dates"] == ["14/10/2026"]
    assert fields["Amounts"] == ["Rs 12,450.00"]
    assert fields["Links"] == ["https://x.org/p"]


def test_extract_date_is_not_a_phone_number():
    assert "Phone numbers" not in extract_fields("Due 14/10/2026")


def test_extract_nothing_found():
    assert extract_fields("hello world") == {}


# ---------- real OCR ----------

@pytest.fixture(scope="module")
def ocr_result():
    engine = LocalOCR("cpu")
    return engine, engine.run(Image.open(SAMPLE))


def test_real_ocr_reads_sample(ocr_result):
    _, result = ocr_result
    assert len(result.lines) >= 5
    assert "anita.rao@example.com" in result.text
    assert "12,450.00" in result.text


def test_real_ocr_reports_cpu_providers(ocr_result):
    engine, _ = ocr_result
    info = engine.describe()
    assert info["runs_on_npu"] is False
    assert all(p == ["CPUExecutionProvider"] for p in info["providers_per_model"].values())


def test_blank_image_gives_no_lines():
    blank = Image.new("RGB", (400, 200), "white")
    assert LocalOCR("cpu").run(blank).lines == []


def test_qnn_refuses_when_unavailable():
    # Without onnxruntime-qnn (or without an NPU device) the QNN backend must refuse, never fake it.
    with pytest.raises(OCRError, match="not installed|No Snapdragon NPU"):
        LocalOCR("qnn").load()


def test_unknown_backend_rejected():
    with pytest.raises(OCRError):
        LocalOCR("gpu")


# ---------- UI (headless Streamlit) ----------

def run_app():
    st.cache_resource.clear()
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()


def click(at, label):
    next(b for b in at.button if b.label == label).click()
    return at.run()


def test_ui_loads():
    at = run_app()
    assert not at.exception
    assert at.title[0].value == "LocalLens"


def test_ui_read_without_upload_is_friendly():
    at = click(run_app(), "Read document")
    assert not at.exception
    assert "upload an image first" in at.error[0].value


def test_ui_sample_image_end_to_end():
    at = click(run_app(), "Use sample image")
    assert not at.exception
    assert not at.error
    html = " ".join(m.value for m in at.markdown)
    assert "Processed locally" in html
    assert "CPU (no NPU in use)" in html
    assert "anita.rao@example.com" in html          # extracted field shown


def test_ui_clear_resets():
    at = click(run_app(), "Use sample image")
    at = click(at, "Clear")
    assert not at.exception
    assert not at.error
