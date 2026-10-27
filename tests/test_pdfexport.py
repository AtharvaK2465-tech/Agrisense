"""
Testing PDF Export generation for valid outputs, word-wrapping, Unicode sanitization,
missing optional indices, and report validation notices.
"""
import pytest
from utils.pdfexport import build_pdf


def test_pdf_export_valid_output():
    report_md = """## Crop Health Report — Selected Area
**Diagnosis:** Healthy Canopy
**Risk level:** LOW

### Observations
Vegetation indices indicate high canopy vigor across the parcel.

### Recommendation
Continue routine field scouting and standard cultivation practices."""
    meta = {
        "district": "Akola",
        "season": "kharif",
        "crop_type": "Cotton",
        "generated_at": "2026-10-28T02:00:00",
    }
    pdf_bytes = build_pdf(report_md, meta)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF")


def test_pdf_export_handles_unicode_and_emojis():
    """PDF generator must safely handle Unicode characters, curly quotes,
    emojis, and dashes without raising Latin-1 encoding errors."""
    unicode_report = """## 🌾 Crop Health Report — महाराष्ट्र (Maharashtra)
*“Smart Agriculture”* — anomaly observed in cotton: −0.12° NDVI.
- Status: 🟢 Healthy (vigor > 70%)
- Unicode test: नमस्कार, शेती अहवाल."""
    meta = {"district": "पुणे (Pune)", "season": "खरीप (Kharif)"}
    pdf_bytes = build_pdf(unicode_report, meta)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")


def test_pdf_export_handles_long_recommendations():
    """Verifies word wrapping on long paragraphs without overflow exceptions."""
    long_text = "Word " * 300
    report_md = f"## Long Report\n\n### Recommendation\n{long_text}"
    pdf_bytes = build_pdf(report_md, {"district": "Nagpur"})
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 800


def test_pdf_export_with_missing_optional_indices():
    report_md = """## Crop Health Report
- NDVI: Calculated (0.65)
- NDMI: Unavailable (SWIR band not available in 4-band imagery)
- NDRE: Unavailable (Red-Edge band not available in imagery)
### Recommendation
Field inspection recommended."""
    pdf_bytes = build_pdf(report_md, {"district": "Amravati"})
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
