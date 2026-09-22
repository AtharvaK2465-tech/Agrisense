"""
Turns a report_markdown string + a bit of metadata into a downloadable
PDF, so the app's output is a shareable deliverable rather than just a
webpage. Deliberately simple (no external markdown-to-PDF dependency
beyond fpdf2) — good enough for the headers/bullets/paragraphs the
Report Agent actually produces.

Does its own word-wrapping with cell() rather than relying on fpdf2's
multi_cell() auto-wrap, which has a known bug in some fpdf2 versions
("Not enough horizontal space to render a single character",
py-pdf/fpdf2#1250 / #1831, fixed in 2.8.8+) that can raise even on
short, plain ASCII lines. Wrapping manually sidesteps it regardless of
which fpdf2 version is installed.
"""
from fpdf import FPDF


def _clean(text: str) -> str:
    # Core PDF fonts are latin-1 only; strip characters that would break
    # rendering (emoji, curly quotes from LLM output, etc.) rather than crash.
    return text.replace("’", "'").replace("‘", "'").replace("—", "-").encode(
        "latin-1", "ignore"
    ).decode("latin-1")


def _write_wrapped(pdf: FPDF, text: str, line_height: float = 6) -> None:
    """Writes `text` to the current position, wrapping at word boundaries
    to fit within the page's printable width, using cell() line-by-line
    instead of multi_cell()'s internal (occasionally buggy) wrapper."""
    if not text:
        pdf.ln(line_height)
        return

    effective_width = pdf.w - pdf.l_margin - pdf.r_margin
    line = ""
    for word in text.split(" "):
        candidate = f"{line} {word}".strip()
        if pdf.get_string_width(candidate) <= effective_width or not line:
            line = candidate
        else:
            pdf.cell(0, line_height, line, new_x="LMARGIN", new_y="NEXT")
            line = word

        # A single word wider than the whole line (long URL/token) — hard
        # break it character by character so we never hand cell() text
        # that can't fit, whatever the wrap outcome above was.
        while pdf.get_string_width(line) > effective_width and len(line) > 1:
            cut = len(line)
            while cut > 1 and pdf.get_string_width(line[:cut]) > effective_width:
                cut -= 1
            pdf.cell(0, line_height, line[:cut], new_x="LMARGIN", new_y="NEXT")
            line = line[cut:]

    if line:
        pdf.cell(0, line_height, line, new_x="LMARGIN", new_y="NEXT")


def build_pdf(report_markdown: str, meta: dict) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    _write_wrapped(pdf, _clean("Crop Health Report - Maharashtra"), line_height=10)

    pdf.set_font("Helvetica", "", 10)
    meta_line = (
        f"District: {meta.get('district', '-')}   "
        f"Season: {meta.get('season', '-')}   "
        f"Crop type: {meta.get('crop_type', '-')}   "
        f"Generated: {meta.get('generated_at', '-')}"
    )
    _write_wrapped(pdf, _clean(meta_line), line_height=6)
    pdf.ln(4)

    for raw_line in report_markdown.splitlines():
        line = raw_line.strip()
        if not line:
            pdf.ln(3)
            continue
        if line.startswith("### "):
            pdf.set_font("Helvetica", "B", 12)
            _write_wrapped(pdf, _clean(line[4:]), line_height=7)
            pdf.set_font("Helvetica", "", 10)
        elif line.startswith("## "):
            pdf.set_font("Helvetica", "B", 14)
            _write_wrapped(pdf, _clean(line[3:]), line_height=8)
            pdf.set_font("Helvetica", "", 10)
        elif line.startswith("- "):
            _write_wrapped(pdf, _clean(f"  - {line[2:]}"), line_height=6)
        elif line.startswith("*") and line.endswith("*") and len(line) > 1:
            pdf.set_font("Helvetica", "I", 9)
            _write_wrapped(pdf, _clean(line.strip("*")), line_height=6)
            pdf.set_font("Helvetica", "", 10)
        else:
            _write_wrapped(pdf, _clean(line.replace("**", "")), line_height=6)

    out = pdf.output(dest="S")
    return bytes(out)
