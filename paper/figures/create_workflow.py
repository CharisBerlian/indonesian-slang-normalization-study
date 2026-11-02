"""Draw the proposed study workflow as a vector PDF and a preview PNG.

Run with Python containing reportlab and pypdfium2. The manuscript embeds
the vector PDF; the PNG is a preview only. All stages describe planned work.
"""
from pathlib import Path
import math
import shutil

from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pypdfium2 as pdfium


pdfmetrics.registerFont(TTFont("WorkflowSans", "C:/Windows/Fonts/arial.ttf"))
pdfmetrics.registerFont(TTFont("WorkflowSans-Bold", "C:/Windows/Fonts/arialbd.ttf"))

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUTPUT = ROOT / "output" / "pdf"
OUTPUT.mkdir(parents=True, exist_ok=True)
PDF = HERE / "proposed_workflow.pdf"
W, H = 252, 406
c = canvas.Canvas(str(PDF), pagesize=(W, H), initialFontName="WorkflowSans")
c.setTitle("Proposed workflow: Indonesian slang normalization study")
c.setAuthor("Research manuscript working draft")

INK = HexColor("#222B33")
EDGE = HexColor("#697780")
ARROW = HexColor("#354BD1")
BLUE = "#BAE0EF"
GREEN = "#BEECC0"
CYAN = "#E0F5F5"
YELLOW = "#FFF6CC"
GRAY = "#ECEDEF"
RED = "#F4B7B7"
OUTPUT_BLUE = "#A7D7F4"


def box(x, y, w, h, lines, fill, size=8.2, bold_first=False):
    c.setStrokeColor(EDGE)
    c.setFillColor(HexColor(fill))
    c.setLineWidth(0.55)
    c.roundRect(x, y, w, h, 1.5, stroke=1, fill=1)
    leading = size + 2.0
    baseline = y + h / 2 + (len(lines) - 1) * leading / 2 - size * 0.34
    c.setFillColor(INK)
    for i, line in enumerate(lines):
        font = "WorkflowSans-Bold" if bold_first and i == 0 else "WorkflowSans"
        c.setFont(font, size)
        if c.stringWidth(line, font, size) > w - 8:
            raise ValueError(f"Workflow label too wide: {line}")
        c.drawCentredString(x + w / 2, baseline - i * leading, line)


def arrow(points):
    c.setStrokeColor(ARROW)
    c.setFillColor(ARROW)
    c.setLineWidth(0.75)
    p = c.beginPath()
    p.moveTo(*points[0])
    for point in points[1:]:
        p.lineTo(*point)
    c.drawPath(p)
    a, b = points[-2], points[-1]
    angle = math.atan2(b[1] - a[1], b[0] - a[0])
    length, halfwidth = 4.2, 1.9
    base = (b[0] - length * math.cos(angle), b[1] - length * math.sin(angle))
    p = c.beginPath()
    p.moveTo(*b)
    p.lineTo(base[0] - halfwidth * math.sin(angle), base[1] + halfwidth * math.cos(angle))
    p.lineTo(base[0] + halfwidth * math.sin(angle), base[1] - halfwidth * math.cos(angle))
    p.close()
    c.drawPath(p, stroke=0, fill=1)


# Resource inputs feed the audit: normalized equivalence is needed for grouping.
box(4, 364, 117, 38, ["Public labeled reviews", "PRDECT-ID / SmSA"], BLUE, bold_first=True)
box(131, 364, 117, 38, ["Kamus Alay", "Pin dictionary and rules"], YELLOW, bold_first=True)
box(22, 300, 208, 44, ["Minimal cleanup and data audit", "Group O/N-equivalent reviews", "Freeze train / validation / test splits"], GREEN, bold_first=True)
arrow([(62.5, 364), (62.5, 354), (94, 354), (94, 344)])
arrow([(189.5, 364), (189.5, 354), (158, 354), (158, 344)])

box(4, 242, 117, 38, ["Original condition (O)", "Minimal cleanup only"], CYAN, bold_first=True)
box(131, 242, 117, 38, ["Normalized condition (N)", "Cleanup + substitution"], YELLOW, size=7.9, bold_first=True)
arrow([(94, 300), (94, 290), (62.5, 290), (62.5, 280)])
arrow([(158, 300), (158, 290), (189.5, 290), (189.5, 280)])

# The enclosing panel represents repeated independent experiments, not an ensemble.
c.setFillColor(HexColor(GRAY))
c.setStrokeColor(EDGE)
c.setLineWidth(0.55)
c.roundRect(4, 164, 244, 58, 1.5, fill=1, stroke=1)
c.setFillColor(INK)
c.setFont("WorkflowSans-Bold", 8.2)
c.drawCentredString(126, 208, "Independent classifiers under each condition")
box(10, 172, 72, 27, ["Word TF-IDF", "+ linear SVM"], CYAN, size=7.8)
box(90, 172, 72, 27, ["Character TF-IDF", "+ linear SVM"], CYAN, size=7.8)
box(170, 172, 72, 27, ["IndoBERT", "fine-tuning"], CYAN, size=7.8)
arrow([(62.5, 242), (62.5, 222)])
arrow([(189.5, 242), (189.5, 222)])

box(22, 112, 208, 32, ["Train and select models", "Validation macro-F1; equal tuning budgets"], GRAY, size=8.0, bold_first=True)
arrow([(126, 164), (126, 144)])
box(22, 60, 208, 32, ["Held-out sentiment evaluation", "Matched pipelines + fixed-model input changes"], RED, size=8.0, bold_first=True)
arrow([(126, 112), (126, 92)])
box(22, 4, 208, 36, ["Normalization effect and error analysis", "Paired intervals; seed variability", "Corrected predictions / introduced errors"], OUTPUT_BLUE, size=8.0, bold_first=True)
arrow([(126, 60), (126, 40)])

c.showPage()
c.save()
shutil.copy2(PDF, OUTPUT / PDF.name)
doc = pdfium.PdfDocument(str(PDF))
doc[0].render(scale=3).to_pil().save(str(HERE / "proposed_workflow.png"))
print(f"Workflow generated: {PDF}")
