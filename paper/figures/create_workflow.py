"""Draw the proposed study workflow as a vector PDF and a preview PNG.

Run with Python containing reportlab and pypdfium2. The manuscript embeds
the vector PDF; the PNG is a preview only. Stages match methodology A--E.
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
W, H = 252, 270
c = canvas.Canvas(str(PDF), pagesize=(W, H), initialFontName="WorkflowSans")
c.setTitle("Study workflow: paired pipelines and fixed-model input switches")
c.setAuthor("Charis Berlian")

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


# The dictionary is fixed before the audit because normalized equivalence
# participates in group formation. These are shared splits, not two data samples.
box(4, 222, 244, 44, [
    "1. Audit and shared splits (III-A)",
    "PRDECT-ID / SmSA; Kamus Alay fixed first",
    "Group original / normalized equivalents",
    "Freeze train / validation / test splits",
], GREEN, size=8.0, bold_first=True)
arrow([(126, 222), (126, 206)])

c.setFillColor(HexColor(GRAY))
c.setStrokeColor(EDGE)
c.setLineWidth(0.55)
c.roundRect(4, 154, 244, 52, 1.5, fill=1, stroke=1)
c.setFillColor(INK)
c.setFont("WorkflowSans-Bold", 8.2)
c.drawCentredString(126, 192, "2. Paired input conditions (III-B)")
box(10, 160, 111, 24, ["Original (O)", "Minimal cleanup"], CYAN,
    size=8.0, bold_first=True)
box(131, 160, 111, 24, ["Normalized (N)", "Cleanup + substitution"], YELLOW,
    size=8.0, bold_first=True)
arrow([(126, 154), (126, 142)])

# Independent classifiers per condition; this panel does not combine predictions.
box(4, 88, 244, 54, [
    "3. Train and select models (III-C)",
    "Word SVM / Character SVM / IndoBERT",
    "Train each model separately under O and N",
    "Select using validation macro-F1",
    "Freeze models before test evaluation",
], BLUE, size=8.0, bold_first=True)

# Two analyses of the same frozen models/test reviews, not sequential training
# stages and not an ensemble. Stage 5 retains each model's fitted parameters.
arrow([(126, 88), (126, 80), (62.5, 80), (62.5, 72)])
arrow([(126, 88), (126, 80), (189.5, 80), (189.5, 72)])
box(4, 4, 117, 68, [
    "4. Pipelines (III-D)",
    "N/N versus O/O",
    "Macro-F1 differences",
    "Paired intervals",
], OUTPUT_BLUE, size=8.0, bold_first=True)
box(131, 4, 117, 68, [
    "5. Input switches (III-E)",
    "O/O to O/N; N/N to N/O",
    "Macro-F1 changes",
    "Corrections and errors",
], RED, size=8.0, bold_first=True)

c.showPage()
c.save()
shutil.copy2(PDF, OUTPUT / PDF.name)
doc = pdfium.PdfDocument(str(PDF))
doc[0].render(scale=3).to_pil().save(str(HERE / "proposed_workflow.png"))
print(f"Workflow generated: {PDF}")
