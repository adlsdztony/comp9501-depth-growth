"""Build the results deck from rendered figures. usage: python build_deck.py FIG_DIR OUT.pptx deck.json"""
import json, sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from PIL import Image

FIG, OUT, SPEC = Path(sys.argv[1]), sys.argv[2], json.loads(Path(sys.argv[3]).read_text(encoding='utf8'))
NAVY, INK, INK2, WHITE, ORANGE = RGBColor(0x14, 0x21, 0x3D), RGBColor(0x0B, 0x0B, 0x0B), RGBColor(0x52, 0x51, 0x4E), RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0xEB, 0x68, 0x34)
FONT = 'Calibri'
W, H = Inches(13.333), Inches(7.5)
prs = Presentation(); prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]


def bg(slide, color):
    f = slide.background.fill; f.solid(); f.fore_color.rgb = color


def text(slide, s, x, y, w, h, size, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False):
    tb = slide.shapes.add_textbox(x, y, w, h); tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0; tf.vertical_anchor = anchor
    lines = s if isinstance(s, list) else [s]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = align
        if i: p.space_before = Pt(10)
        r = p.add_run(); r.text = line; f = r.font
        f.size, f.bold, f.italic, f.name = Pt(size), bold, italic, FONT; f.color.rgb = color
    return tb


def picture(slide, path, x, y, maxw, maxh):
    im = Image.open(path); ar = im.width / im.height
    w, h = maxw, int(maxw / ar)
    if h > maxh: h, w = maxh, int(maxh * ar)
    slide.shapes.add_picture(str(path), x + (maxw - w) // 2, y + (maxh - h) // 2, w, h)


M = Inches(0.6)
for sp in SPEC['slides']:
    s = prs.slides.add_slide(BLANK)
    kind = sp.get('kind', 'figure')
    if kind in ('title', 'closing'):
        bg(s, NAVY)
        if kind == 'title':
            text(s, sp['title'], M, Inches(2.3), W - 2 * M, Inches(1.4), 48, WHITE, True)
            text(s, sp.get('subtitle', ''), M, Inches(3.8), W - 2 * M, Inches(0.8), 22, RGBColor(0xCA, 0xDC, 0xFC))
            text(s, sp.get('footer', ''), M, Inches(6.4), W - 2 * M, Inches(0.5), 14, RGBColor(0xB7, 0xC0, 0xD0))
        else:
            text(s, sp['title'], M, Inches(0.7), W - 2 * M, Inches(0.9), 36, WHITE, True)
            y = Inches(2.0)
            for i, (num, line) in enumerate(sp['points']):
                text(s, num, M, y, Inches(1.0), Inches(0.9), 40, ORANGE, True)
                text(s, line, M + Inches(1.1), y + Inches(0.08), W - 2 * M - Inches(1.1), Inches(0.9), 24, WHITE)
                y += Inches(1.35)
    else:
        bg(s, WHITE)
        text(s, sp['title'], M, Inches(0.45), W - 2 * M, Inches(0.9), 32, INK, True)
        top = Inches(1.45); bottom = Inches(0.55) if not sp.get('caption') else Inches(1.0)
        figs = sp['figures']
        stats = sp.get('stats')
        area_w = W - 2 * M - (Inches(3.3) if stats else 0)
        gap = Inches(0.3); n = len(figs); fw = (area_w - gap * (n - 1)) // n
        for i, f in enumerate(figs):
            picture(s, FIG / f, M + i * (fw + gap), top, fw, H - top - bottom)
        if stats:
            x = W - M - Inches(3.0); y = top + Inches(0.4)
            for big, small in stats:
                text(s, big, x, y, Inches(3.0), Inches(0.9), 44, ORANGE if big == stats[0][0] else NAVY, True)
                text(s, small, x, y + Inches(0.85), Inches(3.0), Inches(0.5), 15, INK2)
                y += Inches(1.6)
        if sp.get('caption'):
            text(s, sp['caption'], M, H - Inches(0.85), W - 2 * M, Inches(0.5), 14, INK2)
    if sp.get('notes'):
        s.notes_slide.notes_text_frame.text = sp['notes']
prs.save(OUT)
print('saved', OUT, len(prs.slides), 'slides')
