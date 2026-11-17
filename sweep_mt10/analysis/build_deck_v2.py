"""Richer results deck. usage: python build_deck_v2.py FIG_DIR OUT.pptx deck_v2.json

Slide kinds: title | closing | split (figure + bullets) | figure (full-width figure + bullets below)
             | cards (2-3 text cards) | table (table + optional figure).
Every content slide may carry `takeaway` (one bold line in a tinted box at the bottom) and `notes`.
"""
import json, sys, copy
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from lxml import etree
from PIL import Image

FIG, OUT, SPEC = Path(sys.argv[1]), sys.argv[2], json.loads(Path(sys.argv[3]).read_text(encoding='utf8'))
NAVY = RGBColor(0x14, 0x21, 0x3D); INK = RGBColor(0x1A, 0x1A, 0x1A); INK2 = RGBColor(0x52, 0x51, 0x4E)
WHITE = RGBColor(0xFF, 0xFF, 0xFF); ORANGE = RGBColor(0xEB, 0x68, 0x34); TINT = RGBColor(0xF1, 0xF4, 0xF9)
ICE = RGBColor(0xCA, 0xDC, 0xFC); MUTED = RGBColor(0xB7, 0xC0, 0xD0)
FONT = 'Calibri'
W, H = Inches(13.333), Inches(7.5)
M = Inches(0.6)
prs = Presentation(); prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]


def bg(slide, color):
    f = slide.background.fill; f.solid(); f.fore_color.rgb = color


def set_bullet(p, char='•', indent=Inches(0.28)):
    pPr = p._p.get_or_add_pPr()
    pPr.set('marL', str(int(indent))); pPr.set('indent', str(-int(indent)))
    for tag in ('a:buNone', 'a:buChar', 'a:buAutoNum'):
        for e in pPr.findall(qn(tag)): pPr.remove(e)
    bu = etree.SubElement(pPr, qn('a:buChar')); bu.set('char', char)


def runs(p, text, size, color, bold=False, italic=False):
    """Supports **bold** segments inside a line."""
    parts = text.split('**')
    for i, seg in enumerate(parts):
        if not seg: continue
        r = p.add_run(); r.text = seg; f = r.font
        f.size, f.name, f.italic = Pt(size), FONT, italic
        f.bold = bold or (i % 2 == 1); f.color.rgb = color


def textbox(slide, lines, x, y, w, h, size, color=INK, bold=False, bullets=False, align=PP_ALIGN.LEFT,
            anchor=MSO_ANCHOR.TOP, space=8, italic=False):
    tb = slide.shapes.add_textbox(x, y, w, h); tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0; tf.vertical_anchor = anchor
    lines = lines if isinstance(lines, list) else [lines]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = align
        p.space_after = Pt(space)
        sub = bullets and line.startswith('- ')
        if sub: line = line[2:]
        runs(p, line, size - (2 if sub else 0), INK2 if sub else color, bold, italic)
        if bullets: set_bullet(p, '–' if sub else '•', Inches(0.55) if sub else Inches(0.28))
    return tb


def picture(slide, path, x, y, maxw, maxh, align='center'):
    im = Image.open(path); ar = im.width / im.height
    w, h = maxw, int(maxw / ar)
    if h > maxh: h, w = maxh, int(maxh * ar)
    xx = x + (maxw - w) // 2 if align == 'center' else x
    slide.shapes.add_picture(str(path), xx, y + (maxh - h) // 2, w, h)


def rect(slide, x, y, w, h, color, rounded=True):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE, x, y, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = color; s.line.fill.background(); s.shadow.inherit = False
    if rounded: s.adjustments[0] = 0.06
    return s


def title(slide, t, kicker=None):
    if kicker:
        textbox(slide, kicker.upper(), M, Inches(0.35), W - 2 * M, Inches(0.3), 12, ORANGE, True)
    textbox(slide, t, M, Inches(0.62), W - 2 * M, Inches(0.8), 28, INK, True)


def takeaway(slide, text):
    y = H - Inches(1.05)
    rect(slide, M, y, W - 2 * M, Inches(0.62), TINT)
    textbox(slide, text, M + Inches(0.25), y, W - 2 * M - Inches(0.5), Inches(0.62), 17, NAVY, True, anchor=MSO_ANCHOR.MIDDLE)


TOP = Inches(1.55)
for sp in SPEC['slides']:
    s = prs.slides.add_slide(BLANK)
    kind = sp.get('kind', 'split')
    if kind == 'title':
        bg(s, NAVY)
        textbox(s, sp['title'], M, Inches(2.0), W - 2 * M, Inches(1.2), 46, WHITE, True)
        textbox(s, sp['subtitle'], M, Inches(3.3), Inches(10.5), Inches(1.0), 22, ICE)
        textbox(s, sp.get('footer', ''), M, Inches(6.3), W - 2 * M, Inches(0.5), 14, MUTED)
    elif kind == 'closing':
        bg(s, NAVY)
        textbox(s, sp['title'], M, Inches(0.6), W - 2 * M, Inches(0.8), 34, WHITE, True)
        y = Inches(1.75)
        for num, head, body in sp['points']:
            textbox(s, num, M, y, Inches(0.8), Inches(0.8), 36, ORANGE, True)
            textbox(s, head, M + Inches(0.9), y + Inches(0.02), Inches(11), Inches(0.5), 22, WHITE, True)
            textbox(s, body, M + Inches(0.9), y + Inches(0.5), Inches(11), Inches(0.6), 16, ICE)
            y += Inches(1.3)
        if sp.get('footer'):
            textbox(s, sp['footer'], M, H - Inches(0.8), W - 2 * M, Inches(0.4), 14, MUTED)
    else:
        bg(s, WHITE)
        title(s, sp['title'], sp.get('kicker'))
        bottom = H - (Inches(1.25) if sp.get('takeaway') else Inches(0.5))
        avail = bottom - TOP
        if kind == 'split':
            fw = int((W - 2 * M) * sp.get('fig_frac', 0.6))
            figs = sp['figures']
            if len(figs) == 1:
                picture(s, FIG / figs[0], M, TOP, fw, avail)
            else:
                hh = (avail - Inches(0.2)) // len(figs)
                for i, f in enumerate(figs):
                    picture(s, FIG / f, M, TOP + i * (hh + Inches(0.2)), fw, hh)
            tx = M + fw + Inches(0.4)
            textbox(s, sp['bullets'], tx, TOP + Inches(0.1), W - M - tx, avail, sp.get('size', 17), INK, bullets=True, space=10)
        elif kind == 'figure':
            bh = Inches(0.45) * len(sp.get('bullets', [])) if sp.get('bullets_below') else 0
            figs = sp['figures']; gap = Inches(0.3)
            fw = (W - 2 * M - gap * (len(figs) - 1)) // len(figs)
            for i, f in enumerate(figs):
                picture(s, FIG / f, M + i * (fw + gap), TOP, fw, avail - bh)
            if bh:
                textbox(s, sp['bullets'], M, bottom - bh, W - 2 * M, bh, 16, INK, bullets=True, space=4)
        elif kind == 'cards':
            cards = sp['cards']; n = len(cards); gap = Inches(0.35)
            cw = (W - 2 * M - gap * (n - 1)) // n
            ch = avail - Inches(0.1)
            for i, (head, body) in enumerate(cards):
                x = M + i * (cw + gap)
                rect(s, x, TOP, cw, ch, TINT)
                textbox(s, str(i + 1), x + Inches(0.3), TOP + Inches(0.25), Inches(0.8), Inches(0.7), 34, ORANGE, True)
                textbox(s, head, x + Inches(0.3), TOP + Inches(1.0), cw - Inches(0.6), Inches(0.9), 20, NAVY, True)
                textbox(s, body, x + Inches(0.3), TOP + Inches(1.95), cw - Inches(0.6), ch - Inches(2.1), 16, INK, bullets=True, space=8)
        elif kind == 'table':
            rows = sp['rows']; tw = int((W - 2 * M) * sp.get('table_frac', 0.55))
            rh = Inches(0.42)
            tbl = s.shapes.add_table(len(rows), len(rows[0]), M, TOP, tw, rh * len(rows)).table
            tbl.columns[0].width = int(tw * 0.3); tbl.columns[1].width = tw - int(tw * 0.3)
            for i, row in enumerate(rows):
                for j, val in enumerate(row):
                    c = tbl.cell(i, j); c.text = ''
                    c.fill.solid(); c.fill.fore_color.rgb = NAVY if i == 0 else (TINT if i % 2 == 0 else WHITE)
                    c.margin_left = c.margin_right = Inches(0.12); c.margin_top = c.margin_bottom = Inches(0.05)
                    p = c.text_frame.paragraphs[0]
                    runs(p, val, 14, WHITE if i == 0 else INK, bold=(i == 0 or j == 0))
                    c.vertical_anchor = MSO_ANCHOR.MIDDLE
            if sp.get('figures'):
                fx = M + tw + Inches(0.4)
                picture(s, FIG / sp['figures'][0], fx, TOP, W - M - fx, Inches(2.2))
            if sp.get('bullets'):
                fx = M + tw + Inches(0.4)
                textbox(s, sp['bullets'], fx, TOP + Inches(2.5), W - M - fx, avail - Inches(2.5), 15, INK, bullets=True, space=8)
        if sp.get('takeaway'):
            takeaway(s, sp['takeaway'])
        if sp.get('source'):
            textbox(s, sp['source'], M, H - Inches(0.38), W - 2 * M, Inches(0.3), 10, INK2)
    if sp.get('notes'):
        s.notes_slide.notes_text_frame.text = sp['notes']
prs.save(OUT)
print('saved', OUT, len(prs.slides), 'slides')
