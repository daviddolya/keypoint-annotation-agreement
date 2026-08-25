# Ported unchanged from polygon-annotation-agreement/tools/chrome.py --
# the demonstration frame is the same across every stage of the portfolio.
#!/usr/bin/env python3
"""The frame around a demonstration picture: header, swatches, footer.

Every picture of a disagreement in this portfolio carries the same frame, so
that it can be read on its own -- saved, pasted into a message, or opened years
later without the README next to it. The frame answers three questions before
the reader looks at the picture:

    which colour is the reference and which is mine   -- the swatches
    what the numbers of this case are                 -- the header facts
    which frame this is                               -- the footer

Two layouts, one rule. With an overlay both swatches sit side by side in the
header. With two panels the swatch sits over its own panel, so colour and side
are named together rather than instead of each other.

The colours are the matplotlib tab10 blue and orange. They are fixed here and
not passed in: a demonstration where blue means the reference in one repository
and something else in another is worse than no legend at all.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REF_COLOR = (31, 119, 180)      # blue: the reference annotation
MINE_COLOR = (255, 127, 14)     # orange: mine

BAR_H = 26                      # header
FOOT_H = 22                     # footer
BAR_BG = (255, 255, 255)
BAR_FG = (20, 20, 20)
FOOT_FG = (90, 90, 90)
SWATCH = 12
PAD = 6

REF_LABEL = "reference"
MINE_LABEL = "mine"

FONT_CANDIDATES = [
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def load_font(size: int = 14):
    """A TrueType font of the requested size, or the built-in default.

    The bitmap font built into Pillow ignores the size argument and comes out
    too small to read once GitHub scales the picture down.
    """
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    for path in sorted(Path("/usr/share/fonts").rglob("*.ttf")):
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def legend(draw: ImageDraw.ImageDraw, x: int, y: int, font,
           ref_label: str = REF_LABEL, mine_label: str = MINE_LABEL) -> int:
    """One swatch-and-label pair per side. Returns the x after the last label."""
    for color, label in ((REF_COLOR, ref_label), (MINE_COLOR, mine_label)):
        draw.rectangle([x, y, x + SWATCH, y + SWATCH], fill=color)
        x += SWATCH + 5
        draw.text((x, y - 2), label, fill=BAR_FG, font=font)
        x += text_width(draw, label, font) + 14
    return x


def with_chrome(image: Image.Image, facts: str = "", footer: str = "",
                panels: int = 1, gutter: int = 8, font=None,
                upscale: bool = False,
                ref_label: str = REF_LABEL, mine_label: str = MINE_LABEL):
    """A white canvas of header + image + footer, with the frame drawn on it.

    panels=1  the image is pasted once and both swatches sit in the header;
    panels=2  the image is pasted twice side by side and each panel gets its
              own swatch above it -- the caller draws the reference on the left
              and mine on the right.

    ref_label / mine_label extend the two labels where the side carries more
    than its name -- a transcription, say. The swatch still comes first, so the
    colour and what it stands for are read together.

    upscale=True enlarges an image narrower than its own legend so that it
    fills the canvas instead of sitting in a corner of white space. Only for
    pictures already drawn on: it invalidates the returned offsets, so callers
    that keep drawing afterwards must leave it off.

    Returns (canvas, draw, top, panel_width). `top` is the y offset of the image
    area: the caller keeps drawing in image coordinates shifted by it. With two
    panels the right one starts at `panel_width + gutter`.
    """
    if panels not in (1, 2):
        raise ValueError("panels must be 1 or 2")
    font = font or load_font()
    w, h = image.width, image.height
    total_w = w * panels + gutter * (panels - 1)

    # A crop can be narrower than its own legend. Widening the canvas is the
    # only option that keeps the legend readable: shrinking the text defeats
    # the purpose, and dropping it takes the picture back to "which colour is
    # which?", which is exactly what this frame exists to answer.
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    needed = PAD * 2
    for label in (ref_label, mine_label):
        needed += SWATCH + 5 + text_width(probe, label, font) + 14
    if facts:
        needed += text_width(probe, facts, font)
    if footer:
        needed = max(needed, PAD * 2 + text_width(probe, footer, font))
    if needed > total_w and upscale:
        # A crop is meant to be looked at closely anyway -- a 30 px contour
        # cannot be judged at native size. Growing it to the legend width
        # costs nothing and removes the white margin.
        factor = needed / total_w
        w, h = int(w * factor), int(h * factor)
        image = image.resize((w, h), Image.LANCZOS)
        total_w = w * panels + gutter * (panels - 1)
    total_w = max(total_w, needed)

    canvas = Image.new("RGB", (total_w, BAR_H + h + FOOT_H), BAR_BG)
    for i in range(panels):
        canvas.paste(image, (i * (w + gutter), BAR_H))
    draw = ImageDraw.Draw(canvas)

    y = (BAR_H - SWATCH) // 2
    if panels == 1:
        x = legend(draw, PAD, y, font, ref_label, mine_label)
        if facts:
            draw.text((x, y - 2), facts, fill=BAR_FG, font=font)
    else:
        for i, (color, label) in enumerate(((REF_COLOR, ref_label),
                                            (MINE_COLOR, mine_label))):
            x = i * (w + gutter) + PAD
            draw.rectangle([x, y, x + SWATCH, y + SWATCH], fill=color)
            draw.text((x + SWATCH + 5, y - 2), label, fill=BAR_FG, font=font)
        if facts:
            x = PAD + SWATCH + 5 + text_width(draw, ref_label, font) + 14
            draw.text((x, y - 2), facts, fill=BAR_FG, font=font)

    if footer:
        draw.text((PAD, BAR_H + h + (FOOT_H - 13) // 2), footer,
                  fill=FOOT_FG, font=font)
    return canvas, draw, BAR_H, w
