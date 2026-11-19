"""Section 9 of the text-readability guide: contrast sweep, WCAG mapping,
and the negative result on min-max auto-levels.

Every number published in §9 re-derives from this file. Run it from the repo root:

    python3 s9_contrast.py

Requires numpy + Pillow and DejaVuSansMono.ttf (ships with DejaVu fonts on
Debian/Ubuntu, at /usr/share/fonts/truetype/dejavu/).

The matcher uses a +-3 px positional search per cell. Without it, the x4 arm
lands on a different sub-pixel phase at some font sizes and loses ~10 points to
crop offset rather than to resampling -- that bug is why every x4 figure here
is measured with the search, and why the first thing this script does is a
control: at full contrast the two arms must agree.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CW, CH, MARG = 16, 24, 3

STRINGS = []
for seed in (11, 12, 13):
    r = np.random.default_rng(seed)
    STRINGS.append("".join(r.choice(list(CHARS), 120)))


def render_line(text, size, ink=0, paper=255, pad=3):
    f = ImageFont.truetype(FONT, size)
    asc, desc = f.getmetrics()
    pitch = f.getlength("A")
    w = int(np.ceil(pitch * len(text))) + 2 * pad
    im = Image.new("L", (w, asc + desc + 2 * pad), paper)
    ImageDraw.Draw(im).text((pad, pad), text, font=f, fill=ink, anchor="la")
    return np.asarray(im).astype(float), pitch, pad


def up4(a):
    return np.asarray(
        Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        .resize((int(round(a.shape[1] * 4)), int(round(a.shape[0] * 4))), Image.LANCZOS)
    ).astype(float)


def cell_raw(cell):
    im = Image.fromarray(np.clip(cell, 0, 255).astype(np.uint8)).resize((CW, CH), Image.BILINEAR)
    return np.asarray(im).astype(float).ravel()


def _tpl_one(c, size, pad):
    f = ImageFont.truetype(FONT, size)
    asc, desc = f.getmetrics()
    pitch = f.getlength("A")
    im = Image.new("L", (int(np.ceil(pitch)) + 2 * pad, asc + desc + 2 * pad), 255)
    ImageDraw.Draw(im).text((pad, pad), c, font=f, fill=0, anchor="la")
    return np.asarray(im).astype(float), pitch


def build_templates(size, blur=False):
    """Templates rendered at the SAME optical path as the arm they judge.

    blur=True replicates what upscaling does to a glyph, so the x4 arm is not
    penalised for a blur its own templates do not have.
    """
    T = []
    for c in CHARS:
        g, pitch = _tpl_one(c, size, MARG)
        pad = MARG
        if blur:
            g, pitch, pad = up4(g), pitch * 4, MARG * 4
        T.append(cell_raw(g[:, pad:pad + int(round(pitch))]))
    return np.stack(T)


def read_shifted(gray, pitch, pad, text, TD):
    n = int((gray.shape[1] - 2 * pad) / pitch)
    hits = 0
    for i in range(n):
        c = pad + i * pitch
        x0, x1 = int(round(c)) - MARG, int(round(c + pitch)) + MARG
        patch = gray[:, max(0, x0):x1]
        best, bc = -1e18, "?"
        for s in range(2 * MARG + 1):
            w = patch[:, s:s + (x1 - x0) - 2 * MARG]
            if w.shape[1] < 2:
                continue
            v = cell_raw(w)
            d = -((TD - v) ** 2).sum(axis=1)
            j = int(np.argmax(d))
            if d[j] > best:
                best, bc = d[j], CHARS[j]
        hits += (bc == text[i])
    return 100.0 * hits / n


def xheight(size):
    g, _, _ = render_line("x", size, pad=0)
    rows = np.where((g < 128).any(axis=1))[0]
    return int(rows.max() - rows.min() + 1) if len(rows) else 0


def srgb_lin(c):
    c = np.asarray(c, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def contrast(ink, paper=255):
    a, b = srgb_lin(ink), srgb_lin(paper)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def stretch(a):
    lo, hi = a.min(), a.max()
    return a.copy() if hi - lo < 1e-6 else np.clip((a - lo) * 255.0 / (hi - lo), 0, 255)


_cache = {}


def arms(size):
    if size not in _cache:
        _cache[size] = (build_templates(size), build_templates(size, blur=True))
    return _cache[size]


def control():
    """Full contrast: the two arms must agree, or the x4 figures mean nothing."""
    print("CONTROL  ink=0, shift-search matcher -- arms must agree")
    worst = 0.0
    for size in (13, 15, 18, 24):
        TC, TB = arms(size)
        dn, du = [], []
        for t in STRINGS:
            g, p, pd = render_line(t, size, 0)
            dn.append(read_shifted(g, p, pd, t, TC))
            du.append(read_shifted(up4(g), p * 4, pd * 4, t, TB))
        d = float(np.mean(du) - np.mean(dn))
        worst = max(worst, abs(d))
        print(f"  size {size:2d}  native {np.mean(dn):5.1f}%  x4 {np.mean(du):5.1f}%  delta {d:+5.1f}")
    assert worst <= 2.0, f"matcher is phase-biased, delta {worst:.1f}"


def sweep():
    """Table 9.1 -- ink sweep at x-height 7."""
    print("\n9.1  ink sweep, font 13 (x-height 7), 3x120 chars")
    print(f"{'ink':>4} {'CR':>6} {'native':>7} {'x4':>6} {'gain':>6}")
    size = 13
    TC, TB = arms(size)
    for ink in (0, 60, 90, 110, 130, 150, 165, 180, 195, 210, 225, 240, 250):
        dn, du = [], []
        for t in STRINGS:
            g, p, pd = render_line(t, size, ink)
            dn.append(read_shifted(g, p, pd, t, TC))
            du.append(read_shifted(up4(g), p * 4, pd * 4, t, TB))
        n, u = float(np.mean(dn)), float(np.mean(du))
        print(f"{ink:4d} {contrast(ink):6.2f} {n:7.1f} {u:6.1f} {u - n:+6.1f}")


def wcag():
    """Table 9.2 -- published accessibility thresholds resolved to ink values."""
    def ink_for(target):
        lo, hi = 0.0, 255.0
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if contrast(mid) > target else (lo, mid)
        return int(round((lo + hi) / 2))

    print("\n9.2  WCAG thresholds vs 36-class accuracy")
    print(f"{'xh':>3} {'threshold':>10} {'CR':>6} {'ink':>4} {'native':>7} {'x4':>6}")
    for size in (10, 13, 18, 24):
        TC, TB = arms(size)
        for target, label in ((3.0, "AA-large"), (4.5, "AA"), (7.0, "AAA")):
            ink = ink_for(target)
            dn, du = [], []
            for t in STRINGS:
                g, p, pd = render_line(t, size, ink)
                dn.append(read_shifted(g, p, pd, t, TC))
                du.append(read_shifted(up4(g), p * 4, pd * 4, t, TB))
            print(f"{xheight(size):3d} {label:>10} {contrast(ink):6.2f} {ink:4d} "
                  f"{np.mean(dn):7.1f} {np.mean(du):6.1f}")


def autolevels():
    """Table 9.3 -- stretch on a text crop vs on a full screenshot."""
    def screenshot(size, ink, text, surroundings):
        f = ImageFont.truetype(FONT, size)
        asc, desc = f.getmetrics()
        pitch = f.getlength("A")
        bw = int(np.ceil(pitch * len(text))) + 6
        bh = asc + desc + 6
        extra = {"none": 0, "button": 40, "photo": 90}[surroundings]
        im = Image.new("L", (bw, bh + extra), 255)
        d = ImageDraw.Draw(im)
        d.text((3, 3), text, font=f, fill=ink, anchor="la")
        if surroundings == "button":
            d.rectangle([10, bh + 8, bw - 12, bh + 38], fill=0)
        elif surroundings == "photo":
            rng = np.random.default_rng(4)
            blk = np.clip(rng.normal(110, 45, (80, bw)), 0, 255)
            im.paste(Image.fromarray(blk.astype(np.uint8)), (0, bh + 4))
        return np.asarray(im).astype(float), pitch, bh

    print("\n9.3  min-max stretch: text crop vs full screenshot")
    print(f"{'xh':>3} {'surroundings':>12} {'native':>7} {'stretch':>8} {'x4':>6}")
    for size in (10, 13, 18):
        TC, TB = arms(size)
        for sur in ("none", "button", "photo"):
            dn, ds, du = [], [], []
            for t in STRINGS:
                a, p, bh = screenshot(size, 200, t, sur)
                band = a[:bh + 6, :]
                dn.append(read_shifted(band, p, 3, t, TC))
                sfull = stretch(a)
                ds.append(read_shifted(sfull[:bh + 6, :], p, 3, t, TC))
                du.append(read_shifted(up4(band), p * 4, 12, t, TB))
            print(f"{xheight(size):3d} {sur:>12} {np.mean(dn):7.1f} {np.mean(ds):8.1f} {np.mean(du):6.1f}")


if __name__ == "__main__":
    control()
    sweep()
    wcag()
    autolevels()
