"""README charts, regenerated from the committed benchmark results (bench/results/). No photos are drawn.

usage: python bench/charts.py            # writes docs/images/*.png

Every chart compares on one machine (GTX 1070 + i7-6700): the GPU against that machine's own CPU decoders and
encoders, same inputs, same settings. Linear axes, one panel per task with its heading above it, and one key at the
top. Under each title, which way is better. Every bar carries its decoder's name (inside the bar, or before the value
when the bar is too short) and its value with units after its end; every axis with numbers is labelled with units.
All "N.Nx faster/slower" labels are computed here. A chart in which any text overlaps other text, a bar edge or an
axis, or leaves the figure, is not written.
"""
import json
import math
import os
import statistics
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "..", "docs", "images")
MACHINE = "NVIDIA GTX 1070 + Intel i7-6700 (Windows 10)"
DATE = "measured 2026-11-18 (v0.1.0, the release source), machine otherwise idle"
LADDER = ("2-24 MP: one public test image (imagecompression.info, 7216 x 5412) cropped to 3:2, resized (Lanczos) "
          "and encoded by Pillow (libjpeg-turbo) at quality 90 (bench/make_ladder.py).")
LOWER, HIGHER = "↓ Lower is better", "↑ Higher is better"
BADGE, INK = "#f6d32d", "#1d1d1b"                # the which-way-is-better badge: a highlighter-yellow pill with an
BADGE_EDGE_PX = 1.5                              # ink border (yellow on white is only 1.5:1) and ink text (11.5:1);
                                                 # no series is yellow, so the badge never reads as a chart colour
C = {  # one colour per series, the same in every chart and key; white text on each reaches 5:1 (WCAG), see contrast()
    "gpu": "#1a5fb4", "gpu1": "#1a5fb4", "gpu2": "#0e3d73", "cpu1": "#6b6965", "cpu8": "#3d3846", "nvjpeg": "#277a4c",
    "nvjpeg_api": "#277a4c", "pillow": "#bc4300", "simplejpeg": "#5e5c64", "gpu_conv": "#0e3d73", "upload": "#7a4fa0",
    "huffman": "#1a5fb4", "pixels": "#1f7a52",
    "gpu_host": "#1a5fb4", "gpu_dev": "#0e3d73", "sj1": "#5e5c64", "pil1": "#bc4300",
    "rep_normal": "#1a5fb4", "rep_light": "#0e3d73", "rep_full": "#7a4fa0", "nvjpeg_one": "#277a4c",
}
BOLD = {"gpu", "gpu1", "gpu2", "gpu_host", "gpu_dev", "rep_light"}   # exact-gpu-jpeg's bars: their outside text in bold
ON_FILL, ON_WHITE = 5.0, 4.5                     # the least contrast for text on a bar, and for text on the page
NAMES = {"gpu": "exact-gpu-jpeg", "gpu1": "exact-gpu-jpeg, 1 GPU thread", "gpu2": "exact-gpu-jpeg, 2 GPU threads",
         "cpu1": "libjpeg-turbo, 1 CPU core", "cpu8": "libjpeg-turbo, 8 CPU threads", "nvjpeg": "nvJPEG (torchvision)",
         "pillow": "Pillow (libjpeg-turbo)", "nvjpeg_api": "nvJPEG at its fastest (its own C API)",
         "simplejpeg": "simplejpeg (libjpeg-turbo)", "gpu_conv": "exact-gpu-jpeg, then a separate conversion",
         "gpu_host": "exact-gpu-jpeg, from host memory", "gpu_dev": "exact-gpu-jpeg, pixels already on the GPU",
         "sj1": "simplejpeg, 1 CPU thread", "pil1": "Pillow, 1 CPU thread",
         "rep_normal": "exact-gpu-jpeg, normal decode", "rep_light": 'exact-gpu-jpeg, from the UnHuff copy, check="light"',
         "rep_full": 'exact-gpu-jpeg, from the UnHuff copy, check="full" (the default)', "nvjpeg_one": "nvJPEG (its own C API)"}
FS = 11          # bar text
BAR_IN = 0.68    # inches per bar in a panel; the bar fills 80% of it
GAP = 1.0        # inches between panels: tick numbers, the axis label below them, then the next panel's heading
BOTTOM = 0.7     # inches below the last panel: its tick numbers and axis label
DPI = 110        # text is measured at the resolution the PNG is written at


def r1(x):
    """One decimal, rounding halves up (19.25 -> 19.3), as in the README."""
    return f"{math.floor(x * 10 + 0.5) / 10:.1f}"


def load(*p):
    return json.load(open(os.path.join(RES, *p)))


def speed_text(r, base_name, why=""):
    if r >= 1:
        return f"{r:.1f}x faster than {base_name}"
    return f"{1 / r:.1f}x slower than {base_name}" + (f": {why}" if why else "")


def contrast(a, b):
    """WCAG 2 contrast ratio of two colours (any form matplotlib reads): (L1 + 0.05) / (L2 + 0.05), L the relative
    luminance of the sRGB colour, L1 the lighter."""
    def lum(c):
        rgb = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in matplotlib.colors.to_rgb(c)]
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def text_colour(k):
    """Text inside a bar is always white (bold); every fill that carries text keeps it at ON_FILL or more."""
    return "white"


def header(fig, h, title, better, subtitle, handles, ncol):
    """Title, which way is better in a badge on its own line under it, the wrapped subtitle and the key; returns
    the height (inches) they take."""
    y = 0.3
    fig.text(0.01, 1 - y / h, title, fontsize=22, fontweight="bold", va="top")
    y += 0.6
    fig.text(0.01 + 0.12 / fig.get_figwidth(), 1 - y / h, better, fontsize=13, fontweight="bold", va="top",
             color=INK, bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.9", fc=BADGE, ec=INK,
                                  lw=BADGE_EDGE_PX * 72 / DPI))
    y += 0.55
    for ln in textwrap.wrap(subtitle, 150):
        fig.text(0.01, 1 - y / h, ln, fontsize=11, va="top", color="#333333")
        y += 0.22
    y += 0.08
    if not handles:
        return y + 0.1
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 1 - y / h), ncol=ncol, frameon=False,
               fontsize=11)
    return y + 0.55


def figure(title, better, subtitle, n_bars, keys, width=13):
    """n_bars: the number of bars in each panel, one entry per panel."""
    head = 2.25 + 0.22 * len(textwrap.wrap(subtitle, 150))
    heights = [n * BAR_IN for n in n_bars]
    h = head + sum(heights) + (len(n_bars) - 1) * GAP + BOTTOM
    fig, axes = plt.subplots(len(n_bars), 1, figsize=(width, h), dpi=DPI, facecolor="white",
                             gridspec_kw=dict(height_ratios=heights))
    axes = [axes] if len(n_bars) == 1 else list(axes)
    top = header(fig, h, title, better, subtitle, [Patch(color=C[k], label=NAMES.get(k, k)) for k in keys],
                 max(len(keys), 1))
    fig.subplots_adjust(left=0.02, right=0.98, top=1 - (top + 0.3) / h, bottom=BOTTOM / h,
                        hspace=GAP / (sum(heights) / len(heights)))
    return fig, axes


VFS = round(FS * 1.35, 2)   # a bar's value ("7,310 images/s", "11.3 ms"): bold and larger than its name
RUN_GAP = 5.0               # pixels between the runs of one line (name, value, comparison)


def _width(ax, renderer, s, bold, size=FS):
    t = ax.text(0, 0, s, fontsize=size, fontweight="bold" if bold else "normal")
    w = t.get_window_extent(renderer).width
    t.remove()
    return w


INSIDE, NAME_INSIDE, OUTSIDE = 2, 1, 0                   # where a bar's text goes, best first


def panel(ax, heading, bars, unit):
    """bars: list of (series key, value, name, value with units[, comparison]). Horizontal, first bar on top.
    Each bar takes the first layout that fits: the name on the left inside the bar and the value with its comparison
    right-aligned inside it ("7,310 images/s · 4.7x faster than ..."); else the name inside and the value after the
    bar's end ("7,310 images/s (4.7x faster than ...)"); else both after the bar's end. The value is bold at VFS,
    larger than the name and the comparison (FS); the runs of a line share one baseline. The x range is made just
    wide enough for the text after the bars."""
    fig = ax.figure
    r = fig.canvas.get_renderer()
    bars = [tuple(b) + ("",) * (5 - len(b)) for b in bars]
    n = len(bars)
    y = list(range(n))[::-1]
    ax.barh(y, [b[1] for b in bars], color=[C[b[0]] for b in bars], height=0.8)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticks([])
    ax.set_title(heading, loc="left", fontsize=13, fontweight="bold", pad=6)
    ax.set_xlabel(unit, fontsize=10, color="#444444", labelpad=7)
    ax.tick_params(axis="x", labelsize=9, colors="#555555")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    box = ax.get_window_extent(r)
    axw = box.width
    pad = 7.0                                          # pixels between a bar's edge and its text
    bold = [k in BOLD for k, *_ in bars]

    def in_runs(b):                                    # the value (and comparison) inside the bar
        return [(b[3], VFS, True)] + ([(f"· {b[4]}", FS, True)] if b[4] else [])

    def out_runs(b, where, bb):                        # the text after the bar's end
        runs = [] if where == NAME_INSIDE else [(f"{b[2]}:", FS, bb)]
        return runs + [(b[3], VFS, True)] + ([(f"({b[4]})", FS, True)] if b[4] else [])

    def runs_width(runs):
        return sum(_width(ax, r, s, bo, sz) for s, sz, bo in runs) + RUN_GAP * (len(runs) - 1)

    name_w = [_width(ax, r, b[2], True) for b in bars]
    in_w = [runs_width(in_runs(b)) for b in bars]
    where = [INSIDE] * n
    for _ in range(2 * n + 2):
        xmax = max(b[1] for b in bars) * 1.02
        for b, w, bb in zip(bars, where, bold):
            if w != INSIDE:
                tw = runs_width(out_runs(b, w, bb))
                xmax = max(xmax, b[1] * axw / (axw - 4 * pad - tw))   # (room to spare before the axis ends)
        fits = []
        for i, (b, w) in enumerate(zip(bars, where)):
            px = b[1] / xmax * axw
            if w == INSIDE and name_w[i] + in_w[i] + 5 * pad > px:
                w = NAME_INSIDE
            if w == NAME_INSIDE and name_w[i] + 2 * pad > px:
                w = OUTSIDE
            fits.append(w)
        if fits == where:
            break
        where = fits                                   # a bar's text only ever moves outward: this ends
    ax.set_xlim(0, xmax)
    to_x = xmax / axw                                  # data units per pixel along x
    drop = 0.36 * VFS * DPI / 72 / (box.height / n)    # baseline below the bar's centre: half a cap of the value

    def draw(x_px, yy, runs, colour):
        for s, sz, bo in runs:
            ax.text(x_px * to_x, yy - drop, s, va="baseline", ha="left", fontsize=sz,
                    fontweight="bold" if bo else "normal", color=colour)
            x_px += _width(ax, r, s, bo, sz) + RUN_GAP

    for yy, b, w, bb in zip(y, bars, where, bold):
        k, v_px = b[0], b[1] / to_x
        if w != OUTSIDE:
            draw(pad, yy, [(b[2], FS, True)], text_colour(k))
        if w == INSIDE:
            runs = in_runs(b)
            draw(v_px - pad - runs_width(runs), yy, runs, text_colour(k))
        else:
            draw(v_px + pad, yy, out_runs(b, w, bb), "black")


def check_layout(fig, name, margin=2.0):
    """Refuse a chart in which any piece of text (titles, the key, panel headings, bar text, tick numbers, axis
    labels) overlaps another, crosses a bar's edge (text inside a bar must lie wholly inside it, with `margin`
    pixels to spare), touches an axis line, or leaves the figure; or whose colour against what lies under it (the
    bar's fill, else the white page) is below ON_FILL (on a bar) or ON_WHITE (on the page)."""
    for ax in fig.axes:          # keep only the ticks inside the axis (the others are never drawn): the same image
        lo, hi = ax.get_xlim()
        ax.set_xticks([t for t in ax.get_xticks() if lo - 1e-9 <= t <= hi + 1e-9])
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    bars = [(p.get_window_extent(r), p.get_facecolor()) for ax in fig.axes for p in ax.patches
            if isinstance(p, Rectangle)]
    lines = [ax.spines["bottom"].get_window_extent(r) for ax in fig.axes]
    boxes = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        b = t.get_window_extent(r)
        s = t.get_text()
        if b.x0 < 0 or b.y0 < 0 or b.x1 > W or b.y1 > H:
            raise SystemExit(f"{name}: text leaves the figure: {s!r}")
        under, least = fig.get_facecolor(), ON_WHITE
        badge = t.get_bbox_patch()                 # text drawn on its own box (the which-way-is-better badge)
        if badge is not None:
            b = badge.get_window_extent(r)
            under, least = badge.get_facecolor(), ON_FILL
            if b.x0 < 0 or b.y0 < 0 or b.x1 > W or b.y1 > H:
                raise SystemExit(f"{name}: a badge leaves the figure: {s!r}")
        for p, fill in bars:
            if b.overlaps(p.padded(margin)):
                if not (p.x0 + margin <= b.x0 and b.x1 <= p.x1 - margin and p.y0 + margin <= b.y0
                        and b.y1 <= p.y1 - margin):
                    raise SystemExit(f"{name}: text crosses a bar's edge: {s!r}")
                under, least = fill, ON_FILL
        if contrast(t.get_color(), under) < least:
            raise SystemExit(f"{name}: text contrast {contrast(t.get_color(), under):.2f}:1 is below {least}:1 "
                             f"({matplotlib.colors.to_hex(t.get_color())} on {matplotlib.colors.to_hex(under)}): {s!r}")
        for ln in lines:
            if b.overlaps(ln.padded(margin)):
                raise SystemExit(f"{name}: text touches an axis: {s!r}")
        if t.axes is not None and t in t.axes.texts and b.x1 > t.axes.get_window_extent(r).x1:
            raise SystemExit(f"{name}: text runs past the end of its axis: {s!r}")
        boxes.append((s, b.padded(margin / 2)))
    for i, (ta, a) in enumerate(boxes):
        for tb, b in boxes[i + 1:]:
            if a.overlaps(b):
                raise SystemExit(f"{name}: text overlaps: {ta!r} and {tb!r}")


def save(fig, name):
    check_layout(fig, name)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=DPI, facecolor="white")
    plt.close(fig)
    print("wrote", os.path.relpath(p, os.path.join(HERE, "..")))


def nvjpeg_best(rows):
    """nvJPEG's fastest valid result in one set of bench/nvjpeg_direct.py: (images per second, which call)."""
    return max((r["images_per_s"], k) for k, r in rows.items()
               if k.startswith("nvjpeg") and "images_per_s" in r and "status" not in r)


def decode_groups():
    """v0.1.0 on the release source (2026-11-18): the 2-24 MP ladder (one public image, see bench/make_ladder.py) and
    the 12 camera photos, with nvJPEG, in one run."""
    return load("20261118", "bench.json")["groups"]


SIZES = [("02mp_420", "2 MP, 4:2:0"), ("06mp_420", "6 MP, 4:2:0"), ("12mp_420", "12 MP, 4:2:0"),
         ("24mp_420", "24 MP, 4:2:0"), ("camera_30mp_422", "30 MP camera photos, 4:2:2 (12 files)")]


def chart_decode_time(G):
    fig, axes = figure("JPEG decode time, one image at a time", LOWER,
                       f"{MACHINE}; {DATE}. Median ms from JPEG bytes in memory to a decoded RGB image (exact-gpu-jpeg and nvJPEG: in GPU memory). "
                       + LADDER,
                       [3] * len(SIZES), ["gpu", "cpu1", "nvjpeg"])
    for ax, (k, head) in zip(axes, SIZES):
        r = G[k]
        g, c, n = r["gpu_ms"], r["cpu_ms"], r["nvjpeg_ms"]
        panel(ax, head, [
            ("gpu", g, NAMES["gpu"], f"{g:.1f} ms", speed_text(c / g, 'libjpeg-turbo on 1 CPU core')),
            ("cpu1", c, NAMES["cpu1"], f"{c:.0f} ms"),
            ("nvjpeg", n, NAMES["nvjpeg"], f"{n:.0f} ms", f"not byte-identical: up to {r['nvjpeg_max_abs_diff']} levels off"),
        ], "milliseconds per image")
    save(fig, "decode-time.png")


def chart_decode_subsampling(G):
    subs = [("24mp_444", "24 MP, 4:4:4"), ("24mp_422", "24 MP, 4:2:2"), ("24mp_420", "24 MP, 4:2:0"),
            ("24mp_420rst", "24 MP, 4:2:0 with a restart marker every MCU row")]
    fig, axes = figure("JPEG decode time by subsampling (24 MP)", LOWER,
                       f"{MACHINE}; {DATE}. Median ms, one image at a time, the same public test image resized to 24 MP "
                       "and encoded by Pillow at quality 90 in each layout.",
                       [3] * len(subs), ["gpu", "cpu1", "nvjpeg"])
    for ax, (k, head) in zip(axes, subs):
        r = G[k]
        g, c, n = r["gpu_ms"], r["cpu_ms"], r["nvjpeg_ms"]
        panel(ax, head, [
            ("gpu", g, NAMES["gpu"], f"{g:.1f} ms", speed_text(c / g, 'libjpeg-turbo on 1 CPU core')),
            ("cpu1", c, NAMES["cpu1"], f"{c:.0f} ms"),
            ("nvjpeg", n, NAMES["nvjpeg"], f"{n:.0f} ms"),
        ], "milliseconds per image")
    save(fig, "decode-subsampling.png")


def chart_throughput(G):
    fig, axes = figure("JPEG decode throughput", HIGHER,
                       f"{MACHINE}; {DATE}. Images per second, repeating the set; CPU threads each run libjpeg-turbo, "
                       "exact-gpu-jpeg's GPU threads each have their own decoder and CUDA stream. " + LADDER,
                       [4] * len(SIZES), ["gpu1", "gpu2", "cpu8", "cpu1"])
    for ax, (k, head) in zip(axes, SIZES):
        r = G[k]
        g1, g2, c8, c1 = r["gpu_1t_per_s"], r["gpu_2t_per_s"], r["cpu_8t_per_s"], r["cpu_1t_per_s"]
        why = "per-image overhead" if r["megapixels"] <= 3 else ""
        base = "libjpeg-turbo on 8 CPU threads"
        panel(ax, head, [
            ("gpu1", g1, NAMES["gpu1"], per_s(g1), speed_text(g1 / c8, base, why)),
            ("gpu2", g2, NAMES["gpu2"], per_s(g2), speed_text(g2 / c8, base, why)),
            ("cpu8", c8, NAMES["cpu8"], per_s(c8)),
            ("cpu1", c1, NAMES["cpu1"], per_s(c1)),
        ], "images per second")
    save(fig, "decode-throughput.png")


def chart_stages():
    steps = [("s0_baseline", "Baseline (the first working version)"),
             ("s1_branchfree_writer", "+ branch-free coefficient writer"),
             ("s2_block_sync_eager", "+ in-block synchronisation, GPU decides convergence"),
             ("s3_vector_unstuff", "+ unstuffing with 16-byte loads"),
             ("s4_write_from_checkpoints", "+ write pass from checkpoints, 4096-bit chunks"),
             ("s8_loop_decoder_wholeread", "v0.1: + faster Huffman tables and unstuffing, IDCT and colour kernels")]
    rows = [(head, load("speed", f"{k}.json")) for k, head in steps]
    h = 3.15 + len(rows) * 1.45
    fig, ax = plt.subplots(figsize=(17, h), dpi=DPI, facecolor="white")
    top = header(fig, h, "exact-gpu-jpeg's decoder, step by step (30 MP camera photos, 4:2:2)", LOWER,
                 f"{MACHINE}; steps measured 2026-11-13, v0.1 on 2026-11-16, machine otherwise idle. exact-gpu-jpeg's GPU time per photo split by stage (CUDA events); wall time includes reading the "
                 "file and Python. 12 photos x 2 repeats, decode alone. Every row reads the whole file, then uploads it.",
                 [Patch(color=C[s], label=s.capitalize()) for s in ("upload", "huffman", "pixels")], 3)
    fig.subplots_adjust(left=0.02, right=0.98, top=1 - (top + 0.2) / h, bottom=BOTTOM / h)
    base = rows[0][1]["one_at_a_time_mean"]["wall_ms"]
    after = []                                 # (bar end, y, text, bold): drawn once the x range is known
    for i, (head, r) in enumerate(rows):
        m = r["one_at_a_time_mean"]
        y = len(rows) - 1 - i
        left = 0
        for s in ("upload", "huffman", "pixels"):
            v = m[f"{s}_ms"]
            ax.barh(y, v, left=left, color=C[s], height=0.62)
            ax.text(left + v / 2, y, f"{v:.1f} ms", ha="center", va="center", fontsize=round(10.5 * 1.35, 2), fontweight="bold",
                    color=text_colour(s))
            left += v
        thr = r["photos_per_s"]
        t1 = (thr["gpu_1_threads_run0"] + thr["gpu_1_threads_run1"]) / 2
        t2 = (thr["gpu_2_threads_run0"] + thr["gpu_2_threads_run1"]) / 2
        vs = "the baseline" if i == 0 else f"{base / m['wall_ms']:.2f}x faster than the baseline"
        txt = (f"exact-gpu-jpeg: {r1(left)} ms GPU, {r1(m['wall_ms'])} ms wall ({vs}); "
               f"{t1:.0f} / {t2:.0f} photos/s on 1 / 2 GPU threads")
        if head.startswith("v0.1"):           # the default reads the file in pieces during the copy: its wall time
            st = load("speed", "s8_loop_decoder.json")["one_at_a_time_mean"]["wall_ms"]
            head += f" (default, file read during the copy: wall {r1(st)} ms)"
        after.append((left, y, txt, i == len(rows) - 1))
        ax.text(0, y + 0.37, head, fontsize=12, fontweight="bold", va="bottom")
    rd = fig.canvas.get_renderer()
    axw, pad = ax.get_window_extent(rd).width, 7.0
    xmax = max([32] + [x * axw / (axw - 4 * pad - _width(ax, rd, s, bb)) for x, _, s, bb in after])
    ax.set_xlim(0, xmax)
    for x, y, s, bb in after:
        ax.text(x + pad / axw * xmax, y, s, va="center", fontsize=FS, fontweight="bold" if bb else "normal")
    ax.set_ylim(-0.55, len(rows) - 0.25)
    ax.set_yticks([])
    ax.set_xlabel("milliseconds of GPU time per photo", fontsize=10, color="#444444", labelpad=7)
    ax.tick_params(axis="x", labelsize=9, colors="#555555")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    save(fig, "decode-speedups.png")


def chart_batches():
    """Batches of JPEGs: exact-gpu-jpeg against nvJPEG's batched decode and libjpeg-turbo on every CPU thread."""
    b = load("20261118", "nvjpeg_direct.json")           # one run: the three decoders in turn on the same files
    sets = b["sets"]
    fig, axes = figure("Decoding batches of JPEGs", HIGHER,
                       f"{MACHINE}; measured {b['measured']} in one run, machine otherwise idle. {b['images_per_set']} different JPEGs per "
                       "size (quality 90, 4:2:0), from JPEG bytes in host memory to decoded RGB images (exact-gpu-jpeg and nvJPEG: in "
                       "GPU memory, synchronised; libjpeg-turbo: in host memory). nvJPEG: the fastest of its own API's calls "
                       "and back ends (bench/nvjpeg_direct.py; nvJPEG 12.3.3); this GPU has no hardware JPEG engine. "
                       "exact-gpu-jpeg's pixels are byte-identical to libjpeg-turbo's.",
                       [3] * len(sets), ["gpu", "cpu8", "nvjpeg_api"])
    for ax, (k, r) in zip(axes, sets.items()):
        g, c = r["exact-gpu-jpeg decode_batch"]["images_per_s"], r["libjpeg-turbo, 8 threads"]["images_per_s"]
        n, call = nvjpeg_best(r)
        w, h = b["sizes"][k]
        panel(ax, f"{k[:-2]} MP ({w} x {h})", [
            ("gpu", g, "exact-gpu-jpeg, batches of 64",
             per_s(g), f"{g / n:.1f}x faster than nvJPEG, {g / c:.1f}x faster than libjpeg-turbo on 8 CPU threads"),
            ("cpu8", c, NAMES["cpu8"], per_s(c)),
            ("nvjpeg_api", n, f"nvJPEG at its fastest ({call})", per_s(n)),
        ], "images per second")
    save(fig, "batch-throughput.png")


def per_s(x):
    """Images per second as in the README: whole numbers from 100 up, one decimal below (88.5, 6.1)."""
    return f"{x:,.0f} images/s" if x >= 100 else f"{x:.1f} images/s"


def ms_text(x):
    return f"{x:.1f} ms" if x >= 10 else f"{x:.2f} ms"


def chart_encode():
    """A photo's three web exports, encoded one at a time (bench/encode_bench.py; Pillow is the reference)."""
    e = load("20261118", "encode.json")
    sizes = [("3840_q95_444", "4:4:4"), ("1500_q90_444", "4:4:4"), ("1200_q87_420", "4:2:0")]
    fig, axes = figure("Encoding a photo's three web exports, one at a time", LOWER,
                       f"{MACHINE}; measured 2026-11-18 (v0.1.0, the release source), machine otherwise idle. Median ms "
                       f"to encode one RGB image to JPEG bytes in host memory, {e['photos']} camera photos resized "
                       "(Lanczos) to each size, with their EXIF (bench/encode_bench.py). exact-gpu-jpeg's files are "
                       "byte-identical to Pillow's; simplejpeg's (libjpeg-turbo with its own defaults) are not. From "
                       "host memory: a NumPy array, uploaded by the call (like for like with the CPU encoders).",
                       [4] * len(sizes), ["gpu_host", "gpu_dev", "sj1", "pil1"])
    for ax, (k, sub) in zip(axes, sizes):
        r = e["sizes"][k]
        h, g, s, p = r["gpu_from_host_ms"], r["gpu_ms"], r["simplejpeg_ms"], r["pillow_ms"]
        panel(ax, f"{r['width']} x {r['height']}, quality {r['quality']}, {sub}", [
            ("gpu_host", h, NAMES["gpu_host"], ms_text(h), speed_text(p / h, "Pillow")),
            ("gpu_dev", g, NAMES["gpu_dev"], ms_text(g), speed_text(p / g, "Pillow")),
            ("sj1", s, NAMES["sj1"], ms_text(s), "not byte-identical to Pillow"),
            ("pil1", p, NAMES["pil1"], ms_text(p), "the reference"),
        ], "milliseconds per image")
    save(fig, "encode.png")


def chart_repeat():
    """A repeat decode from the in-memory UnHuff copy against a normal decode (bench/repeat_paths.py): the median
    over the files of each file's median."""
    rp = load("20261118", "repeat_paths.json")
    sets = [("photos", "12 camera photos (30 MP, 4:2:2)"),
            ("public", "14 public-set photos (imagecompression.info, 3.4 to 39 MP, quality 95, 4:2:0)")]
    fig, axes = figure("Decoding the same file again: the UnHuff copy", LOWER,
                       "UnHuff: a Huffman-free coefficient format for exact repeat JPEG decodes. A RepeatCache keeps an "
                       "UnHuff copy of a file's coefficients in GPU memory; a later decode of that file starts from it and "
                       "skips the entropy decode, with the same pixels. "
                       f"{MACHINE}; measured {rp['measured'][:10]} (v{rp['version']}), machine otherwise idle. Median ms "
                       "a call from a path (the file in the operating system's cache) to RGB in GPU memory: each file's "
                       f"median of {rp['rounds']} rounds of {rp['calls_a_round']} calls, then the median over the files. "
                       'check="light" compares the size, modification time and first 64 KiB with the last full read; '
                       'check="full" reads the whole file and compares its full identity.',
                       [3, 3], ["rep_normal", "rep_light", "rep_full"], width=15)
    for ax, (s, head) in zip(axes, sets):
        rows = [x for x in rp["rows"] if x["set"] == s]
        m = {p: statistics.median(x["ms"][p]["median"] for x in rows) for p in ("normal", "cache, light", "cache, full")}
        n, li, fu = m["normal"], m["cache, light"], m["cache, full"]
        panel(ax, head, [
            ("rep_normal", n, NAMES["rep_normal"], f"{r1(n)} ms"),
            ("rep_light", li, NAMES["rep_light"], f"{r1(li)} ms", f"{n / li:.1f}x faster than a normal decode"),
            ("rep_full", fu, NAMES["rep_full"], f"{r1(fu)} ms",
             "slower than a normal decode: reads and hashes the whole file"),
        ], "milliseconds per call")
    save(fig, "repeat-cache.png")


def chart_one_at_a_time():
    """One call per image at each size, against nvJPEG's own C API one call per image and libjpeg-turbo on one core
    (bench/nvjpeg_direct.py, the README's first table)."""
    b = load("20261118", "nvjpeg_direct.json")
    groups = [(f"{k[:-2]} MP ({w} x {h}, 4:2:0)", b["sets"][k]) for k, (w, h) in b["sizes"].items()]
    groups.append((f"{b['camera_photos']['files']} camera photos (30 MP, 4:2:2)", b["camera_photos"]["results"]))
    fig, axes = figure("One image at a time, by size: where exact-gpu-jpeg wins and loses", HIGHER,
                       f"{MACHINE}; measured {b['measured']} in one run (v{b['env']['exact_gpu_jpeg']}), machine otherwise "
                       f"idle. Images per second, one call per image, each waited for, from JPEG bytes in host memory to "
                       f"RGB pixels ({b['images_per_set']} different JPEGs per size, quality 90). nvJPEG: its own C API "
                       f"(nvjpegDecode, DEFAULT back end, nvJPEG {b['env']['nvjpeg']}); libjpeg-turbo: simplejpeg, accurate "
                       "IDCT, to host memory. At 0.25 MP exact-gpu-jpeg is slower: one call per small image pays its fixed "
                       "per-call cost each time. A list is decoded as one batch instead: "
                       f"{b['sets']['0.25mp']['exact-gpu-jpeg decode_batch']['images_per_s']:,.0f} images/s at 0.25 MP "
                       "(batch-throughput.png).",
                       [3] * len(groups), ["gpu", "nvjpeg_one", "cpu1"])
    for ax, (head, r) in zip(axes, groups):
        g = r["exact-gpu-jpeg decode, one call per image"]["images_per_s"]
        n = r["nvjpegDecode DEFAULT"]["images_per_s"]
        c = r["libjpeg-turbo, 1 thread"]["images_per_s"]
        if g >= max(n, c):
            cmp = f"{g / n:.1f}x faster than nvJPEG, {g / c:.1f}x faster than libjpeg-turbo on 1 CPU core"
        else:
            cmp = f"slower: {g / n:.2f}x nvJPEG, {g / c:.2f}x libjpeg-turbo on 1 CPU core"
        panel(ax, head, [
            ("gpu", g, NAMES["gpu"], per_s(g), cmp),
            ("nvjpeg_one", n, NAMES["nvjpeg_one"], per_s(n)),
            ("cpu1", c, NAMES["cpu1"], per_s(c)),
        ], "images per second")
    save(fig, "one-at-a-time.png")


if __name__ == "__main__":
    G = decode_groups()
    chart_batches()
    chart_decode_time(G)
    chart_decode_subsampling(G)
    chart_throughput(G)
    chart_stages()
    chart_encode()
    chart_repeat()
    chart_one_at_a_time()
