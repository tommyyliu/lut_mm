# SPDX-License-Identifier: GPL-3.0-or-later
"""Render results/thread_scaling.csv as results/thread_scaling.svg.

One panel per shape, throughput vs thread count (log2 x-axis), our kernel
vs BitNet TL2. Regenerate after re-running the sweep so the CSV, the SVG,
and the README table all come from the same data.
"""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "thread_scaling.csv")
SVG = os.path.join(ROOT, "results", "thread_scaling.svg")

WIDTH, HEIGHT = 1100, 480
PANEL_W, PANEL_H = 464, 364
TOP, BOTTOM = 52, 416
PANEL_X = [72, 608]
SERIES = [("ours_gops", "#0b6efd", "LUT-MM AVX-512"),
          ("bitnet_gops", "#d9480f", "BitNet TL2")]


def panel(out, x0, rows, title, y_max):
    threads = [int(r["threads"]) for r in rows]
    out.append(f'<text x="{x0 + PANEL_W / 2}" y="44" text-anchor="middle" '
               f'class="label">{title}</text>')
    for gops in range(0, y_max + 1, 1000):
        y = BOTTOM - gops / y_max * PANEL_H
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + PANEL_W}" '
                   f'y2="{y:.1f}" class="grid"/>')
        if x0 == PANEL_X[0]:
            out.append(f'<text x="{x0 - 10}" y="{y + 4:.1f}" '
                       f'text-anchor="end" class="tick">{gops}</text>')
    out.append(f'<line x1="{x0}" y1="{TOP}" x2="{x0}" y2="{BOTTOM}" '
               f'class="axis"/>')
    out.append(f'<line x1="{x0}" y1="{BOTTOM}" x2="{x0 + PANEL_W}" '
               f'y2="{BOTTOM}" class="axis"/>')

    def tx(i):
        return x0 + i * PANEL_W / (len(threads) - 1)

    for i, t in enumerate(threads):
        out.append(f'<line x1="{tx(i):.1f}" y1="{BOTTOM}" x2="{tx(i):.1f}" '
                   f'y2="{BOTTOM + 5}" stroke="#444"/>')
        out.append(f'<text x="{tx(i):.1f}" y="{BOTTOM + 22}" '
                   f'text-anchor="middle" class="tick">{t}</text>')
    for key, color, _ in SERIES:
        pts = " ".join(
            f"{tx(i):.1f},{BOTTOM - float(r[key]) / y_max * PANEL_H:.1f}"
            for i, r in enumerate(rows))
        out.append(f'<polyline fill="none" stroke="{color}" '
                   f'stroke-width="3" points="{pts}"/>')
        for i, r in enumerate(rows):
            y = BOTTOM - float(r[key]) / y_max * PANEL_H
            out.append(f'<circle cx="{tx(i):.1f}" cy="{y:.1f}" r="4" '
                       f'fill="{color}"/>')


def main():
    with open(CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    shapes = sorted({r["shape"] for r in rows})
    assert len(shapes) == 2, "plot layout expects exactly two shapes"
    peak = max(float(r[k]) for r in rows for k, _, _ in SERIES)
    y_max = (int(peak) // 1000 + 1) * 1000

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
        f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#202124} '
        '.axis{stroke:#444;stroke-width:1} .grid{stroke:#e6e6e6;'
        'stroke-width:1} .label{font-size:13px} .title{font-size:18px;'
        'font-weight:600} .tick{font-size:12px;fill:#555} '
        '.legend{font-size:13px}</style>',
        f'<text x="{WIDTH / 2}" y="24" text-anchor="middle" class="title">'
        'Thread scaling: LUT-MM AVX-512 vs BitNet TL2</text>',
    ]
    for x0, shape in zip(PANEL_X, shapes):
        srows = sorted((r for r in rows if r["shape"] == shape),
                       key=lambda r: int(r["threads"]))
        title = shape.replace("M256_", "M=256 ").replace(
            "K", "K=").replace("_N", " N=")
        panel(out, x0, srows, title, y_max)
    out.append(f'<text x="{WIDTH / 2}" y="466" text-anchor="middle" '
               'class="label">Threads (log2 scale)</text>')
    out.append('<text x="18" y="240" transform="rotate(-90 18 240)" '
               'text-anchor="middle" class="label">Throughput (Gop/s)'
               '</text>')
    for i, (_, color, label) in enumerate(SERIES):
        y = 18 + 20 * i
        out.append(f'<line x1="840" y1="{y}" x2="870" y2="{y}" '
                   f'stroke="{color}" stroke-width="3"/>')
        out.append(f'<text x="878" y="{y + 4}" class="legend">{label}'
                   '</text>')
    out.append('</svg>')
    with open(SVG, "w", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", SVG)


if __name__ == "__main__":
    main()
