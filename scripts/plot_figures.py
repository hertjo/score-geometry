#!/usr/bin/env python3
"""Draw the paper's figures with Splinx from the archived results in results/.

    SPLINX_CLI=/path/to/splinx-cli python scripts/plot_figures.py [results_dir] [figures_dir] [names...]

Every plotted number is read from results/*.csv, except the closed-form curves the captions name (limit
curves, tilted-Gaussian densities) and the spiral sample of panel (a) of the spiral figure, which is redrawn
from the clean density with the seed of scripts/generate_figures.py.
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from figure_style import (CYAN, GRAY, INK, LILAC, MINT, NAVY, SPINE, VIOLET, Ax, labelled_row, legend_box,  # noqa: E402
                          legend_row, log_minor, log_ticks, marker, panel_title, pow10, sx, ticks, two_panels)

ROOT = HERE.parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "figures"
ONLY = set(sys.argv[3:])
sx.OUT = str(OUT)
sx.QUIET = False

TW = 468.0                       # text width of the paper in pt
SIGMA4 = [(CYAN, None, "circle"), (MINT, [4.5, 1.8], "square"), (VIOLET, [1.2, 1.8], "triangle"),
          (NAVY, [6, 1.8, 1.2, 1.8], "diamond")]


def read(name):
    with (DATA / name).open() as f:
        return list(csv.DictReader(f))


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def save(name, W, H, draw):
    if ONLY and name not in ONLY:
        return
    def body():
        draw()
        issues = sx.check(scale=1.0, min_pt=5.5)
        for i in issues:
            print("  CHECK", name, i)
    sx.single(name, W, H, body)


# ---------------------------------------------------------------- tilted posterior
def tilted_posterior():
    W, H = 432.0, 300.0
    lim = (-3.4, 3.4)
    g = np.linspace(*lim, 241)
    U1, U2 = np.meshgrid(g, g)
    r2 = U1 ** 2 + U2 ** 2
    gauss = np.exp(-r2 / 2)
    rows = {r["panel"]: r for r in read("tilted_posterior.csv")}
    panels = [
        ("a", r"isotropic, $m=0$", gauss),
        ("b", r"isotropic, $m=2$", r2 * gauss),
        ("c", r"isotropic, $m=4$", r2 ** 2 * gauss),
        ("d", r"anisotropic, $m=2$", (3 * U1 ** 2 + U2 ** 2) * gauss),
        ("e", r"anisotropic, $m=4$", (U1 ** 4 + 0.1 * U2 ** 4) * gauss),
    ]
    side = 96.0
    cols = [44.0, 44.0 + 144.0, 44.0 + 288.0]
    rws = [30.0, 30.0 + 138.0]
    stops = [(0.0, "#ffffff"), (0.25, "#e4e2ff"), (0.55, LILAC), (0.8, VIOLET), (1.0, NAVY)]

    def draw():
        legend_row([("limiting tangent posterior density", VIOLET, None, None, True, False),
                    (r"covariance $C_{q_m}$ (one standard deviation)", INK, [3.2, 1.6], None),
                    (r"$k+m$", INK, [2.2, 1.4], None)], 0, 9.0, gap=16, center=(0, W))
        # the shading swatch of the first entry
        x0 = W / 2 - (sx.texw("limiting tangent posterior density", 7.4) + sx.texw(r"covariance $C_{q_m}$ (one standard deviation)", 7.4) + sx.texw(r"$k+m$", 7.4) + 3 * 20 + 2 * 16) / 2
        for i, c in enumerate([LILAC, VIOLET, NAVY]):
            sx.rect(x0 + 1 + i * 5, 5.5, 5, 7, fill=c, stroke=None)
        for idx, (lab, title, field) in enumerate(panels):
            px, py = cols[idx % 3], rws[idx // 3] + 12
            A = Ax(px, py, side, side, lim, lim)
            panel_title(lab, title, px - 26, py - 6)
            A.frame(ticks([-3, 0, 3]), ticks([-3, 0, 3]), r"$u_1$", r"$u_2$", grid=False, ylabel_gap=17)
            F = field / field.max()
            levels = list(np.linspace(0.04, 0.96, 12))
            colors = [sx_cmap_hex(l, stops) for l in levels]
            P = sx.Plot(px, py, side, side, lim, lim)
            sx.iso_bands(P, F, levels, colors, xs=g, ys=g)
            sx.line(px, py + side / 2, px + side, py + side / 2, stroke="#ffffff", width=0.4, opacity=70)
            sx.line(px + side / 2, py, px + side / 2, py + side, stroke="#ffffff", width=0.4, opacity=70)
            r = rows[lab]
            cov = np.diag([fnum(r["lambda_1"]), fnum(r["lambda_2"])])
            t = np.linspace(0, 2 * np.pi, 241)
            ev, V = np.linalg.eigh(cov)
            pts = V @ (np.sqrt(ev)[:, None] * np.vstack([np.cos(t), np.sin(t)]))
            A.line(pts[0], pts[1], INK, 1.15, [3.2, 1.6])
            sx.rect(px, py, side, side, fill=None, stroke=SPINE, width=0.7)
        # (f) trace law
        px, py = cols[2], rws[1] + 12
        A = Ax(px, py, side, side, (-0.6, 8.6), (0.0, 11.0))
        panel_title("f", "trace law", px - 26, py - 6)
        A.frame(ticks([0, 2, 4, 6, 8]), ticks([2, 4, 6, 8, 10]), r"vanishing order $m$", r"$\operatorname{tr} C_{q_m}$",
                ylabel_gap=17)
        A.line([-0.6, 8.6], [1.4, 10.6], INK, 0.9, [2.2, 1.4])
        ms = np.arange(0, 9, 2)
        tr = [2.0 * math.gamma(2.0 + m / 2.0) / math.gamma(1.0 + m / 2.0) for m in ms]
        A.marks(ms, tr, "circle", NAVY, 2.4)
    save("tilted-posterior", W, H, draw)


def sx_cmap_hex(v, stops):
    for (a, ca), (b, cb) in zip(stops[:-1], stops[1:]):
        if a <= v <= b:
            t = (v - a) / (b - a)
            c1 = [int(ca[i:i + 2], 16) for i in (1, 3, 5)]
            c2 = [int(cb[i:i + 2], 16) for i in (1, 3, 5)]
            c = [round(x + (y - x) * t) for x, y in zip(c1, c2)]
            return "#%02x%02x%02x" % tuple(c)
    return stops[-1][1]


# ---------------------------------------------------------------- circle spectrum
def circle_spectrum():
    W, H = 384.0, 205.0
    rows = read("circle_spectrum.csv")
    by = defaultdict(list)
    for r in rows:
        by[int(r["r"])].append((fnum(r["sigma"]), fnum(r["tangent_eigenvalue"]), fnum(r["normal_eigenvalue"])))
    spec = [(0, GRAY, "circle"), (1, CYAN, "square"), (2, VIOLET, "triangle"), (4, NAVY, "diamond")]

    def draw():
        s_all = [v[0] for v in by[0]]
        A = Ax(50, 40, W - 62, H - 40 - 32, (min(s_all), max(s_all)), (0, 9.6), xlog=True)
        A.frame(ticks([0.02, 0.05, 0.1, 0.2]), ticks([0, 1, 3, 5, 7, 9]), r"noise scale $\sigma$",
                "normalized covariance eigenvalue", xminor=[0.03, 0.04, 0.06, 0.07, 0.08, 0.09, 0.3])
        for r, col, mk in spec:
            d = np.array(sorted(by[r]))
            A.hline(1 + 2 * r, col, 0.6, [1.2, 1.8])
            A.line(d[:, 0], d[:, 1], col, 1.35)
            A.marks(d[::8, 0], d[::8, 1], mk, col, 2.3)
            if r in (1, 4):
                A.line(d[:, 0], d[:, 2], col, 1.0, [3.2, 1.6])
        legend_row([(r"$r=0$", GRAY, None, "circle"), (r"$r=1$", CYAN, None, "square"),
                    (r"$r=2$", VIOLET, None, "triangle"), (r"$r=4$", NAVY, None, "diamond")], 0, 9, gap=14,
                   center=(A.x0, A.x0 + A.w))
        legend_row([("tangent eigenvalue", INK, None, None), (r"normal eigenvalue ($r=1,4$)", INK, [3.2, 1.6], None),
                    (r"limit $k+m=1+2r$", INK, [1.2, 1.8], None)], 0, 22, gap=14, center=(A.x0, A.x0 + A.w))
    save("circle-spectrum", W, H, draw)


# ---------------------------------------------------------------- crossover
def crossover():
    W, H = 384.0, 200.0
    rows = read("crossover.csv")
    by = defaultdict(list)
    for r in rows:
        by[fnum(r["sigma"])].append((fnum(r["delta_over_sigma2"]), fnum(r["tangent_eigenvalue"])))
    sig = sorted(by)

    def draw():
        A = Ax(46, 34, W - 58, H - 34 - 32, (1e-3, 1e3), (0.9, 3.1), xlog=True)
        A.frame(log_ticks(-3, 3), ticks([1, 1.5, 2, 2.5, 3]), r"density floor $\delta/\sigma^2$",
                "tangent eigenvalue", xminor=[10.0 ** e for e in range(-3, 3)])
        eta = np.geomspace(1e-3, 1e3, 400)
        A.line(eta, (eta + 1.5) / (eta + 0.5), INK, 2.4, opacity=35)
        A.line(eta, (eta + 1.5) / (eta + 0.5), INK, 0.7)
        for i, (s, (col, dash, mk)) in enumerate(zip(sig, SIGMA4)):
            d = np.array(sorted(by[s]))
            A.line(d[:, 0], d[:, 1], col, 1.1, dash)
            A.marks(d[3 * i::12, 0], d[3 * i::12, 1], mk, col, 2.2)
        legend_row([(r"limit $(\lambda+3/2)/(\lambda+1/2)$", INK, None, None)] +
                   [(rf"$\sigma={s:g}$", c, d, m) for s, (c, d, m) in zip(sig, SIGMA4)],
                   0, 12, gap=11, center=(A.x0, A.x0 + A.w))
    save("crossover", W, H, draw)


# ---------------------------------------------------------------- spatial boundary layer
def spatial_boundary_layer():
    W, H = 384.0, 200.0
    rows = read("spatial_boundary_layer.csv")
    by = defaultdict(list)
    for r in rows:
        by[fnum(r["sigma"])].append((fnum(r["angle_over_sigma"]), fnum(r["tangent_rayleigh_quotient"])))
    sig = sorted(by)

    def draw():
        A = Ax(46, 34, W - 58, H - 34 - 32, (0, 4), (0.65, 3.1))
        A.frame([(0, "$0$"), (1, "$1$"), (math.sqrt(3), r"$\sqrt{3}$"), (3, "$3$"), (4, "$4$")],
                ticks([0.75, 1, 1.5, 2, 2.5, 3]), r"distance from the zero $a=\theta/\sigma$", "tangent Rayleigh quotient")
        a = np.linspace(0, 4, 400)
        th = (a ** 4 + 3) / (a ** 2 + 1) ** 2
        A.line(a, th, INK, 2.4, opacity=35)
        A.line(a, th, INK, 0.7)
        for i, (s, (col, dash, mk)) in enumerate(zip(sig, SIGMA4)):
            d = np.array(sorted(by[s]))
            A.line(d[:, 0], d[:, 1], col, 1.1, dash)
            A.marks(d[4 * i::16, 0], d[4 * i::16, 1], mk, col, 2.2)
        legend_row([(r"limit $(a^4+3)/(a^2+1)^2$", INK, None, None)] +
                   [(rf"$\sigma={s:g}$", c, d, m) for s, (c, d, m) in zip(sig, SIGMA4)],
                   0, 12, gap=11, center=(A.x0, A.x0 + A.w))
    save("spatial-boundary-layer", W, H, draw)


# ---------------------------------------------------------------- landing error
def landing_error():
    W, H = 468.0, 196.0
    rows = read("landing_error.csv")
    by = defaultdict(list)
    for r in rows:
        by[fnum(r["sigma"])].append((fnum(r["angle_over_sigma"]), fnum(r["tangent_error_over_sigma"]),
                                     fnum(r["normal_only_error_over_sigma2"])))
    sig = sorted(by)

    def draw():
        xa, xb, pw = two_panels(W)
        A = Ax(xa, 48, pw, H - 48 - 32, (0, 4), (0, 1.1))
        B = Ax(xb, 48, pw, H - 48 - 32, (0, 4), (0, 1.7))
        panel_title("a", "tangent drift of the raw denoiser", A.x0 - 30, A.y0 - 7)
        panel_title("b", "residual after normal-only landing", B.x0 - 30, B.y0 - 7)
        A.frame(ticks([0, 1, 2, 3, 4]), ticks([0, 0.25, 0.5, 0.75, 1]), r"offset $/\,\sigma$", r"tangent error $/\,\sigma$")
        B.frame(ticks([0, 1, 2, 3, 4]), ticks([0, 0.5, 1, 1.5]), r"offset $/\,\sigma$", r"normal residual $/\,\sigma^2$")
        a = np.linspace(0, 4, 400)
        A.line(a, 2 * a / (a ** 2 + 1), INK, 2.4, opacity=35)
        A.line(a, 2 * a / (a ** 2 + 1), INK, 0.7)
        for i, (s, (col, dash, mk)) in enumerate(zip(sig, SIGMA4)):
            d = np.array(sorted(by[s]))
            A.line(d[:, 0], d[:, 1], col, 1.1, dash)
            A.marks(d[5 * i::20, 0], d[5 * i::20, 1], mk, col, 2.2)
            B.line(d[:, 0], np.abs(d[:, 2]), col, 1.1, dash)
            B.marks(d[5 * i::20, 0], np.abs(d[5 * i::20, 2]), mk, col, 2.2)
        legend_row([(r"limit $2a/(a^2+1)$", INK, None, None)] +
                   [(rf"$\sigma={s:g}$", c, d, m) for s, (c, d, m) in zip(sig, SIGMA4)],
                   0, 12, gap=12, center=(0, W))
    save("landing-error", W, H, draw)


# ---------------------------------------------------------------- finite-sample KDE
def finite_sample_kde():
    W, H = 468.0, 222.0
    rows = read("finite_sample_kde_summary.csv")
    sig = sorted({fnum(r["sigma"]) for r in rows})
    cols = dict(zip(sig, [CYAN, MINT, VIOLET, NAVY, INK]))
    mks = {20000: "circle", 80000: "triangle", 320000: "diamond"}

    def draw():
        xa, xb, pw = two_panels(W)
        top = 58.0
        A = Ax(xa, top, pw, H - top - 32, (0.5, 4000), (1e-4, 4), xlog=True, ylog=True)
        B = Ax(xb, top, pw, H - top - 32, (0.5, 4000), (0, 1.05), xlog=True)
        panel_title("a", "sampling and range errors", A.x0 - 34, A.y0 - 7)
        panel_title("b", r"order recovery ($m=2$)", B.x0 - 34, B.y0 - 7)
        xt = [(1, "$1$"), (10, "$10$"), (100, "$100$"), (1000, "$1000$")]
        A.frame(xt, log_ticks(-4, 0), "effective sample size", "finite-sample error", xminor=log_minor(0, 3))
        B.frame(xt, ticks([0, 0.25, 0.5, 0.75, 1]), "effective sample size", "rounded-order recovery",
                xminor=log_minor(0, 3))
        for r in rows:
            c, mk = cols[fnum(r["sigma"])], mks[int(r["n"])]
            e = fnum(r["ess"])
            A.errbar(e, fnum(r["operator_q1"]), fnum(r["operator_q3"]), c)
            A.marks([e], [fnum(r["operator_med"])], mk, c, 2.4, True)
            A.marks([e], [fnum(r["proj_med"])], mk, c, 2.4, False)
            B.errbar(e, fnum(r["order_success_wilson_low"]), fnum(r["order_success_wilson_high"]), c)
            B.marks([e], [fnum(r["order_success"])], mk, c, 2.4, True)
        labelled_row(r"noise $\sigma$:", [(f"${v:g}$", cols[v], None, "circle", True, False) for v in sig], 10.0, W)
        labelled_row(r"sample size $N$:", [(r"$20\mathrm{k}$", INK, None, "circle", True, False),
                                      (r"$80\mathrm{k}$", INK, None, "triangle", True, False),
                                      (r"$320\mathrm{k}$", INK, None, "diamond", True, False)], 25.0, W)
        legend_box([(r"median $\|\widehat H_N-H_\sigma\|_{\mathrm{op}}$", INK, None, "circle", True, False),
                    ("median projector error", INK, None, "circle", False, False)], A.x0 + 4, A.y0 + A.h - 4, anchor="sw")
    save("finite-sample-kde", W, H, draw)


# ---------------------------------------------------------------- conditioning
def conditioning():
    W, H = 468.0, 214.0
    rows = read("conditioning.csv")
    wrows = read("conditioning_stability_width.csv")
    names = ["calibrated inner + normal-only landing", "raw denoiser + raw inner Jacobian",
             "raw denoiser + calibrated inner step", "raw additive preconditioner"]
    spec = {names[0]: (NAVY, None, "circle", "calibrated inner step and landing"),
            names[1]: (CYAN, [4.5, 1.8], "square", "raw outer and raw inner"),
            names[2]: (VIOLET, [1.2, 1.8], "triangle", "raw outer, calibrated inner"),
            names[3]: (MINT, [6, 1.8, 1.2, 1.8], "diamond", "raw additive metric")}
    g = np.array([fnum(r["step_size"]) for r in rows])
    orders = np.array([fnum(r["vanishing_order"]) for r in wrows])

    def draw():
        xa, xb, pw = two_panels(W)
        top = 62.0
        A = Ax(xa, top, pw, H - top - 32, (0, 2), (0.03, 200), ylog=True)
        B = Ax(xb, top, pw, H - top - 32, (0, 20), (1e-3, 3), ylog=True)
        panel_title("a", r"spectral radius, $C_q=\operatorname{diag}(9,1)$", A.x0 - 34, A.y0 - 7)
        panel_title("b", "width of the stable step-size interval", B.x0 - 34, B.y0 - 7)
        A.frame(ticks([0, 0.5, 1, 1.5, 2]), [(0.1, "$0.1$"), (1, "$1$"), (10, "$10$"), (100, "$100$")],
                r"step size $\gamma$", "spectral radius", yminor=[0.05, 0.5, 5, 50])
        B.frame(ticks([0, 4, 8, 12, 16, 20]), log_ticks(-3, 0), r"vanishing order $m$", "interval width",
                yminor=log_minor(-3, 0))
        A.hline(1.0, INK, 0.8, [2, 2])
        for i, n in enumerate(names):
            col, dash, mk, _ = spec[n]
            v = np.array([fnum(r[n]) for r in rows])
            A.line(g, v, col, 1.3, dash)
            A.marks(g[::250], v[::250], mk, col, 2.3)
            wv = np.array([fnum(r[n]) for r in wrows])
            B.line(orders, wv, col, 1.3, dash)
            sl = slice(i % 2, None, 2)
            B.marks(orders[sl], wv[sl], mk, col, 2.3)
        legend_row([(spec[n][3], spec[n][0], spec[n][1], spec[n][2]) for n in names[:2]] +
                   [("stability threshold", INK, [2, 2], None)], 0, 11, gap=14, center=(0, W))
        legend_row([(spec[n][3], spec[n][0], spec[n][1], spec[n][2]) for n in names[2:]], 0, 26, gap=14,
                   center=(0, W))
    save("conditioning", W, H, draw)


# ---------------------------------------------------------------- learned figures
def summary(rows, problem, field):
    sel = sorted([r for r in rows if r["problem"] == problem], key=lambda r: fnum(r["sigma"]))
    x = np.array([fnum(r["sigma"]) for r in sel])
    return (x, np.array([fnum(r[f"{field}_median"]) for r in sel]), np.array([fnum(r[f"{field}_q1"]) for r in sel]),
            np.array([fnum(r[f"{field}_q3"]) for r in sel]))


def learned_geometry():
    W, H = 468.0, 240.0
    rows = read("learned_score_summary.csv")

    def draw():
        xa, xb, pw = two_panels(W)
        top = 84.0
        A = Ax(xa, top, pw, H - top - 32, (0.072, 0.188), (0, 3.4))
        B = Ax(xb, top, pw, H - top - 32, (0.072, 0.188), (1e-3, 5), ylog=True)
        panel_title("a", "learned circle spectrum", A.x0 - 30, 12)
        panel_title("b", r"learned range recovery at the zero", B.x0 - 34, 12)
        xt = ticks([0.08, 0.12, 0.18])
        A.frame(xt, ticks([0, 1, 2, 3]), r"training noise $\sigma$", "largest learned-link eigenvalue")
        B.frame(xt, log_ticks(-3, 0), r"training noise $\sigma$", "operator-norm error", yminor=log_minor(-3, 0))
        for prob, col, mk in (("circle_zero", VIOLET, "circle"), ("circle_regular", CYAN, "square")):
            x, med, lo, hi = summary(rows, prob, "top_eigenvalue")
            A.band(x, lo, hi, col, 22)
            _, tru, _, _ = summary(rows, prob, "true_top_eigenvalue")
            A.line(x, tru, col, 1.1, [3.2, 1.6])
            A.line(x, med, col, 1.4)
            A.marks(x, med, mk, col, 2.4)
        spec = [("raw_projector_error", NAVY, None, "circle", "raw link against the tangent projector"),
                ("hard_projector_error", MINT, [4.5, 1.8], "square", "hard-calibrated range error"),
                ("population_jacobian_error", VIOLET, [1.2, 1.8], "triangle", "learned link against the population link"),
                ("coupling_block_norm", CYAN, [6, 1.8, 1.2, 1.8], "diamond", r"coupling block $\|P^\perp\widehat A_\theta P\|_{\mathrm{op}}$")]
        for f, col, dash, mk, _ in spec:
            x, med, lo, hi = summary(rows, "circle_zero", f)
            B.band(x, np.maximum(lo, 1e-3), np.maximum(hi, 1e-3), col, 18)
            B.line(x, np.maximum(med, 1e-3), col, 1.3, dash)
            B.marks(x, np.maximum(med, 1e-3), mk, col, 2.3)
        ya, dy = 29.0, 12.5
        legend_row([("quadratic zero", VIOLET, None, "circle"), ("positive density", CYAN, None, "square")],
                   A.x0 - 30, ya, gap=12)
        legend_row([("learned median", INK, None, None), ("population (quadrature)", INK, [3.2, 1.6], None)],
                   A.x0 - 30, ya + dy, gap=12)
        for i, (f, col, dash, mk, lab) in enumerate(spec):
            legend_row([(lab, col, dash, mk)], B.x0 - 34, ya + dy * i, gap=10)
    save("learned-geometry", W, H, draw)


def learned_plane():
    W, H = 468.0, 248.0
    rows = read("learned_score_summary.csv")

    def draw():
        xa, xb, pw = two_panels(W)
        top = 96.0
        A = Ax(xa, top, pw, H - top - 32, (0.11, 0.26), (0, 5.2))
        B = Ax(xb, top, pw, H - top - 32, (0.11, 0.26), (1e-3, 6), ylog=True)
        panel_title("a", r"learned spectrum ($d=16$, $k=2$)", A.x0 - 30, 12)
        panel_title("b", "tangent-range recovery", B.x0 - 34, 12)
        xt = ticks([0.12, 0.18, 0.25])
        A.frame(xt, ticks([0, 1, 2, 3, 4, 5]), r"training noise $\sigma$", "tangent eigenvalue")
        B.frame(xt, log_ticks(-3, 0), r"training noise $\sigma$", "operator-norm error", yminor=log_minor(-3, 0))
        for f, tf, col, mk in (("top_eigenvalue", "true_top_eigenvalue", VIOLET, "circle"),
                               ("second_eigenvalue", "true_second_eigenvalue", MINT, "square")):
            x, med, lo, hi = summary(rows, "plane_zero", f)
            _, tru, _, _ = summary(rows, "plane_zero", tf)
            A.band(x, lo, hi, col, 22)
            A.line(x, tru, col, 1.1, [3.2, 1.6])
            A.line(x, med, col, 1.4)
            A.marks(x, med, mk, col, 2.4)
        spec = [("raw_projector_error", NAVY, None, "circle", "raw link against the tangent projector"),
                ("hard_projector_error", MINT, [4.5, 1.8], "square", "hard top-two projector"),
                ("soft_projector_error", CYAN, [2.5, 1.5], "triangle_down", r"soft calibration $\Phi_{0.25}$"),
                ("population_jacobian_error", VIOLET, [1.2, 1.8], "triangle", "learned link against the population link"),
                ("coupling_block_norm", GRAY, [6, 1.8, 1.2, 1.8], "diamond",
                 r"coupling block $\|P^\perp\widehat A_\theta P\|_{\mathrm{op}}$")]
        for f, col, dash, mk, _ in spec:
            x, med, lo, hi = summary(rows, "plane_zero", f)
            B.band(x, np.maximum(lo, 1e-3), np.maximum(hi, 1e-3), col, 18)
            B.line(x, np.maximum(med, 1e-3), col, 1.3, dash)
            B.marks(x, np.maximum(med, 1e-3), mk, col, 2.3)
        ya, dy = 29.0, 12.5
        legend_row([("first eigenvalue", VIOLET, None, "circle"), ("second eigenvalue", MINT, None, "square")],
                   A.x0 - 30, ya, gap=12)
        legend_row([("learned median", INK, None, None), ("population (exact)", INK, [3.2, 1.6], None)],
                   A.x0 - 30, ya + dy, gap=12)
        for i, sp in enumerate(spec):
            legend_row([(sp[4], sp[1], sp[2], sp[3])], B.x0 - 34, ya + dy * i, gap=10)
    save("learned-plane", W, H, draw)


def stable_intervals(eigenvalues):
    raw_lo = max((c - 1.0) / (c * c) for c in eigenvalues)
    raw_hi = min((c + 1.0) / (c * c) for c in eigenvalues)
    inner_lo = 1.0 - 1.0 / max(eigenvalues)
    inner_hi = 1.0 + 1.0 / max(eigenvalues)
    return {"raw": (raw_lo, raw_hi), "inner_only": (inner_lo, inner_hi)}


def learned_optimization():
    W, H = 468.0, 222.0
    rows = read("learned_optimization_summary.csv")
    srows = read("learned_score_summary.csv")
    cols = {"oracle": GRAY, "raw": CYAN, "inner_only": VIOLET, "calibrated": NAVY}
    dashes = {"oracle": [3.2, 1.6], "raw": [4.5, 1.8], "inner_only": [1.2, 1.8], "calibrated": None}
    labels = {"calibrated": "SC-DRGD (calibrated)", "oracle": "Riemannian oracle", "inner_only": "inner calibrated only",
              "raw": "fully raw"}

    def pop(problem, sigma):
        r = [r for r in srows if r["problem"] == problem and abs(fnum(r["sigma"]) - sigma) < 1e-12][0]
        ev = [fnum(r["true_top_eigenvalue_median"])]
        if problem == "plane_zero":
            ev.append(fnum(r["true_second_eigenvalue_median"]))
        return ev

    def draw():
        xa, xb, pw = two_panels(W)
        top = 62.0
        panels = [("circle_zero", 0.12, xa, (0.05, 1.0), 35, "circle, 35 iterations",
                   ticks([0.05, 0.25, 0.5, 0.75, 1.0])),
                  ("plane_zero", 0.18, xb, (0.05, 1.2), 30, r"rotated plane ($d=16$), 30 iterations",
                   ticks([0.05, 0.3, 0.6, 0.9, 1.2]))]
        for i, (prob, s, x0, xlim, it, title, xt) in enumerate(panels):
            A = Ax(x0, top + 14, pw, H - top - 14 - 32, xlim, (1e-8, 1.0), ylog=True)
            panel_title("ab"[i], title, x0 - 34, top - 7)
            A.frame(xt, [(10.0 ** e, pow10(e)) for e in (-8, -6, -4, -2, 0)], r"step size $\gamma$",
                    "final objective gap", yminor=[10.0 ** e for e in (-7, -5, -3, -1)])
            iv = stable_intervals(pop(prob, s))
            bars = [("raw",) + iv["raw"], ("inner_only",) + iv["inner_only"]]
            overlap = bars[0][2] > bars[1][1] and bars[1][2] > bars[0][1]
            for j, (m, lo, hi) in enumerate(bars):
                y = A.y0 - 7.5 - (j * 6.0 if overlap else 0.0)
                a, b = float(A.X(max(lo, xlim[0]))), float(A.X(min(hi, xlim[1])))
                if b > a:
                    sx.rect(a, y - 1.3, b - a, 2.6, fill=cols[m], stroke=None)
                    for e in (a, b):
                        sx.line(e, y - 3.0, e, y + 3.0, stroke=cols[m], width=0.8)
            sx.tex(r"stable $\gamma$", A.x0 - 4, A.y0 - 7.5, 6.6, "#4a4766", anchor="right", valign="middle")
            for m in ("raw", "inner_only", "oracle", "calibrated"):
                sel = sorted([r for r in rows if r["problem"] == prob and r["method"] == m
                              and abs(fnum(r["sigma"]) - s) < 1e-12 and int(r["iteration"]) == it],
                             key=lambda r: fnum(r["gamma"]))
                x = np.array([fnum(r["gamma"]) for r in sel])
                med, lo, hi = [np.minimum(np.maximum([fnum(r[k]) for r in sel], 1e-8), 1.0)
                               for k in ("objective_median", "objective_q1", "objective_q3")]
                A.band(x, lo, hi, cols[m], 18)
                A.line(x, med, cols[m], 1.35, dashes[m])
        legend_row([(labels[m], cols[m], dashes[m], None) for m in ("calibrated", "oracle", "inner_only", "raw")],
                   0, 11, gap=16, center=(0, W))
        sx.tex(r"values at or below $10^{-8}$ are shown at the display floor", W / 2, 27, 7.0, "#4a4766",
               anchor="center", valign="middle")
    save("learned-optimization", W, H, draw)


# ---------------------------------------------------------------- spiral
SPIRAL_A, SPIRAL_B = 0.6, 0.35
SPIRAL_T0, SPIRAL_T1 = 0.0, 4.0 * math.pi


def spiral_point(t):
    r = SPIRAL_A + SPIRAL_B * t
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=-1)


def spiral_speed(t):
    r = SPIRAL_A + SPIRAL_B * t
    return np.sqrt(SPIRAL_B ** 2 + r ** 2)


def spiral_samples(t_star, count=1400, seed=20260821):
    """The sample of scripts/generate_figures.py: inverse-CDF draws from the clean density."""
    rng = np.random.default_rng(seed)
    t = np.linspace(SPIRAL_T0, SPIRAL_T1, 20_001)
    speed = spiral_speed(t)
    arc = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(t))])
    arc_star = float(np.interp(t_star, t, arc))
    weight = (arc - arc_star) ** 2 * speed
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (weight[1:] + weight[:-1]) * np.diff(t))])
    cdf /= cdf[-1]
    draws = np.interp(rng.random(count), cdf, t)
    return spiral_point(draws)


def spiral():
    W, H = 468.0, 200.0
    rows = read("spiral.csv")

    def draw():
        side = 108.0
        top = 56.0
        xs = [36.0, 36 + side + 50, 36 + 2 * (side + 50)]
        A = Ax(xs[0], top, side, side, (-5.6, 5.6), (-5.6, 5.6))
        B = Ax(xs[1], top, side, side, (0.004, 0.024), (2.9985, 3.0004))
        C = Ax(xs[2], top, side, side, (0, 1.35), (0, 3.4))
        panel_title("a", "sample and zero", xs[0] - 28, top - 7)
        panel_title("b", "limit is curvature free", xs[1] - 36, top - 7)
        panel_title("c", r"remainder against $\kappa^2$", xs[2] - 28, top - 7)
        A.frame(ticks([-4, 0, 4]), ticks([-4, 0, 4]), r"$x_1$", r"$x_2$", ylabel_gap=16)
        guide = spiral_point(np.linspace(SPIRAL_T0, SPIRAL_T1, 2000))
        A.line(guide[:, 0], guide[:, 1], LILAC, 0.9, clamp=False)
        pts = spiral_samples(9.0)
        P = np.c_[A.X(pts[:, 0]), A.Y(pts[:, 1])]
        inside = (P[:, 0] > A.x0) & (P[:, 0] < A.x0 + side) & (P[:, 1] > A.y0) & (P[:, 1] < A.y0 + side)
        sx.dots(P[inside], r=0.75, col=NAVY, opacity=55)
        for t_star, mk, shown in ((9.0, "square", True), (2.2, "circle", False), (12.0, "triangle", False)):
            q = spiral_point(np.asarray(t_star))
            marker(float(A.X(q[0])), float(A.Y(q[1])), mk, MINT if shown else VIOLET, 3.2 if shown else 2.6, shown,
                   edge=1.0)
        stars = sorted({fnum(r["t_star"]) for r in rows})
        kap = {t: fnum([r for r in rows if fnum(r["t_star"]) == t][0]["curvature"]) for t in stars}
        B.frame([(0.005, "$0.005$"), (0.01, "$0.01$"), (0.02, "$0.02$")],
                [(2.999, "$2.999$"), (2.9995, "$2.9995$"), (3.0, "$3$")], r"noise scale $\sigma$", "tangent eigenvalue",
                ylabel_gap=33)
        B.hline(3.0, INK, 0.9, [3.2, 1.6])
        curves = ((1.0, VIOLET, None, "circle"), (4.5, CYAN, [4.5, 1.8], "triangle"), (12.0, NAVY, [1.2, 1.8], "diamond"))
        for t, col, dash, mk in curves:
            sel = sorted([r for r in rows if fnum(r["t_star"]) == t], key=lambda r: fnum(r["sigma"]))
            x = [fnum(r["sigma"]) for r in sel]
            y = [fnum(r["tangent_eigenvalue"]) for r in sel]
            B.line(x, y, col, 1.3, dash)
            B.marks(x, y, mk, col, 2.3)
        C.frame(ticks([0, 0.5, 1]), ticks([0, 1, 2, 3]), r"curvature squared $\kappa^2$", r"remainder $/\,\sigma^2$",
                ylabel_gap=14)
        k2 = np.array([kap[t] ** 2 for t in stars])
        pick = lambda t, key: fnum([r for r in rows if fnum(r["t_star"]) == t and fnum(r["sigma"]) == 0.005][0][key])
        for key, col, mk, slope in (("tangent_deficit_over_sigma2", NAVY, "circle", 2.5),
                                    ("normal_over_sigma2", MINT, "square", 1.5)):
            C.line([0, 1.35], [0, slope * 1.35], col, 0.8, [3.2, 1.6])
            C.marks(k2, [pick(t, key) for t in stars], mk, col, 2.4)
        legend_row([(rf"$\kappa={kap[t]:.2f}$", c, d, m) for t, c, d, m in curves] +
                   [("predicted limit $k+m=3$", INK, [3.2, 1.6], None)], 0, 11, gap=14, center=(0, W))
        legend_row([("tangent deficit, slope $2.5$", NAVY, [3.2, 1.6], "circle"),
                    ("normal eigenvalue, slope $1.5$", MINT, [3.2, 1.6], "square"),
                    ("zero of the density", MINT, None, "square", True, False),
                    ("other base points", VIOLET, None, "circle", False, False)], 0, 26, gap=14, center=(0, W))
    save("spiral", W, H, draw)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (tilted_posterior, circle_spectrum, crossover, spatial_boundary_layer, landing_error,
               finite_sample_kde, conditioning, learned_geometry, learned_plane, learned_optimization, spiral):
        fn()
