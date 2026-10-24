#!/usr/bin/env python3
"""Generate the figures that need no trained network.

Population quantities come from posterior quadrature; the finite-sample
experiment uses fixed seeds.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from reportlab import rl_config
# Deterministic output, so regenerated figures match byte for byte.
rl_config.invariant = 1
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen.canvas import Canvas


ROOT = Path(__file__).resolve().parents[1]
# Quick-look plots drawn alongside the experiments; the paper figures are drawn by plot_figures.py.
FIG = ROOT / "output" / "draft-figures"
DATA = ROOT / "results"
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(exist_ok=True)

# A restrained, colorblind-safe palette.  Every color is paired with a line
# style or marker so that the figures remain legible in grayscale.
BLUE = HexColor("#285C86")
ORANGE = HexColor("#C45A00")
TEAL = HexColor("#007F7B")
PURPLE = HexColor("#70539B")
GRAY = HexColor("#68717D")
LIGHT = HexColor("#DCE1E8")
BLACK = HexColor("#20242B")

SIGMA_STYLES = [
    (PURPLE, None, "circle"),
    (BLUE, [5, 2], "square"),
    (TEAL, [1.2, 2.0], "triangle"),
    (ORANGE, [7, 2, 1.2, 2], "diamond"),
]


def circle_moments(sigma: float, order_r: int, floor: float = 0.0, n: int = 200_001):
    """Return posterior mean and H=Cov/sigma^2 at y=(1,0) on the circle.

    The clean density is proportional to floor + (1-cos(theta))**order_r.
    Normalization of that density cancels in the posterior weights.
    """
    theta = np.linspace(-math.pi, math.pi, n, endpoint=False)
    one_minus_cos = 1.0 - np.cos(theta)
    rho = floor + (one_minus_cos**order_r if order_r else np.ones_like(theta))
    log_kernel = -one_minus_cos / (sigma * sigma)
    log_kernel -= np.max(log_kernel)
    weights = rho * np.exp(log_kernel)
    weights /= np.sum(weights)
    points = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    mean = np.sum(weights[:, None] * points, axis=0)
    centered = points - mean
    cov = (centered.T * weights) @ centered
    return mean, cov / (sigma * sigma)


def circle_moments_at(sigma: float, order_r: int, query_angle: float,
                      n: int = 200_001):
    """Tangent/normal Rayleigh quotients near the density zero on the circle."""
    theta = np.linspace(-math.pi, math.pi, n, endpoint=False)
    rho = (1.0 - np.cos(theta))**order_r
    squared_distance = 2.0 * (1.0 - np.cos(theta - query_angle))
    log_kernel = -squared_distance / (2.0 * sigma * sigma)
    log_kernel -= np.max(log_kernel)
    weights = rho * np.exp(log_kernel)
    weights /= np.sum(weights)
    points = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    mean = np.sum(weights[:, None] * points, axis=0)
    centered = points - mean
    cov = (centered.T * weights) @ centered / (sigma * sigma)
    tangent = np.array([-math.sin(query_angle), math.cos(query_angle)])
    normal = np.array([math.cos(query_angle), math.sin(query_angle)])
    return float(tangent @ cov @ tangent), float(normal @ cov @ normal)


def circle_denoiser_error_at(sigma: float, order_r: int, query_angle: float,
                             n: int = 200_001):
    """Tangential and normal errors of the population denoiser on S^1."""
    theta = np.linspace(-math.pi, math.pi, n, endpoint=False)
    rho = (1.0 - np.cos(theta))**order_r
    squared_distance = 2.0 * (1.0 - np.cos(theta - query_angle))
    log_kernel = -squared_distance / (2.0 * sigma * sigma)
    log_kernel -= np.max(log_kernel)
    weights = rho * np.exp(log_kernel)
    weights /= np.sum(weights)
    points = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    mean = np.sum(weights[:, None] * points, axis=0)
    query = np.array([math.cos(query_angle), math.sin(query_angle)])
    tangent = np.array([-math.sin(query_angle), math.cos(query_angle)])
    normal = query
    error = mean - query
    return float(tangent @ error), float(normal @ error)


def _map(v, lo, hi, a, b, log=False):
    v = np.asarray(v, dtype=float)
    if log:
        v = np.log10(np.maximum(v, lo))
        lo, hi = math.log10(lo), math.log10(hi)
    return a + (v - lo) * (b - a) / (hi - lo)


def _tick_label(value):
    value = float(value)
    if value and (abs(value) < 1e-3 or abs(value) >= 1e4):
        return f"{value:.0e}".replace("e+0", "e").replace("e-0", "e-")
    return f"{value:g}"


def _axes(c: Canvas, x0, y0, w, h, xlim, ylim, xlabel, ylabel,
          xlog=False, ylog=False, xticks=None, yticks=None,
          xtick_labels=None, ytick_labels=None, grid=True):
    """Draw the axes, frame and tick labels; grid=False omits the rules."""
    c.setFont("Helvetica", 10.2)
    if xticks is None:
        xticks = np.geomspace(*xlim, 5) if xlog else np.linspace(*xlim, 5)
    if yticks is None:
        yticks = np.geomspace(*ylim, 5) if ylog else np.linspace(*ylim, 5)
    if xtick_labels is None:
        xtick_labels = [_tick_label(tick) for tick in xticks]
    if ytick_labels is None:
        ytick_labels = [_tick_label(tick) for tick in yticks]
    if grid:
        c.setStrokeColor(LIGHT)
        c.setLineWidth(0.45)
        for tick in xticks:
            xx = float(_map(tick, *xlim, x0, x0 + w, log=xlog))
            c.line(xx, y0, xx, y0 + h)
        for tick in yticks:
            yy = float(_map(tick, *ylim, y0, y0 + h, log=ylog))
            c.line(x0, yy, x0 + w, yy)
    c.setStrokeColor(BLACK)
    c.setLineWidth(0.75)
    c.line(x0, y0, x0 + w, y0)
    c.line(x0, y0, x0, y0 + h)
    c.setFillColor(BLACK)
    for tick, label in zip(xticks, xtick_labels):
        xx = float(_map(tick, *xlim, x0, x0 + w, log=xlog))
        c.line(xx, y0 - 2.3, xx, y0)
        c.drawCentredString(xx, y0 - 12, label)
    for tick, label in zip(yticks, ytick_labels):
        yy = float(_map(tick, *ylim, y0, y0 + h, log=ylog))
        c.line(x0 - 2, yy, x0, yy)
        c.drawRightString(x0 - 4.5, yy - 2.5, label)
    c.setFillColor(BLACK)
    c.setFont("Helvetica", 11.0)
    c.drawCentredString(x0 + w / 2, y0 - 27, xlabel)
    c.saveState()
    # Keep the rotated label clear of wide tick labels.
    widest = max((pdfmetrics.stringWidth(str(t), "Helvetica", 10.2)
                  for t in ytick_labels), default=0.0)
    c.translate(x0 - max(38.0, widest + 20.0), y0 + h / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, ylabel)
    c.restoreState()


def _marker(c, x, y, marker, color, size=2.4, filled=True):
    c.setStrokeColor(color)
    c.setFillColor(color if filled else HexColor("#FFFFFF"))
    c.setLineWidth(0.9)
    if marker == "circle":
        c.circle(x, y, size, fill=1, stroke=1)
    elif marker == "square":
        c.rect(x - size, y - size, 2 * size, 2 * size, fill=1, stroke=1)
    elif marker == "triangle":
        p = c.beginPath()
        p.moveTo(x, y + 1.2 * size)
        p.lineTo(x - 1.1 * size, y - .9 * size)
        p.lineTo(x + 1.1 * size, y - .9 * size)
        p.close()
        c.drawPath(p, fill=1, stroke=1)
    elif marker == "diamond":
        p = c.beginPath()
        p.moveTo(x, y + 1.2 * size)
        p.lineTo(x - 1.1 * size, y)
        p.lineTo(x, y - 1.2 * size)
        p.lineTo(x + 1.1 * size, y)
        p.close()
        c.drawPath(p, fill=1, stroke=1)


def _curve(c: Canvas, xs, ys, x0, y0, w, h, xlim, ylim, color,
           xlog=False, ylog=False, width=1.7, dash=None, marker=None,
           marker_stride=None, marker_offset=0):
    xx = _map(xs, *xlim, x0, x0 + w, log=xlog)
    yy = _map(ys, *ylim, y0, y0 + h, log=ylog)
    p = c.beginPath()
    p.moveTo(float(xx[0]), float(yy[0]))
    for a, b in zip(xx[1:], yy[1:]):
        p.lineTo(float(a), float(b))
    c.saveState()
    clip = c.beginPath()
    clip.rect(x0, y0, w, h)
    c.clipPath(clip, stroke=0, fill=0)
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.setLineCap(1)
    c.setLineJoin(1)
    c.setDash(dash or [])
    c.drawPath(p)
    c.setDash([])
    if marker and marker_stride:
        for index in range(marker_offset, len(xx), marker_stride):
            _marker(c, float(xx[index]), float(yy[index]), marker, color, size=2.0)
    c.restoreState()


def _legend_grid(c, entries, x, y, columns, column_width, row_height=13,
                 font_size=9.2):
    """Draw a row-major legend above the axes."""
    c.setFont("Helvetica", font_size)
    for index, entry in enumerate(entries):
        label, color, dash = entry[:3]
        marker = entry[3] if len(entry) > 3 else None
        filled = entry[4] if len(entry) > 4 else True
        column, row = index % columns, index // columns
        xx = x + column * column_width
        yy = y - row * row_height
        c.setStrokeColor(color)
        c.setLineWidth(1.7)
        c.setDash(dash or [])
        c.line(xx, yy, xx + 16, yy)
        c.setDash([])
        if marker:
            _marker(c, xx + 8, yy, marker, color, size=2.1, filled=filled)
        c.setFillColor(BLACK)
        c.drawString(xx + 21, yy - 2.5, label)


def _panel_title(c, x, y, text):
    c.setFillColor(BLACK)
    c.setFont("Helvetica-Bold", 11.5)
    c.drawString(x, y, text)


def _error_bar(c, x, low, high, color, cap=2.3):
    c.setStrokeColor(color)
    c.setLineWidth(0.85)
    c.line(x, low, x, high)
    c.line(x - cap, low, x + cap, low)
    c.line(x - cap, high, x + cap, high)




def _auto_box(c: Canvas, x0, y0, w, h, yticks=None, ytick_labels=None,
              top_pad=10.0):
    """Frame one panel, wide enough to contain its axis labels."""
    labels = ytick_labels if ytick_labels else (
        [_tick_label(t) for t in yticks] if yticks is not None else [])
    widest = max((pdfmetrics.stringWidth(str(t), "Helvetica", 10.2)
                  for t in labels), default=0.0)
    offset = max(38.0, widest + 20.0)
    _panel_box(c, x0 - offset - 8.0, y0 - 36.0,
               w + offset + 18.0, h + 36.0 + top_pad)


def _grid_only(c: Canvas, x0, y0, w, h, xlim, ylim, xticks, yticks):
    """Interior grid rules, drawn before a field so they sit underneath."""
    c.saveState()
    c.setStrokeColor(LIGHT)
    c.setLineWidth(0.45)
    for tick in xticks:
        xx = float(_map(tick, *xlim, x0, x0 + w))
        c.line(xx, y0, xx, y0 + h)
    for tick in yticks:
        yy = float(_map(tick, *ylim, y0, y0 + h))
        c.line(x0, yy, x0 + w, yy)
    c.restoreState()


def _heatmap(c: Canvas, values, x0, y0, w, h, color):
    """Render a scalar field as a grid of filled cells."""
    rows, cols = values.shape
    peak = float(values.max())
    if peak <= 0:
        return
    cw, ch = w / cols, h / rows
    red, green, blue = color.red, color.green, color.blue
    c.saveState()
    clip = c.beginPath()
    clip.rect(x0, y0, w, h)
    c.clipPath(clip, stroke=0, fill=0)
    for i in range(rows):
        yy = y0 + i * ch
        for j in range(cols):
            level = float(values[i, j]) / peak
            if level < 0.012:
                continue
            t = level ** 0.65
            c.setFillColorRGB(1.0 - t * (1.0 - red),
                              1.0 - t * (1.0 - green),
                              1.0 - t * (1.0 - blue))
            # Slight overlap avoids hairline seams between cells.
            c.rect(x0 + j * cw - 0.25, yy - 0.25,
                   cw + 0.5, ch + 0.5, stroke=0, fill=1)
    c.restoreState()


def _ellipse(c: Canvas, cov, x0, y0, w, h, xlim, ylim, color, scale=1.0):
    """Draw the covariance ellipse of a symmetric 2x2 matrix."""
    eigenvalues, vectors = np.linalg.eigh(cov)
    angle = np.linspace(0.0, 2.0 * math.pi, 241)
    unit = np.stack([np.cos(angle), np.sin(angle)])
    points = vectors @ (np.sqrt(eigenvalues)[:, None] * unit) * scale
    xs = _map(points[0], *xlim, x0, x0 + w)
    ys = _map(points[1], *ylim, y0, y0 + h)
    path = c.beginPath()
    path.moveTo(float(xs[0]), float(ys[0]))
    for xx, yy in zip(xs[1:], ys[1:]):
        path.lineTo(float(xx), float(yy))
    path.close()
    c.saveState()
    c.setStrokeColor(color)
    c.setLineWidth(1.5)
    c.setDash(4, 2)
    c.drawPath(path, stroke=1, fill=0)
    c.restoreState()


def _panel_box(c: Canvas, x, y, w, h, radius=5.0):
    """Rounded frame around one subfigure."""
    c.saveState()
    c.setStrokeColor(LIGHT)
    c.setLineWidth(0.6)
    c.roundRect(x, y, w, h, radius, stroke=1, fill=0)
    c.restoreState()


def make_tilted_posterior():
    """Mechanism figure: the leading density jet survives the rescaling."""
    lim = (-3.4, 3.4)
    ticks = [-3, 0, 3]
    grid = np.linspace(*lim, 80)
    U1, U2 = np.meshgrid(grid, grid)
    radius2 = U1 ** 2 + U2 ** 2
    gauss = np.exp(-radius2 / 2.0)
    rows = []

    panels = [
        ("(a) Isotropic, m = 0", gauss, np.eye(2), 0),
        ("(b) Isotropic, m = 2", radius2 * gauss, 2.0 * np.eye(2), 2),
        ("(c) Isotropic, m = 4", radius2 ** 2 * gauss, 3.0 * np.eye(2), 4),
        ("(d) Anisotropic, m = 2", (3.0 * U1 ** 2 + U2 ** 2) * gauss,
         np.diag([2.5, 1.5]), 2),
        ("(e) Anisotropic, m = 4", (U1 ** 4 + 0.1 * U2 ** 4) * gauss,
         np.diag([4.63636363636, 1.36363636364]), 4),
    ]

    path = FIG / "tilted_posterior.pdf"
    c = Canvas(str(path), pagesize=(590, 442))
    box_w, box_h, gap_x = 186.0, 168.0, 8.0
    xs = [8.0, 8.0 + box_w + gap_x, 8.0 + 2.0 * (box_w + gap_x)]
    ys = [212.0, 26.0]
    boxes = [(x, y) for y in ys for x in xs]
    for bx, by in boxes:
        _panel_box(c, bx, by, box_w, box_h)
    sq = box_w - 74.0

    for (bx, by), (title, field, cov, order) in zip(boxes, panels):
        x0, y0 = bx + 62.0, by + 42.0
        _heatmap(c, field, x0, y0, sq, sq, BLUE)
        _axes(c, x0, y0, sq, sq, lim, lim, "u1", "u2",
              xticks=ticks, yticks=ticks)
        _ellipse(c, cov, x0, y0, sq, sq, lim, lim, TEAL)
        rows.append({"panel": title[1], "order_m": order, "k": 2,
                     "lambda_1": float(cov[0, 0]), "lambda_2": float(cov[1, 1]),
                     "trace": float(np.trace(cov)), "k_plus_m": 2 + order})

    # (f) trace law
    fx, fy = boxes[5][0] + 62.0, boxes[5][1] + 42.0
    degrees = np.arange(0, 9, 2)
    traces = [2.0 * math.gamma(2.0 + m / 2.0) / math.gamma(1.0 + m / 2.0)
              for m in degrees]
    _axes(c, fx, fy, sq, sq, (-0.4, 8.4), (0.0, 11.0),
          "vanishing order m", "trace of the limit",
          xticks=[0, 4, 8], yticks=[2, 6, 10])
    _curve(c, np.asarray([-0.4, 8.4]), np.asarray([1.6, 10.4]),
           fx, fy, sq, sq, (-0.4, 8.4), (0.0, 11.0), BLACK, width=1.0, dash=[3, 2])
    _curve(c, degrees.astype(float), np.asarray(traces),
           fx, fy, sq, sq, (-0.4, 8.4), (0.0, 11.0), BLUE,
           width=0.0, marker="circle", marker_stride=1)

    titles = [t[0] for t in panels] + ["(f) Trace law"]
    for (bxx, byy), text in zip(boxes, titles):
        _panel_title(c, bxx + 6, byy + box_h + 6, text)

    legend = [("posterior density", BLUE, None),
              ("covariance ellipse", TEAL, [4, 2]),
              ("k + m", BLACK, [3, 2])]
    entry_width = 21.0 + max(pdfmetrics.stringWidth(text, "Helvetica", 9.2)
                             for text, _, _ in legend)
    _legend_grid(c, legend, (590.0 - entry_width) / 2.0, 430,
                 columns=1, column_width=entry_width)
    c.save()

    with (DATA / "tilted_posterior.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows



# Archimedean spiral r(t) = SPIRAL_A + SPIRAL_B t, used as a curved k = 1
# manifold whose curvature varies along the arc.
SPIRAL_A, SPIRAL_B = 0.6, 0.35
SPIRAL_T0, SPIRAL_T1 = 0.0, 4.0 * math.pi


def _spiral_point(t):
    r = SPIRAL_A + SPIRAL_B * t
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=-1)


def _spiral_speed(t):
    r = SPIRAL_A + SPIRAL_B * t
    return np.sqrt(SPIRAL_B ** 2 + r ** 2)


def _spiral_curvature(t):
    r = SPIRAL_A + SPIRAL_B * t
    return (r * r + 2.0 * SPIRAL_B ** 2) / (r * r + SPIRAL_B ** 2) ** 1.5


def _spiral_tangent(t):
    r = SPIRAL_A + SPIRAL_B * t
    d = np.array([SPIRAL_B * np.cos(t) - r * np.sin(t),
                  SPIRAL_B * np.sin(t) + r * np.cos(t)])
    return d / np.linalg.norm(d)


def spiral_moments(sigma, t_star, n=300_001):
    """Tangent and normal Rayleigh quotients at a quadratic zero on the spiral.

    The clean density is the squared arc-length distance from the zero, so m = 2.
    """
    t = np.linspace(SPIRAL_T0, SPIRAL_T1, n)
    speed = _spiral_speed(t)
    arc = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(t))])
    arc_star = float(np.interp(t_star, t, arc))
    density = (arc - arc_star) ** 2
    points = _spiral_point(t)
    query = _spiral_point(np.asarray(float(t_star)))
    squared_distance = np.sum((points - query) ** 2, axis=1)
    log_kernel = -squared_distance / (2.0 * sigma * sigma)
    log_kernel -= np.max(log_kernel)
    weights = density * np.exp(log_kernel) * speed
    weights /= np.sum(weights)
    mean = np.sum(weights[:, None] * points, axis=0)
    centered = points - mean
    hessian = (centered.T * weights) @ centered / (sigma * sigma)
    tangent = _spiral_tangent(float(t_star))
    normal = np.asarray([-tangent[1], tangent[0]])
    return float(tangent @ hessian @ tangent), float(normal @ hessian @ normal)


def _spiral_samples(t_star, count=1400, seed=20260821):
    """Sample points along the spiral from the clean density, by inverse CDF."""
    rng = np.random.default_rng(seed)
    t = np.linspace(SPIRAL_T0, SPIRAL_T1, 20_001)
    speed = _spiral_speed(t)
    arc = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(t))])
    arc_star = float(np.interp(t_star, t, arc))
    weight = (arc - arc_star) ** 2 * speed
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (weight[1:] + weight[:-1]) * np.diff(t))])
    cdf /= cdf[-1]
    draws = np.interp(rng.random(count), cdf, t)
    return _spiral_point(draws)


def make_spiral():
    """Quadratic zero on a spiral, whose curvature varies along the arc."""
    stars = [1.0, 2.2, 4.5, 9.0, 12.0]
    sigmas = [0.02, 0.01, 0.005]
    rows = []
    for t_star in stars:
        kappa = float(_spiral_curvature(np.asarray(t_star)))
        for sigma in sigmas:
            tangent, normal = spiral_moments(sigma, t_star)
            rows.append({"t_star": t_star, "curvature": kappa, "sigma": sigma,
                         "tangent_eigenvalue": tangent,
                         "tangent_deficit_over_sigma2": (3.0 - tangent) / sigma ** 2,
                         "normal_over_sigma2": normal / sigma ** 2})
    with (DATA / "spiral.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    path = FIG / "spiral.pdf"
    c = Canvas(str(path), pagesize=(590, 258))
    # One framed box per subfigure.
    box_w, box_h, box_y, gap = 187.0, 175.0, 26.0, 6.0
    boxes = [8.0 + i * (box_w + gap) for i in range(3)]
    for bxx in boxes:
        _panel_box(c, bxx, box_y, box_w, box_h)
    y0, h = box_y + 42.0, 109.0
    ax, bx, cx = (bxx + 66.0 for bxx in boxes)
    aw, pw = h, h                   # square panels, so the spiral is undistorted

    # (a) sampled point cloud
    span = (-5.6, 5.6)
    _axes(c, ax, y0, aw, aw, span, span, "x1", "x2",
          xticks=[-4, 0, 4], yticks=[-4, 0, 4])
    guide = _spiral_point(np.linspace(SPIRAL_T0, SPIRAL_T1, 2000))
    gx = _map(guide[:, 0], *span, ax, ax + aw)
    gy = _map(guide[:, 1], *span, y0, y0 + aw)
    c.saveState()
    clip = c.beginPath(); clip.rect(ax, y0, aw, aw); c.clipPath(clip, stroke=0, fill=0)
    # faint guide for the manifold, where the samples have thinned away
    c.setStrokeColor(LIGHT)
    c.setLineWidth(0.9)
    path = c.beginPath()
    path.moveTo(float(gx[0]), float(gy[0]))
    for xx, yy in zip(gx[1:], gy[1:]):
        path.lineTo(float(xx), float(yy))
    c.drawPath(path, stroke=1, fill=0)
    c.restoreState()

    pts = _spiral_samples(9.0, count=1400)
    xs = _map(pts[:, 0], *span, ax, ax + aw)
    ys = _map(pts[:, 1], *span, y0, y0 + aw)
    c.saveState()
    clip = c.beginPath(); clip.rect(ax, y0, aw, aw); c.clipPath(clip, stroke=0, fill=0)
    c.setFillColor(BLUE)
    c.setFillAlpha(0.55)
    for xx, yy in zip(xs, ys):
        c.circle(float(xx), float(yy), 0.9, stroke=0, fill=1)
    c.restoreState()
    for t_star, marker, shown in ((9.0, "square", True), (2.2, "circle", False),
                                  (12.0, "triangle", False)):
        q = _spiral_point(np.asarray(t_star))
        _marker(c, float(_map(q[0], *span, ax, ax + aw)),
                float(_map(q[1], *span, y0, y0 + aw)), marker, ORANGE,
                size=3.4 if shown else 2.6, filled=shown)

    # (b) convergence at three curvatures
    curves = ((1.0, PURPLE, None, "circle"), (4.5, ORANGE, [5, 2], "triangle"),
              (12.0, GRAY, [1.2, 2], "diamond"))
    _axes(c, bx, y0, pw, h, (0.004, 0.024), (2.9985, 3.0004),
          "noise scale \u03c3", "tangent eigenvalue",
          xticks=[0.005, 0.01, 0.02], yticks=[2.999, 2.9995, 3.0],
          ytick_labels=["2.999", "2.9995", "3"])
    _curve(c, np.asarray([0.004, 0.024]), np.asarray([3.0, 3.0]),
           bx, y0, pw, h, (0.004, 0.024), (2.9985, 3.0004), BLACK,
           width=1.0, dash=[3, 2])
    for (t_star, color, dash, marker) in curves:
        sel = sorted([r for r in rows if r["t_star"] == t_star],
                     key=lambda r: r["sigma"])
        _curve(c, np.asarray([r["sigma"] for r in sel]),
               np.asarray([r["tangent_eigenvalue"] for r in sel]),
               bx, y0, pw, h, (0.004, 0.024), (2.9985, 3.0004), color,
               dash=dash, marker=marker, marker_stride=1)

    # (c) remainder against curvature squared
    kappas = np.asarray([float(_spiral_curvature(np.asarray(v))) for v in stars])
    pick = lambda v, key: [r for r in rows
                           if r["t_star"] == v and r["sigma"] == 0.005][0][key]
    deficit = np.asarray([pick(v, "tangent_deficit_over_sigma2") for v in stars])
    normal = np.asarray([pick(v, "normal_over_sigma2") for v in stars])
    _axes(c, cx, y0, pw, h, (0.0, 1.35), (0.0, 3.4),
          "curvature squared", "remainder / \u03c3\u00b2",
          xticks=[0, 0.5, 1.0], yticks=[0, 1, 2, 3])
    for values, color, marker, slope in ((deficit, BLUE, "circle", 2.5),
                                         (normal, TEAL, "square", 1.5)):
        _curve(c, np.asarray([0.0, 1.35]), np.asarray([0.0, slope * 1.35]),
               cx, y0, pw, h, (0.0, 1.35), (0.0, 3.4), BLACK, width=0.9, dash=[3, 2])
        _curve(c, kappas ** 2, values, cx, y0, pw, h, (0.0, 1.35), (0.0, 3.4),
               color, width=0.0, marker=marker, marker_stride=1)

    title_y = box_y + box_h + 6.0
    _panel_title(c, boxes[0] + 6, title_y, "(a) Sampled data and zero")
    _panel_title(c, boxes[1] + 6, title_y, "(b) Limit is curvature free")
    _panel_title(c, boxes[2] + 6, title_y, "(c) Remainder vs \u03ba\u00b2")
    legend = [(f"\u03ba = {float(_spiral_curvature(np.asarray(t))):.2f}", col, dash, mk)
              for t, col, dash, mk in curves]
    legend += [("tangent deficit, slope 2.5", BLUE, None, "circle"),
               ("normal eigenvalue, slope 1.5", TEAL, None, "square")]
    _legend_grid(c, legend, 14, 246, columns=3, column_width=190)
    c.save()
    return rows


def make_circle_spectrum():
    sigmas = np.geomspace(0.015, 0.35, 34)
    orders = [0, 1, 2, 4]
    colors = [GRAY, BLUE, ORANGE, PURPLE]
    rows = []
    curves = {}
    for r in orders:
        tangent, normal = [], []
        for sigma in sigmas:
            _, h = circle_moments(float(sigma), r)
            normal.append(h[0, 0])
            tangent.append(h[1, 1])
            rows.append((sigma, r, h[1, 1], h[0, 0]))
        curves[r] = (np.array(tangent), np.array(normal))
    with (DATA / "circle_spectrum.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma", "r", "tangent_eigenvalue", "normal_eigenvalue"])
        w.writerows(rows)

    path = FIG / "circle_spectrum.pdf"
    c = Canvas(str(path), pagesize=(500, 265))
    x0, y0, w, h = 58, 43, 420, 170
    _auto_box(c, x0, y0, w, h, yticks=[0, 1, 3, 5, 7, 9], top_pad=10.0)
    _axes(c, x0, y0, w, h, (sigmas.min(), sigmas.max()), (0, 9.5),
          "noise scale \u03c3", "normalized covariance eigenvalue", xlog=True,
          xticks=[0.02, 0.05, 0.1, 0.2], yticks=[0, 1, 3, 5, 7, 9])
    markers = ["circle", "square", "triangle", "diamond"]
    for r, color, marker in zip(orders, colors, markers):
        tangent, normal = curves[r]
        _curve(c, sigmas, tangent, x0, y0, w, h,
               (sigmas.min(), sigmas.max()), (0, 9.5), color, xlog=True,
               marker=marker, marker_stride=8)
        if r in (1, 4):
            _curve(c, sigmas, normal, x0, y0, w, h,
                   (sigmas.min(), sigmas.max()), (0, 9.5), color,
                   xlog=True, width=1.0, dash=[3, 2])
    _legend_grid(c, [("r=0 (limit 1)", GRAY, None, "circle"),
                     ("r=1 (limit 3)", BLUE, None, "square"),
                     ("r=2 (limit 5)", ORANGE, None, "triangle"),
                     ("r=4 (limit 9)", PURPLE, None, "diamond"),
                     ("tangent", BLACK, None),
                     ("normal (r=1,4)", BLACK, [3, 2])],
                 58, 247, columns=3, column_width=140)
    c.save()


def make_crossover():
    sigma_values = [0.025, 0.04, 0.065, 0.1]
    eta = np.geomspace(1e-3, 1e3, 61)
    rows = []
    curves = []
    for sigma in sigma_values:
        vals = []
        for ratio in eta:
            floor = ratio * sigma**2
            _, h = circle_moments(sigma, 1, floor=floor, n=120_001)
            vals.append(h[1, 1])
            rows.append((sigma, ratio, floor, h[1, 1]))
        curves.append(np.array(vals))
    theory = (eta + 1.5) / (eta + 0.5)
    with (DATA / "crossover.csv").open("w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["sigma", "delta_over_sigma2", "delta", "tangent_eigenvalue"])
        wri.writerows(rows)

    path = FIG / "crossover.pdf"
    c = Canvas(str(path), pagesize=(500, 265))
    x0, y0, w, h = 58, 43, 420, 168
    _auto_box(c, x0, y0, w, h, yticks=[1, 1.5, 2, 2.5, 3], top_pad=10.0)
    _axes(c, x0, y0, w, h, (eta.min(), eta.max()), (0.9, 3.1),
          "density floor \u03b4 / \u03c3\u00b2", "tangent eigenvalue",
          xlog=True, xticks=[1e-3, 1e-1, 1e1, 1e3],
          yticks=[1, 1.5, 2, 2.5, 3])
    _curve(c, eta, theory, x0, y0, w, h, (eta.min(), eta.max()),
           (0.9, 3.1), BLACK, xlog=True, width=2.1)
    for (sigma, vals), (color, dash, marker) in zip(
            zip(sigma_values, curves), SIGMA_STYLES):
        _curve(c, eta, vals, x0, y0, w, h, (eta.min(), eta.max()),
               (0.9, 3.1), color, xlog=True, width=1.25, dash=dash,
               marker=marker, marker_stride=12)
    entries = [("joint-limit curve", BLACK, None)]
    entries += [(f"\u03c3 = {s:g}", color, dash, marker)
                for s, (color, dash, marker) in zip(sigma_values, SIGMA_STYLES)]
    _legend_grid(c, entries, 58, 247, columns=3, column_width=140)
    c.save()


def make_spatial_boundary_layer():
    sigma_values = [0.02, 0.035, 0.06, 0.1]
    offsets = np.linspace(0.0, 4.0, 81)
    rows = []
    curves = []
    for sigma in sigma_values:
        vals = []
        for a in offsets:
            tangent, normal = circle_moments_at(
                sigma, 1, query_angle=float(sigma * a), n=120_001
            )
            vals.append(tangent)
            rows.append((sigma, a, tangent, normal))
        curves.append(np.asarray(vals))
    theory = (offsets**4 + 3.0) / (offsets**2 + 1.0)**2
    with (DATA / "spatial_boundary_layer.csv").open("w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["sigma", "angle_over_sigma",
                      "tangent_rayleigh_quotient",
                      "normal_rayleigh_quotient"])
        wri.writerows(rows)

    path = FIG / "spatial_boundary_layer.pdf"
    c = Canvas(str(path), pagesize=(500, 285))
    x0, y0, w, h = 58, 43, 420, 160
    _auto_box(c, x0, y0, w, h, yticks=[0.75, 1, 1.5, 2, 2.5, 3], top_pad=10.0)
    _axes(c, x0, y0, w, h, (0.0, 4.0), (0.65, 3.1),
          "distance from zero / \u03c3",
          "tangent Rayleigh quotient", xticks=[0, 1, math.sqrt(3), 3, 4],
          yticks=[0.75, 1, 1.5, 2, 2.5, 3],
          xtick_labels=["0", "1", "\u221a3", "3", "4"])
    _curve(c, offsets, theory, x0, y0, w, h, (0.0, 4.0), (0.65, 3.1),
           BLACK, width=2.1)
    for (sigma, vals), (color, dash, marker) in zip(
            zip(sigma_values, curves), SIGMA_STYLES):
        _curve(c, offsets, vals, x0, y0, w, h, (0.0, 4.0),
               (0.65, 3.1), color, width=1.25, dash=dash,
               marker=marker, marker_stride=16)
    entries = [("boundary-layer limit", BLACK, None)]
    entries += [(f"\u03c3 = {s:g}", color, dash, marker)
                for s, (color, dash, marker) in zip(sigma_values, SIGMA_STYLES)]
    _legend_grid(c, entries, 58, 270, columns=2, column_width=215,
                 row_height=17)
    c.save()


def make_landing_error():
    sigma_values = [0.02, 0.035, 0.06, 0.1]
    offsets = np.linspace(0.0, 4.0, 81)
    tangent_curves, normal_curves, rows = [], [], []
    for sigma in sigma_values:
        tan_vals, normal_vals = [], []
        for a in offsets:
            tangent, normal = circle_denoiser_error_at(
                sigma, 1, query_angle=float(sigma * a), n=120_001
            )
            tan_vals.append(tangent / sigma)
            normal_vals.append(abs(normal) / (sigma * sigma))
            rows.append((sigma, a, tangent, normal, tangent / sigma,
                         abs(normal) / (sigma * sigma)))
        tangent_curves.append(np.asarray(tan_vals))
        normal_curves.append(np.asarray(normal_vals))
    tangent_theory = 2.0 * offsets / (offsets**2 + 1.0)
    with (DATA / "landing_error.csv").open("w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["sigma", "angle_over_sigma", "raw_tangent_error",
                      "raw_normal_error", "tangent_error_over_sigma",
                      "normal_only_error_over_sigma2"])
        wri.writerows(rows)

    path = FIG / "landing_error.pdf"
    c = Canvas(str(path), pagesize=(580, 285))
    x1, x2, y0, w, h = 58, 350, 45, 200, 170
    _auto_box(c, x1, y0, w, h, yticks=[0, .25, .5, .75, 1])
    _axes(c, x1, y0, w, h, (0.0, 4.0), (0.0, 1.1),
          "offset / \u03c3", "tangent error / \u03c3",
          xticks=[0, 1, 2, 3, 4], yticks=[0, .25, .5, .75, 1])
    _auto_box(c, x2, y0, w, h, yticks=[0, .5, 1, 1.5])
    _axes(c, x2, y0, w, h, (0.0, 4.0), (0.0, 1.7),
          "offset / \u03c3", "normal residual / \u03c3\u00b2",
          xticks=[0, 1, 2, 3, 4], yticks=[0, .5, 1, 1.5])
    _curve(c, offsets, tangent_theory, x1, y0, w, h, (0.0, 4.0),
           (0.0, 1.1), BLACK, width=2.0)
    for (tan, normal), (color, dash, marker) in zip(
            zip(tangent_curves, normal_curves), SIGMA_STYLES):
        _curve(c, offsets, tan, x1, y0, w, h, (0.0, 4.0), (0.0, 1.1),
               color, width=1.15, dash=dash, marker=marker,
               marker_stride=20)
        _curve(c, offsets, normal, x2, y0, w, h, (0.0, 4.0), (0.0, 1.7),
               color, width=1.15, dash=dash, marker=marker,
               marker_stride=20, marker_offset=6)
    _panel_title(c, x1, 229, "(a) Tangent drift")
    _panel_title(c, x2, 229, "(b) Normal-only residual")
    entries = [("limit curve", BLACK, None)]
    entries += [(f"\u03c3 = {s:g}", color, dash, marker)
                for s, (color, dash, marker) in zip(sigma_values, SIGMA_STYLES)]
    _legend_grid(c, entries, 58, 271, columns=3, column_width=170)
    c.save()


def _sample_quadratic_zero_circle(rng, n):
    """Exact rejection sampler for rho(theta)=(1-cos(theta))/(2*pi)."""
    chunks, have = [], 0
    while have < n:
        proposal = rng.uniform(-math.pi, math.pi, size=max(4096, 2 * (n - have)))
        keep = rng.random(proposal.size) <= (1.0 - np.cos(proposal)) / 2.0
        accepted = proposal[keep]
        chunks.append(accepted)
        have += accepted.size
    return np.concatenate(chunks)[:n]


def circle_blocks_at_offset(sigma: float, order_r: int, a: float, b: float,
                            n: int = 400_001):
    """Posterior blocks at y = (1 + sigma*b)*(cos(sigma*a), sin(sigma*a)).

    Here a and b are the rescaled tangential and normal coordinates.
    """
    theta = np.linspace(-math.pi, math.pi, n, endpoint=False)
    rho = (1.0 - np.cos(theta))**order_r
    phi = sigma * a
    radius = 1.0 + sigma * b
    squared_distance = radius**2 + 1.0 - 2.0 * radius * np.cos(theta - phi)
    log_kernel = -squared_distance / (2.0 * sigma * sigma)
    log_kernel -= np.max(log_kernel)
    weights = rho * np.exp(log_kernel)
    weights /= np.sum(weights)
    points = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    mean = np.sum(weights[:, None] * points, axis=0)
    centered = points - mean
    hessian = (centered.T * weights) @ centered / (sigma * sigma)
    tangent = np.array([-math.sin(phi), math.cos(phi)])
    normal = np.array([math.cos(phi), math.sin(phi)])
    query = radius * normal
    footpoint = query / np.linalg.norm(query)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    top = eigenvectors[:, -1]
    projector = np.outer(top, top)
    landing = query + (np.eye(2) - projector) @ (mean - query)
    return {
        "tangent_block": float(tangent @ hessian @ tangent),
        "cross_block_over_sigma": float(normal @ hessian @ tangent) / sigma,
        "normal_block_over_sigma2": float(normal @ hessian @ normal) / sigma**2,
        "second_eigenvalue_over_sigma2": float(eigenvalues[0]) / sigma**2,
        "landing_error_over_sigma2": float(np.linalg.norm(landing - footpoint)) / sigma**2,
    }


def make_normal_offset_audit():
    """Numerical check of Theorem 3.7 at nonzero normal offset."""
    rows = []
    for a, b in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.3), (-0.8, 1.3), (2.0, -1.1)):
        for sigma in (0.02, 0.01, 0.005):
            row = {"a": a, "b": b, "sigma": sigma}
            row.update(circle_blocks_at_offset(sigma, 1, a, b))
            rows.append(row)
    path = DATA / "normal_offset_audit.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def make_finite_sample_kde():
    """Separate finite-sigma population bias from finite-sample error."""
    sample_sizes = [20_000, 80_000, 320_000]
    sigmas = [0.035, 0.05, 0.075, 0.11, 0.16]
    seeds = list(range(48))
    population = {}
    for sigma in sigmas:
        _, h_sigma = circle_moments(sigma, 1)
        population[sigma] = (h_sigma, float(np.linalg.eigvalsh(h_sigma)[-1]))
    rows = []
    grouped = {(n, s): [] for n in sample_sizes for s in sigmas}
    for seed in seeds:
        actual_seed = 20260819 + seed
        rng = np.random.default_rng(actual_seed)
        theta_all = _sample_quadratic_zero_circle(rng, max(sample_sizes))
        for n in sample_sizes:
            theta = theta_all[:n]
            points = np.stack([np.cos(theta), np.sin(theta)], axis=1)
            one_minus_cos = 1.0 - np.cos(theta)
            for sigma in sigmas:
                logw = -one_minus_cos / (sigma * sigma)
                logw -= np.max(logw)
                w = np.exp(logw)
                sumw = np.sum(w)
                wn = w / sumw
                ess = float(sumw * sumw / np.sum(w * w))
                mean = np.sum(wn[:, None] * points, axis=0)
                centered = points - mean
                h = (centered.T * wn) @ centered / (sigma * sigma)
                evals, evecs = np.linalg.eigh(h)
                v = evecs[:, -1]
                projector_error = math.sqrt(max(0.0, 1.0 - v[1] * v[1]))
                h_sigma, population_top = population[sigma]
                operator_sampling_error = float(np.linalg.norm(h - h_sigma, ord=2))
                eigenvalue_sampling_error = abs(float(evals[-1]) - population_top)
                asymptotic_bias = abs(population_top - 3.0)
                rounded_order_success = int(round(float(evals[-1]) - 1.0) == 2)
                raw_geometry_error = float(np.linalg.norm(
                    h - np.diag([0.0, 1.0]), ord=2
                ))
                record = (actual_seed, n, sigma, ess, projector_error,
                          operator_sampling_error, eigenvalue_sampling_error,
                          asymptotic_bias, rounded_order_success,
                          raw_geometry_error, float(evals[-1]), population_top)
                rows.append(record)
                grouped[(n, sigma)].append(record)
    with (DATA / "finite_sample_kde.csv").open("w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["seed", "sample_size", "sigma", "ess",
                      "calibrated_projector_error", "operator_sampling_error",
                      "eigenvalue_sampling_error", "finite_sigma_bias_to_limit",
                      "rounded_order_success", "raw_geometry_error",
                      "top_eigenvalue", "population_top_eigenvalue"])
        wri.writerows(rows)

    def wilson(successes, total, z=1.959963984540054):
        probability = successes / total
        denominator = 1.0 + z * z / total
        center = (probability + z * z / (2.0 * total)) / denominator
        half = z * math.sqrt(
            probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
        ) / denominator
        return probability, max(0.0, center - half), min(1.0, center + half)

    summaries = []
    for (n, sigma), values in grouped.items():
        arr = np.asarray([[r[3], r[4], r[5], r[6], r[7], r[8], r[9]]
                          for r in values])
        success, success_low, success_high = wilson(int(np.sum(arr[:, 5])), len(values))
        summaries.append({
            "n": n, "sigma": sigma,
            "ess": float(np.median(arr[:, 0])),
            "proj_med": float(np.median(arr[:, 1])),
            "proj_q1": float(np.quantile(arr[:, 1], .25)),
            "proj_q3": float(np.quantile(arr[:, 1], .75)),
            "operator_med": float(np.median(arr[:, 2])),
            "operator_q1": float(np.quantile(arr[:, 2], .25)),
            "operator_q3": float(np.quantile(arr[:, 2], .75)),
            "eigenvalue_med": float(np.median(arr[:, 3])),
            "finite_sigma_bias": float(arr[0, 4]),
            "order_success": success,
            "order_success_wilson_low": success_low,
            "order_success_wilson_high": success_high,
            "raw_geometry_med": float(np.median(arr[:, 6])),
            "replications": len(values),
        })
    with (DATA / "finite_sample_kde_summary.csv").open("w", newline="") as f:
        fields = list(summaries[0])
        wri = csv.DictWriter(f, fieldnames=fields)
        wri.writeheader()
        wri.writerows(summaries)

    path = FIG / "finite_sample_kde.pdf"
    c = Canvas(str(path), pagesize=(590, 315))
    x1, x2, y0, w, h = 58, 365, 45, 205, 175
    xlim, ylim1, ylim2 = (0.5, 4000), (1e-4, 4), (0.0, 1.05)
    _auto_box(c, x1, y0, w, h, yticks=[1e-4, 1e-2, 1])
    _axes(c, x1, y0, w, h, xlim, ylim1, "effective sample size",
          "finite-sample error", xlog=True, ylog=True,
          xticks=[1, 10, 100, 1000], yticks=[1e-4, 1e-2, 1],
          ytick_labels=["1e-4", "1e-2", "1"])
    _auto_box(c, x2, y0, w, h, yticks=[0, .25, .5, .75, 1])
    _axes(c, x2, y0, w, h, xlim, ylim2, "effective sample size",
          "rounded-order recovery", xlog=True, ylog=False,
          xticks=[1, 10, 100, 1000], yticks=[0, .25, .5, .75, 1])
    colors = {0.035: PURPLE, 0.05: BLUE, 0.075: TEAL,
              0.11: ORANGE, 0.16: GRAY}
    markers = {20_000: "circle", 80_000: "triangle", 320_000: "diamond"}

    for s in summaries:
        color = colors[s["sigma"]]
        xx1 = float(_map(s["ess"], *xlim, x1, x1 + w, log=True))
        yy1 = float(_map(max(s["operator_med"], ylim1[0]), *ylim1,
                         y0, y0 + h, log=True))
        lo1 = float(_map(max(s["operator_q1"], ylim1[0]), *ylim1,
                         y0, y0 + h, log=True))
        hi1 = float(_map(max(s["operator_q3"], ylim1[0]), *ylim1,
                         y0, y0 + h, log=True))
        _error_bar(c, xx1, lo1, hi1, color)
        _marker(c, xx1, yy1, markers[s["n"]], color, size=2.7, filled=True)
        projector_y = float(_map(max(s["proj_med"], ylim1[0]), *ylim1,
                                 y0, y0 + h, log=True))
        _marker(c, xx1, projector_y, markers[s["n"]], color,
                size=2.7, filled=False)
        xx2 = float(_map(s["ess"], *xlim, x2, x2 + w, log=True))
        yy2 = float(_map(s["order_success"], *ylim2, y0, y0 + h))
        lo2 = float(_map(s["order_success_wilson_low"], *ylim2, y0, y0 + h))
        hi2 = float(_map(s["order_success_wilson_high"], *ylim2, y0, y0 + h))
        _error_bar(c, xx2, lo2, hi2, color)
        _marker(c, xx2, yy2, markers[s["n"]], color, size=2.7, filled=True)

    _panel_title(c, x1, 235, "(a) Sampling and range errors")
    _panel_title(c, x2, 235, "(b) Order recovery (m=2)")
    c.setFillColor(BLACK)
    c.setFont("Helvetica-Bold", 9.2)
    c.drawString(58, 298, "\u03c3")
    noise_entries = [(f"{sigma:g}", colors[sigma], None, "circle")
                     for sigma in sigmas]
    _legend_grid(c, noise_entries, 102, 300, columns=5, column_width=80)
    c.setFont("Helvetica-Bold", 9.2)
    c.drawString(58, 282, "sample size")
    _legend_grid(c, [("20k", BLACK, None, "circle"),
                     ("80k", BLACK, None, "triangle"),
                     ("320k", BLACK, None, "diamond")],
                 126, 284, columns=3, column_width=86)
    c.setFont("Helvetica-Bold", 9.2)
    c.drawString(361, 282, "measure")
    _legend_grid(c, [("operator / order", BLACK, None, "circle", True),
                     ("range", BLACK, None, "circle", False)],
                 408, 284, columns=2, column_width=92, font_size=8.4)
    c.setFillColor(GRAY)
    c.setFont("Helvetica", 8.2)
    c.drawString(58, 265, "bars: IQR in (a), Wilson 95% interval in (b); 48 replications")
    c.save()


def make_conditioning():
    m = 8
    cmax = m + 1.0
    gammas = np.linspace(0.0, 2.0, 1001)
    vals = {
        "calibrated inner + normal-only landing": np.abs(1.0 - gammas),
        "raw denoiser + raw inner Jacobian": np.maximum(
            np.abs(1.0 - gammas), cmax * np.abs(1.0 - cmax * gammas)
        ),
        "raw denoiser + calibrated inner step": cmax * np.abs(1.0 - gammas),
        "raw additive preconditioner": np.maximum(
            np.abs(1.0 - gammas), np.abs(1.0 - cmax * gammas)
        ),
    }
    with (DATA / "conditioning.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step_size", *vals.keys()])
        for i, gamma in enumerate(gammas):
            w.writerow([gamma, *(v[i] for v in vals.values())])

    orders = np.arange(0, 21, 2)
    widths = {
        "calibrated inner + normal-only landing": np.full_like(orders, 2.0, dtype=float),
        "raw denoiser + raw inner Jacobian": 2.0 / (orders + 1.0) ** 2,
        "raw denoiser + calibrated inner step": 2.0 / (orders + 1.0),
        "raw additive preconditioner": 2.0 / (orders + 1.0),
    }
    with (DATA / "conditioning_stability_width.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vanishing_order", *widths.keys()])
        for i, order in enumerate(orders):
            w.writerow([order, *(v[i] for v in widths.values())])

    path = FIG / "conditioning.pdf"
    c = Canvas(str(path), pagesize=(590, 300))
    x1, x2, y0, w, h = 58, 365, 45, 205, 175
    ylim = (0.03, 200.0)
    _auto_box(c, x1, y0, w, h, yticks=[.05, .1, .5, 1, 5, 20, 100])
    _axes(c, x1, y0, w, h, (0.0, 2.0), ylim, "step size \u03b3",
          "spectral radius", ylog=True, xticks=[0, .5, 1, 1.5, 2],
          yticks=[.05, .1, .5, 1, 5, 20, 100])
    specifications = [
        ("calibrated inner + normal-only landing", BLUE, None, "circle"),
        ("raw denoiser + raw inner Jacobian", ORANGE, [6, 2], "square"),
        ("raw denoiser + calibrated inner step", PURPLE, [1.2, 2], "triangle"),
        ("raw additive preconditioner", TEAL, [7, 2, 1.2, 2], "diamond"),
    ]
    for name, color, dash, marker in specifications:
        _curve(c, gammas, vals[name], x1, y0, w, h,
               (0.0, 2.0), ylim, color, ylog=True, width=1.45, dash=dash,
               marker=marker, marker_stride=250)
    _curve(c, np.asarray([0.0, 2.0]), np.asarray([1.0, 1.0]), x1, y0, w, h,
           (0.0, 2.0), ylim, BLACK, ylog=True, width=1.0, dash=[2, 2])

    _auto_box(c, x2, y0, w, h, yticks=[1e-3, 1e-2, .1, 1])
    _axes(c, x2, y0, w, h, (0.0, 20.0), (1e-3, 3.0), "vanishing order m",
          "stable step-size interval width", ylog=True,
          xticks=[0, 4, 8, 12, 16, 20], yticks=[1e-3, 1e-2, .1, 1],
          ytick_labels=["0.001", "0.01", "0.1", "1"])
    for index, (name, color, dash, marker) in enumerate(specifications):
        _curve(c, orders, widths[name], x2, y0, w, h, (0.0, 20.0),
               (1e-3, 3.0), color, ylog=True, width=1.45, dash=dash,
               marker=marker, marker_stride=2, marker_offset=index % 2)
    legend_entries = [("calibrated inner + landing", BLUE, None),
                      ("raw outer + raw inner", ORANGE, [6, 2]),
                      ("raw outer + calibrated inner", PURPLE, [1.2, 2]),
                      ("raw additive metric", TEAL, [7, 2, 1.2, 2]),
                      ("stability threshold", BLACK, [2, 2])]
    _legend_grid(c, legend_entries, 58, 284, columns=3, column_width=170)
    _panel_title(c, x1, 236, "(a) Spectral radius (m=8)")
    _panel_title(c, x2, 236, "(b) Exact stability interval")
    c.save()


if __name__ == "__main__":
    make_tilted_posterior()
    make_circle_spectrum()
    make_spiral()
    make_crossover()
    make_spatial_boundary_layer()
    make_landing_error()
    make_normal_offset_audit()
    make_finite_sample_kde()
    make_conditioning()
    print(f"Wrote figures to {FIG}")
    print(f"Wrote numerical results to {DATA}")
