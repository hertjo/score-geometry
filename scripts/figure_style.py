"""Shared look for the paper's figures, drawn with Splinx.

Units are pt at the size the figure is shown in the paper (text width 468 pt), origin top left, y down.
"""
from __future__ import annotations

import math

import numpy as np
from splinx import draw as sx
import splinx.tex as _tex

_tex.set_preamble(r"""\documentclass[border=0pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}\usepackage{amsmath,amssymb,bm,graphicx}
\usepackage{xcolor}
\begin{document}
""")

# palette
INK = "#1e1b3a"
NAVY = "#0500b5"
VIOLET = "#6b67ee"
LILAC = "#c9c6ff"
CYAN = "#3fc4f0"
MINT = "#2fbf8f"
GRAY = "#8b89a6"       # a muted tint of the ink, for reference curves
GRID = "#e6e4fa"
MINOR = "#f3f2fd"
SPINE = "#b4b1e6"
TICK = "#4a4766"
PANEL = "#f7f6ff"

TICK_PT = 7.0
LABEL_PT = 8.0
TITLE_PT = 8.6
LEGEND_PT = 7.4


def fmt_num(v):
    """Tick label in LaTeX, without trailing zeros."""
    if v == 0:
        return "$0$"
    s = f"{v:g}"
    if "e" in s:
        mant, exp = s.split("e")
        exp = int(exp)
        if mant in ("1", "1.0"):
            return f"$10^{{{exp}}}$"
        return f"${mant}\\cdot 10^{{{exp}}}$"
    return f"${s}$"


def pow10(e):
    return f"$10^{{{e}}}$"


class Ax:
    """A data window drawn in the box (x0, y0, w, h); y up inside, optional log scales."""

    def __init__(self, x0, y0, w, h, xlim, ylim, xlog=False, ylog=False):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.xlim, self.ylim, self.xlog, self.ylog = xlim, ylim, xlog, ylog

    def _t(self, v, log):
        v = np.asarray(v, dtype=float)
        return np.log10(v) if log else v

    def X(self, v):
        a, b = self._t(self.xlim, self.xlog)
        return self.x0 + (self._t(v, self.xlog) - a) / (b - a) * self.w

    def Y(self, v):
        a, b = self._t(self.ylim, self.ylog)
        return self.y0 + self.h - (self._t(v, self.ylog) - a) / (b - a) * self.h

    def clampy(self, ys):
        lo, hi = self.ylim
        return np.clip(np.asarray(ys, dtype=float), lo, hi)

    # ---- frame ----
    def frame(self, xticks, yticks, xlabel=None, ylabel=None, grid=True, ylabel_gap=None, xminor=(), yminor=()):
        x0, y0, w, h = self.x0, self.y0, self.w, self.h
        sx.rect(x0, y0, w, h, fill="#ffffff", stroke=None)
        if grid:
            for v in list(xminor):
                X = float(self.X(v))
                sx.line(X, y0, X, y0 + h, stroke=MINOR, width=0.3)
            for v in list(yminor):
                Y = float(self.Y(v))
                sx.line(x0, Y, x0 + w, Y, stroke=MINOR, width=0.3)
            for v, _ in xticks:
                X = float(self.X(v))
                sx.line(X, y0, X, y0 + h, stroke=GRID, width=0.6)
            for v, _ in yticks:
                Y = float(self.Y(v))
                sx.line(x0, Y, x0 + w, Y, stroke=GRID, width=0.6)
        sx.rect(x0, y0, w, h, fill=None, stroke=SPINE, width=0.7)
        for v, lab in xticks:
            X = float(self.X(v))
            sx.line(X, y0 + h, X, y0 + h + 2.2, stroke=SPINE, width=0.6)
            if lab:
                sx.tex(lab, X, y0 + h + 4.0, TICK_PT, TICK, anchor="center", valign="top")
        maxw = 0.0
        for v, lab in yticks:
            Y = float(self.Y(v))
            sx.line(x0 - 2.2, Y, x0, Y, stroke=SPINE, width=0.6)
            if lab:
                maxw = max(maxw, sx.texw(lab, TICK_PT))
                sx.tex(lab, x0 - 4.0, Y, TICK_PT, TICK, anchor="right", valign="middle")
        if xlabel:
            sx.tex(xlabel, x0 + w / 2, y0 + h + 15.5, LABEL_PT, INK, anchor="center", valign="top")
        if ylabel:
            gap = ylabel_gap if ylabel_gap is not None else maxw + 13.0
            ylab(ylabel, x0 - gap, y0 + h / 2)

    # ---- marks ----
    def line(self, xs, ys, col, w=1.3, dash=None, clamp=True, opacity=None):
        xs = np.asarray(xs, float)
        ys = self.clampy(ys) if clamp else np.asarray(ys, float)
        keep = (xs >= self.xlim[0] - 1e-12) & (xs <= self.xlim[1] + 1e-12)
        P = np.c_[self.X(xs[keep]), self.Y(ys[keep])]
        kw = dict(stroke=col, width=w, fill=None)
        if dash:
            kw["dash"] = dash
        if opacity is not None:
            kw["opacity"] = opacity
        sx.poly(P, **kw)

    def band(self, xs, lo, hi, col, opacity=22):
        xs = np.asarray(xs, float)
        lo, hi = self.clampy(lo), self.clampy(hi)
        P = np.r_[np.c_[self.X(xs), self.Y(hi)], np.c_[self.X(xs[::-1]), self.Y(lo[::-1])]]
        sx.poly(P, closed=True, fill=col, stroke=None, opacity=opacity)

    def marks(self, xs, ys, kind, col, size=2.4, filled=True, clamp=True):
        ys = self.clampy(ys) if clamp else ys
        for x, y in zip(np.atleast_1d(xs), np.atleast_1d(ys)):
            marker(float(self.X(x)), float(self.Y(y)), kind, col, size, filled)

    def errbar(self, x, lo, hi, col, cap=1.8, w=0.8):
        X = float(self.X(x))
        a, b = float(self.Y(self.clampy(lo))), float(self.Y(self.clampy(hi)))
        sx.line(X, a, X, b, stroke=col, width=w)
        sx.line(X - cap, a, X + cap, a, stroke=col, width=w)
        sx.line(X - cap, b, X + cap, b, stroke=col, width=w)

    def hline(self, y, col, w=0.8, dash=None):
        Y = float(self.Y(y))
        kw = dict(stroke=col, width=w)
        if dash:
            kw["dash"] = dash
        sx.line(self.x0, Y, self.x0 + self.w, Y, **kw)


def ylab(text, x, y, size=LABEL_PT, color=INK):
    """A vertical axis label centred at (x, y), reading upwards."""
    sx.rotate(lambda: sx.tex(text, x, y, size, color, anchor="center", valign="middle"), -90, about=(x, y))()


def marker(x, y, kind, col, size=2.4, filled=True, edge=0.8):
    fill = col if filled else "#ffffff"
    look = dict(fill=fill, stroke=col, width=edge)
    if kind == "circle":
        sx.circle(x, y, size, **look)
    elif kind == "square":
        s = size * 0.88
        sx.rect(x - s, y - s, 2 * s, 2 * s, **look)
    elif kind == "triangle":
        s = size * 1.2
        sx.poly(np.array([[x, y - s], [x + s * 0.92, y + s * 0.6], [x - s * 0.92, y + s * 0.6]]), closed=True, **look)
    elif kind == "triangle_down":
        s = size * 1.2
        sx.poly(np.array([[x, y + s], [x + s * 0.92, y - s * 0.6], [x - s * 0.92, y - s * 0.6]]), closed=True, **look)
    elif kind == "diamond":
        s = size * 1.25
        sx.poly(np.array([[x, y - s], [x + s * 0.8, y], [x, y + s], [x - s * 0.8, y]]), closed=True, **look)
    elif kind is None:
        return
    else:
        raise ValueError(kind)


def panel_title(label, title, x, y):
    """'(a) Title' with a bold violet label, baseline at y."""
    w = sx.tex(r"\textbf{(" + label + ")}", x, y, TITLE_PT, VIOLET)
    sx.tex(title, x + w + 4.0, y, TITLE_PT, INK)


def legend_entry_width(label, sample=16.0):
    return sample + 4.0 + sx.texw(label, LEGEND_PT)


def legend_entry(x, y, label, col, dash=None, mark=None, filled=True, line=True, sample=16.0, lw=1.3):
    """One legend item with its baseline-centre at y: a line sample, an optional marker, the label."""
    if line:
        kw = dict(stroke=col, width=lw)
        if dash:
            kw["dash"] = dash
        sx.line(x, y, x + sample, y, **kw)
    if mark:
        marker(x + sample / 2, y, mark, col, 2.2, filled)
    sx.tex(label, x + sample + 4.0, y, LEGEND_PT, INK, valign="middle")
    return legend_entry_width(label, sample)


def legend_row(entries, x, y, gap=12.0, center=None):
    """Entries left to right: each (label, col, dash, mark[, filled[, line]]). center=(x0, x1) centres the row."""
    widths = []
    for e in entries:
        widths.append(legend_entry_width(e[0]))
    total = sum(widths) + gap * (len(entries) - 1)
    if center is not None:
        x = (center[0] + center[1]) / 2 - total / 2
    for e, wd in zip(entries, widths):
        label, col, dash, mark = e[:4]
        filled = e[4] if len(e) > 4 else True
        line = e[5] if len(e) > 5 else True
        legend_entry(x, y, label, col, dash, mark, filled, line)
        x += wd + gap
    return total


def legend_box(entries, x, y, row=11.0, pad=5.0, anchor="nw"):
    """A stacked legend on a white plate inside a plot; anchor nw/ne/sw/se at (x, y)."""
    widths = [legend_entry_width(e[0]) for e in entries]
    W = max(widths) + 2 * pad
    H = row * len(entries) + 2 * pad - (row - 8.0)
    if "e" in anchor:
        x = x - W
    if "s" in anchor:
        y = y - H
    sx.rect(x, y, W, H, radius=2.0, fill="#ffffff", stroke=GRID, width=0.6, opacity=92)
    yy = y + pad + 4.0
    for e in entries:
        label, col, dash, mark = e[:4]
        filled = e[4] if len(e) > 4 else True
        line = e[5] if len(e) > 5 else True
        legend_entry(x + pad, yy, label, col, dash, mark, filled, line)
        yy += row
    return W, H


def log_ticks(e0, e1, labels=None):
    out = []
    for e in range(e0, e1 + 1):
        out.append((10.0 ** e, pow10(e) if labels is None else labels(e)))
    return out


def log_minor(e0, e1):
    out = []
    for e in range(e0, e1):
        for k in range(2, 10):
            out.append(k * 10.0 ** e)
    return out


def ticks(vals, fmt=None):
    return [(v, (fmt(v) if fmt else fmt_num(v))) for v in vals]


def two_panels(W=468.0, left=48.0, gap=66.0, right=8.0):
    """x positions and width of two equal panels across the figure."""
    pw = (W - left - gap - right) / 2
    return left, left + pw + gap, pw


def legend_row_width(entries, gap=12.0):
    return sum(legend_entry_width(e[0]) for e in entries) + gap * (len(entries) - 1)


def labelled_row(head, entries, y, W, gap=10.0):
    """'head:' followed by legend entries, the whole row centred on the figure."""
    hw = sx.texw(head, LEGEND_PT) + 8
    total = hw + legend_row_width(entries, gap)
    x = W / 2 - total / 2
    sx.tex(head, x, y, LEGEND_PT, INK, valign="middle")
    legend_row(entries, x + hw, y, gap=gap)
