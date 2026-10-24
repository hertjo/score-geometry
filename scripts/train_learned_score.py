#!/usr/bin/env python3
"""Train and evaluate the denoisers for the learned-score experiments.

Only NumPy and ReportLab are used, so each model is a hand-written
two-hidden-layer MLP whose input Jacobian is available analytically.  Training
is denoising regression on fresh Gaussian pairs.  Minibatches mix ordinary and
query-focused samples; the reweighting depends only on the noisy input Y, so
E[X | Y=y] is unchanged wherever the weight is positive.

Outputs are the per-seed measurements, the summaries, the run metadata and the
figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Keep parallel replications from spawning nested BLAS thread pools.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

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

BLUE = HexColor("#285C86")
ORANGE = HexColor("#C45A00")
TEAL = HexColor("#007F7B")
PURPLE = HexColor("#70539B")
GRAY = HexColor("#68717D")
CRIMSON = HexColor("#96233C")
GREEN = HexColor("#3F6B23")
LIGHT = HexColor("#DCE1E8")
BLACK = HexColor("#20242B")


@dataclass(frozen=True)
class TrainConfig:
    hidden: int = 96
    steps: int = 3500
    batch_size: int = 768
    pool_factor: int = 8
    focus_fraction: float = 0.85
    learning_rate: float = 2.5e-3
    final_learning_rate: float = 2.5e-4
    weight_decay: float = 2.0e-6
    gradient_clip: float = 8.0
    seeds: int = 12
    circle_sigmas: tuple[float, ...] = (0.08, 0.12, 0.18)
    plane_sigmas: tuple[float, ...] = (0.12, 0.18, 0.25)
    circle_focus_radius: float = 0.22
    plane_focus_radius: float = 0.38
    plane_dim: int = 16
    plane_order: int = 4
    plane_anisotropy: float = 0.10
    plane_tau: float = 0.35
    validation_size: int = 20_000


class Adam:
    def __init__(self, parameters: list[np.ndarray]):
        self.m = [np.zeros_like(p) for p in parameters]
        self.v = [np.zeros_like(p) for p in parameters]
        self.t = 0

    def step(self, parameters, gradients, learning_rate):
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        correction1 = 1.0 - beta1**self.t
        correction2 = 1.0 - beta2**self.t
        for i, (parameter, gradient) in enumerate(zip(parameters, gradients)):
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * gradient
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * gradient * gradient
            parameter -= learning_rate * (self.m[i] / correction1) / (
                np.sqrt(self.v[i] / correction2) + eps
            )


class DenoiserMLP:
    """Residual noise-conditional MLP with an analytic input Jacobian."""

    def __init__(self, ambient_dim, sigmas, hidden, y_scale, rng):
        self.d = int(ambient_dim)
        self.sigmas = np.asarray(sigmas, dtype=np.float64)
        self.log_lo = float(np.log(np.min(self.sigmas)))
        self.log_hi = float(np.log(np.max(self.sigmas)))
        self.y_scale = float(y_scale)
        din = self.d + 1
        h = int(hidden)
        self.W1 = (rng.normal(size=(din, h)) * math.sqrt(2.0 / (din + h))).astype(np.float64)
        self.b1 = np.zeros(h, dtype=np.float64)
        self.W2 = (rng.normal(size=(h, h)) * math.sqrt(1.0 / h)).astype(np.float64)
        self.b2 = np.zeros(h, dtype=np.float64)
        self.W3 = (rng.normal(size=(h, self.d)) * 0.02).astype(np.float64)
        self.b3 = np.zeros(self.d, dtype=np.float64)

    @property
    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def _features(self, y, sigma):
        y = np.asarray(y, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
        if self.log_hi == self.log_lo:
            t = np.zeros_like(sigma)
        else:
            t = 2.0 * (np.log(sigma) - self.log_lo) / (self.log_hi - self.log_lo) - 1.0
        return np.concatenate([y / self.y_scale, t[:, None]], axis=1)

    def forward(self, y, sigma, cache=False):
        x = self._features(y, sigma)
        z1 = x @ self.W1 + self.b1
        h1 = np.tanh(z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.tanh(z2)
        standardized_residual = h2 @ self.W3 + self.b3
        if cache:
            return standardized_residual, (x, h1, h2)
        return standardized_residual

    def denoise(self, y, sigma):
        y = np.asarray(y, dtype=np.float64)
        scalar = np.ndim(sigma) == 0
        if y.ndim == 1:
            y = y[None, :]
        sigma_array = np.full(y.shape[0], float(sigma)) if scalar else np.asarray(sigma)
        residual = self.forward(y, sigma_array)
        out = y + sigma_array[:, None] * residual
        return out[0] if scalar and out.shape[0] == 1 else out

    def loss_and_gradients(self, y, sigma, target_x, weight_decay, gradient_clip):
        residual, (features, h1, h2) = self.forward(y, sigma, cache=True)
        target_residual = (target_x - y) / sigma[:, None]
        error = residual - target_residual
        loss = float(np.mean(error * error))
        upstream = 2.0 * error / error.size

        gW3 = h2.T @ upstream + weight_decay * self.W3
        gb3 = np.sum(upstream, axis=0)
        gh2 = (upstream @ self.W3.T) * (1.0 - h2 * h2)
        gW2 = h1.T @ gh2 + weight_decay * self.W2
        gb2 = np.sum(gh2, axis=0)
        gh1 = (gh2 @ self.W2.T) * (1.0 - h1 * h1)
        gW1 = features.T @ gh1 + weight_decay * self.W1
        gb1 = np.sum(gh1, axis=0)
        gradients = [gW1, gb1, gW2, gb2, gW3, gb3]

        norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients))
        if norm > gradient_clip:
            scale = gradient_clip / norm
            gradients = [g * scale for g in gradients]
        return loss, gradients

    def denoise_and_jacobian(self, y, sigma):
        y = np.asarray(y, dtype=np.float64).reshape(1, self.d)
        sigma = float(sigma)
        x = self._features(y, np.asarray([sigma]))
        h1 = np.tanh(x @ self.W1 + self.b1)[0]
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        residual = h2 @ self.W3 + self.b3
        # Rows are output coordinates and columns are noisy-input coordinates.
        jacobian_residual = (
            self.W3.T
            @ ((1.0 - h2 * h2)[:, None] * self.W2.T)
            @ ((1.0 - h1 * h1)[:, None] * self.W1[: self.d, :].T)
        ) / self.y_scale
        denoised = y[0] + sigma * residual
        jacobian = np.eye(self.d) + sigma * jacobian_residual
        return denoised, jacobian


def _circle_angles(rng, n, order):
    if order == 0:
        return rng.uniform(-math.pi, math.pi, size=n)
    if order != 1:
        raise ValueError("The learned circle benchmark uses order 0 or 1.")
    chunks = []
    have = 0
    while have < n:
        proposal = rng.uniform(-math.pi, math.pi, size=max(4096, 2 * (n - have)))
        keep = rng.random(proposal.size) <= (1.0 - np.cos(proposal)) / 2.0
        accepted = proposal[keep]
        chunks.append(accepted)
        have += accepted.size
    return np.concatenate(chunks)[:n]


def _circle_pairs(rng, n, sigmas, order):
    sigma = rng.choice(np.asarray(sigmas), size=n)
    theta = _circle_angles(rng, n, order)
    clean = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    noisy = clean + sigma[:, None] * rng.normal(size=clean.shape)
    return noisy, sigma, clean


def fixed_plane_frame(d, k=2):
    rng = np.random.default_rng(2026081901)
    q, _ = np.linalg.qr(rng.normal(size=(d, k)))
    # Fix QR's sign convention for cross-platform reproducibility.
    for j in range(k):
        pivot = int(np.argmax(np.abs(q[:, j])))
        if q[pivot, j] < 0:
            q[:, j] *= -1.0
    return q


def _plane_latents(rng, n, order, anisotropy, tau):
    if order % 2:
        raise ValueError("The polynomial order must be even.")
    component_two = rng.random(n) < anisotropy / (1.0 + anisotropy)
    z = tau * rng.normal(size=(n, 2))
    weighted = tau * np.sqrt(rng.chisquare(order + 1, size=n))
    weighted *= rng.choice(np.asarray([-1.0, 1.0]), size=n)
    z[~component_two, 0] = weighted[~component_two]
    z[component_two, 1] = weighted[component_two]
    return z


def _plane_pairs(rng, n, sigmas, frame, order, anisotropy, tau):
    sigma = rng.choice(np.asarray(sigmas), size=n)
    z = _plane_latents(rng, n, order, anisotropy, tau)
    clean = z @ frame.T
    noisy = clean + sigma[:, None] * rng.normal(size=clean.shape)
    return noisy, sigma, clean


def _focused_batch(rng, pair_generator, batch_size, pool_factor,
                   focus_fraction, center, radius):
    pool_size = int(batch_size * pool_factor)
    noisy, sigma, clean = pair_generator(rng, pool_size)
    n_focus = int(round(batch_size * focus_fraction))
    n_global = batch_size - n_focus
    global_indices = rng.choice(pool_size, size=n_global, replace=False)
    distance2 = np.sum((noisy - center[None, :]) ** 2, axis=1)
    probabilities = np.exp(-0.5 * distance2 / (radius * radius)) + 1e-4
    probabilities /= np.sum(probabilities)
    focus_indices = rng.choice(pool_size, size=n_focus, replace=True, p=probabilities)
    indices = np.concatenate([global_indices, focus_indices])
    rng.shuffle(indices)
    return noisy[indices], sigma[indices], clean[indices]


def train_model(problem, seed, config, frame=None):
    rng = np.random.default_rng(seed)
    if problem == "circle_regular":
        d, sigmas, y_scale = 2, config.circle_sigmas, 1.25
        center, radius = np.asarray([1.0, 0.0]), config.circle_focus_radius
        pair_generator = lambda r, n: _circle_pairs(r, n, sigmas, order=0)
    elif problem == "circle_zero":
        d, sigmas, y_scale = 2, config.circle_sigmas, 1.25
        center, radius = np.asarray([1.0, 0.0]), config.circle_focus_radius
        pair_generator = lambda r, n: _circle_pairs(r, n, sigmas, order=1)
    elif problem == "plane_zero":
        d, sigmas, y_scale = config.plane_dim, config.plane_sigmas, 0.85
        center, radius = np.zeros(d), config.plane_focus_radius
        pair_generator = lambda r, n: _plane_pairs(
            r, n, sigmas, frame, config.plane_order,
            config.plane_anisotropy, config.plane_tau
        )
    else:
        raise ValueError(problem)

    model = DenoiserMLP(d, sigmas, config.hidden, y_scale, rng)
    optimizer = Adam(model.parameters)
    losses = []
    start = time.perf_counter()
    for step in range(config.steps):
        y, sigma, clean = _focused_batch(
            rng, pair_generator, config.batch_size, config.pool_factor,
            config.focus_fraction, center, radius
        )
        phase = step / max(1, config.steps - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * phase))
        learning_rate = config.final_learning_rate + (
            config.learning_rate - config.final_learning_rate
        ) * cosine
        loss, gradients = model.loss_and_gradients(
            y, sigma, clean, config.weight_decay, config.gradient_clip
        )
        optimizer.step(model.parameters, gradients, learning_rate)
        if step % 100 == 0 or step == config.steps - 1:
            losses.append(loss)
    elapsed = time.perf_counter() - start

    yv, sv, xv = pair_generator(rng, config.validation_size)
    prediction = model.denoise(yv, sv)
    global_mse = float(np.mean((prediction - xv) ** 2))
    yf, sf, xf = _focused_batch(
        rng, pair_generator, config.validation_size, max(2, config.pool_factor),
        1.0, center, radius
    )
    focused_prediction = model.denoise(yf, sf)
    focused_mse = float(np.mean((focused_prediction - xf) ** 2))
    diagnostics = {
        "last_training_loss": float(losses[-1]),
        "global_denoising_mse": global_mse,
        "focused_denoising_mse": focused_mse,
        "training_seconds": float(elapsed),
    }
    return model, diagnostics


def _sym(matrix):
    return 0.5 * (matrix + matrix.T)


def _opnorm(matrix):
    return float(np.linalg.norm(matrix, ord=2))


def _hard_projector(matrix, k):
    values, vectors = np.linalg.eigh(_sym(matrix))
    basis = vectors[:, -k:]
    return basis @ basis.T, values


def _soft_calibration(matrix, tau):
    values, vectors = np.linalg.eigh(_sym(matrix))
    clipped = np.maximum(values, 0.0)
    filtered = clipped / (clipped + tau)
    return (vectors * filtered[None, :]) @ vectors.T


def _circle_population(sigma, order, n=120_001):
    # Periodic trapezoidal quadrature: do not count the identified endpoint twice.
    theta = np.linspace(-math.pi, math.pi, n, endpoint=False)
    one_minus_cos = 1.0 - np.cos(theta)
    density = np.ones_like(theta) if order == 0 else one_minus_cos
    log_kernel = -one_minus_cos / (sigma * sigma)
    log_kernel -= np.max(log_kernel)
    weights = density * np.exp(log_kernel)
    weights /= np.sum(weights)
    points = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    mean = np.sum(weights[:, None] * points, axis=0)
    centered = points - mean
    hessian = (centered.T * weights) @ centered / (sigma * sigma)
    return mean, hessian


def _plane_population(sigma, frame, config):
    alpha = config.plane_tau**2 / (config.plane_tau**2 + sigma**2)
    m = config.plane_order
    eps = config.plane_anisotropy
    c1 = (m + 1.0 + eps) / (1.0 + eps)
    c2 = (1.0 + eps * (m + 1.0)) / (1.0 + eps)
    tangent = np.diag([alpha * c1, alpha * c2])
    return frame @ tangent @ frame.T, np.asarray([alpha * c1, alpha * c2])


def _residual_diagnostics(a_hat, projector, k):
    """Coupling block, computed separation and their ratio.

    These are the numerator and denominator of the residual sin-Theta bound,
    which unlike the operator-norm bound ignores error in the tangent block.
    """
    dim = projector.shape[0]
    _, vectors = np.linalg.eigh(projector)
    tangent = vectors[:, dim - k:]
    normal = vectors[:, : dim - k]
    coupling = _opnorm((np.eye(dim) - projector) @ a_hat @ projector)
    low = float(np.linalg.eigvalsh(tangent.T @ a_hat @ tangent)[0])
    high = float(np.linalg.eigvalsh(normal.T @ a_hat @ normal)[-1])
    separation = low - high
    bound = coupling / separation if separation > 0 else float("inf")
    return coupling, separation, bound


def evaluate_model(problem, seed, model, diagnostics, config, frame=None):
    rows = []
    if problem.startswith("circle"):
        order = 0 if problem == "circle_regular" else 1
        p = np.asarray([1.0, 0.0])
        projector = np.diag([0.0, 1.0])
        sigmas = config.circle_sigmas
        k = 1
    else:
        p = np.zeros(config.plane_dim)
        projector = frame @ frame.T
        sigmas = config.plane_sigmas
        k = 2

    for sigma in sigmas:
        denoised, jacobian = model.denoise_and_jacobian(p, sigma)
        learned_h = _sym(jacobian)
        hard, eigenvalues = _hard_projector(learned_h, k)
        soft = _soft_calibration(learned_h, tau=0.25)
        if problem.startswith("circle"):
            true_d, true_h = _circle_population(sigma, order)
            true_tangent_eigenvalues = [float(true_h[1, 1])]
        else:
            true_d = np.zeros_like(p)
            true_h, true_tangent_eigenvalues = _plane_population(sigma, frame, config)
        top = np.sort(eigenvalues)[-k:][::-1]
        coupling, separation, residual_bound = _residual_diagnostics(
            learned_h, projector, k
        )
        row = {
            "seed": seed,
            "problem": problem,
            "sigma": sigma,
            "raw_projector_error": _opnorm(learned_h - projector),
            "hard_projector_error": _opnorm(hard - projector),
            "soft_projector_error": _opnorm(soft - projector),
            "population_jacobian_error": _opnorm(learned_h - true_h),
            "coupling_block_norm": coupling,
            "computed_separation": separation,
            "residual_bound": residual_bound,
            "denoiser_value_error": float(np.linalg.norm(denoised - true_d)),
            "estimated_rank_threshold_half": int(np.sum(eigenvalues > 0.5)),
            "true_rank": k,
            "top_eigenvalue": float(top[0]),
            "true_top_eigenvalue": float(true_tangent_eigenvalues[0]),
            "second_eigenvalue": float(top[1]) if k == 2 else float("nan"),
            "true_second_eigenvalue": (
                float(true_tangent_eigenvalues[1]) if k == 2 else float("nan")
            ),
            **diagnostics,
        }
        rows.append(row)
    return rows


def _circle_objective_gap(x):
    norm = float(np.linalg.norm(x))
    if not np.isfinite(norm) or norm < 1e-12:
        return 1e6, 1e6
    projected = x / norm
    return max(0.0, 1.0 - float(projected[0])), abs(norm - 1.0)


def _plane_objective_gap(x, frame):
    if not np.all(np.isfinite(x)):
        return 1e6, 1e6
    tangent = frame.T @ x
    normal = x - frame @ tangent
    return 0.5 * float(tangent @ tangent), float(np.linalg.norm(normal))


def _run_optimizer(model, problem, sigma, method, steps, gamma, frame=None):
    if problem == "circle_zero":
        angle = 0.55
        x = np.asarray([math.cos(angle), math.sin(angle)])
        target = np.asarray([1.0, 0.0])
        # f(x)=||x-target||^2/2 has ambient gradient zero at the minimizer,
        # as required by the local optimization corollary.  On the unit
        # circle its objective gap is exactly 1-x_1, which is what
        # _circle_objective_gap reports after radial projection.
        gradient = lambda value: value - target
        measure = _circle_objective_gap
        k = 1
    elif problem == "plane_zero":
        x = frame @ np.asarray([0.40, -0.32])
        projector = frame @ frame.T
        gradient = lambda value: projector @ value
        measure = lambda value: _plane_objective_gap(value, frame)
        k = 2
    else:
        raise ValueError(problem)

    records = []
    for iteration in range(steps + 1):
        gap, feasibility = measure(x)
        records.append((iteration, min(gap, 1e6), min(feasibility, 1e6)))
        if iteration == steps or gap >= 1e6:
            continue
        if method == "oracle":
            if problem == "circle_zero":
                projected = x / max(float(np.linalg.norm(x)), 1e-12)
                q_oracle = np.eye(2) - np.outer(projected, projected)
                z = x - gamma * (q_oracle @ gradient(x))
                x = z / max(float(np.linalg.norm(z)), 1e-12)
            else:
                q_oracle = frame @ frame.T
                z = x - gamma * (q_oracle @ gradient(x))
                x = q_oracle @ z
            continue
        denoised_x, jacobian_x = model.denoise_and_jacobian(x, sigma)
        h_x = _sym(jacobian_x)
        q_x, _ = _hard_projector(h_x, k)
        grad = gradient(x)
        if method == "raw":
            z = x - gamma * (h_x @ grad)
            x = model.denoise(z, sigma)
        elif method == "inner_only":
            z = x - gamma * (q_x @ grad)
            x = model.denoise(z, sigma)
        elif method == "calibrated":
            z = x - gamma * (q_x @ grad)
            denoised_z, jacobian_z = model.denoise_and_jacobian(z, sigma)
            q_z, _ = _hard_projector(_sym(jacobian_z), k)
            x = z + (np.eye(z.size) - q_z) @ (denoised_z - z)
        else:
            raise ValueError(method)
        if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 1e6:
            x = np.full_like(x, np.inf)
    return records


def evaluate_optimization(seed, circle_model, plane_model, config, frame):
    rows = []
    # The denoisers are noise conditional, so every trained level can be swept
    # without retraining.
    specifications = [
        ("circle_zero", circle_model, config.circle_sigmas,
         np.linspace(0.05, 1.0, 20), 35),
        ("plane_zero", plane_model, config.plane_sigmas,
         np.linspace(0.05, 1.2, 24), 30),
    ]
    for problem, model, sigmas, gammas, steps in specifications:
        for sigma in sigmas:
            for method in ("oracle", "raw", "inner_only", "calibrated"):
                for gamma in gammas:
                    for iteration, gap, feasibility in _run_optimizer(
                        model, problem, float(sigma), method, steps,
                        float(gamma), frame
                    ):
                        rows.append({
                            "seed": seed,
                            "problem": problem,
                            "sigma": float(sigma),
                            "gamma": float(gamma),
                            "method": method,
                            "iteration": iteration,
                            "objective_gap": gap,
                            "feasibility_error": feasibility,
                        })
    return rows


def run_replication(index, config):
    seed = 2026081900 + index
    frame = fixed_plane_frame(config.plane_dim)
    metric_rows = []
    models = {}
    for offset, problem in enumerate(("circle_regular", "circle_zero", "plane_zero")):
        model, diagnostics = train_model(
            problem, seed + 100_000 * (offset + 1), config, frame
        )
        models[problem] = model
        metric_rows.extend(evaluate_model(
            problem, seed, model, diagnostics, config, frame
        ))
    optimization_rows = evaluate_optimization(
        seed, models["circle_zero"], models["plane_zero"], config, frame
    )
    return metric_rows, optimization_rows


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _quantiles(values):
    values = np.asarray(values, dtype=float)
    return (
        float(np.median(values)),
        float(np.quantile(values, 0.25)),
        float(np.quantile(values, 0.75)),
    )


def summarize_metrics(rows):
    groups = {}
    for row in rows:
        key = (row["problem"], float(row["sigma"]))
        groups.setdefault(key, []).append(row)
    fields = [
        "raw_projector_error", "hard_projector_error", "soft_projector_error",
        "population_jacobian_error", "coupling_block_norm",
        "computed_separation", "residual_bound",
        "denoiser_value_error", "top_eigenvalue",
        "true_top_eigenvalue", "second_eigenvalue", "true_second_eigenvalue",
        "global_denoising_mse", "focused_denoising_mse", "training_seconds",
    ]
    summaries = []
    for (problem, sigma), values in sorted(groups.items()):
        summary = {"problem": problem, "sigma": sigma, "replications": len(values)}
        for field in fields:
            array = [float(v[field]) for v in values if np.isfinite(float(v[field]))]
            if not array:
                continue
            med, q1, q3 = _quantiles(array)
            summary[f"{field}_median"] = med
            summary[f"{field}_q1"] = q1
            summary[f"{field}_q3"] = q3
        summary["rank_recovery_rate"] = float(np.mean([
            int(v["estimated_rank_threshold_half"]) == int(v["true_rank"])
            for v in values
        ]))
        summaries.append(summary)
    return summaries


def summarize_optimization(rows):
    groups = {}
    for row in rows:
        key = (row["problem"], float(row["sigma"]), row["method"], float(row["gamma"]), int(row["iteration"]))
        groups.setdefault(key, []).append(row)
    summaries = []
    for (problem, sigma, method, gamma, iteration), values in sorted(groups.items()):
        gaps = [float(v["objective_gap"]) for v in values]
        feasibilities = [float(v["feasibility_error"]) for v in values]
        med, q1, q3 = _quantiles(gaps)
        fmed, fq1, fq3 = _quantiles(feasibilities)
        summaries.append({
            "problem": problem,
            "sigma": sigma,
            "method": method,
            "gamma": gamma,
            "iteration": iteration,
            "objective_median": med,
            "objective_q1": q1,
            "objective_q3": q3,
            "feasibility_median": fmed,
            "feasibility_q1": fq1,
            "feasibility_q3": fq3,
            "replications": len(values),
        })
    return summaries


def paired_bootstrap(rows, problem, sigma, first, second, samples=20_000):
    selected = [r for r in rows if r["problem"] == problem and abs(float(r["sigma"]) - sigma) < 1e-12]
    a = np.asarray([float(r[first]) for r in selected])
    b = np.asarray([float(r[second]) for r in selected])
    difference = a - b
    rng = np.random.default_rng(2026081999)
    indices = rng.integers(0, difference.size, size=(samples, difference.size))
    boot = np.median(difference[indices], axis=1)
    return {
        "problem": problem,
        "sigma": sigma,
        "contrast": f"{first} minus {second}",
        "paired_median_difference": float(np.median(difference)),
        "bootstrap_95_low": float(np.quantile(boot, 0.025)),
        "bootstrap_95_high": float(np.quantile(boot, 0.975)),
        "paired_wins": int(np.sum(difference > 0.0)),
        "replications": int(difference.size),
        "bootstrap_samples": samples,
    }


def numerical_self_checks():
    """Return deterministic audits of the analytic Jacobian and plane sampler."""
    jacobian_seed = 2026081998
    jacobian_rng = np.random.default_rng(jacobian_seed)
    audit_model = DenoiserMLP(
        ambient_dim=5,
        sigmas=(0.08, 0.12, 0.18),
        hidden=17,
        y_scale=1.1,
        rng=jacobian_rng,
    )
    audit_y = jacobian_rng.normal(size=5)
    audit_sigma = 0.12
    finite_difference_step = 1e-6
    _, analytic = audit_model.denoise_and_jacobian(audit_y, audit_sigma)
    basis = np.eye(audit_y.size)
    finite_difference = np.column_stack([
        (
            audit_model.denoise(
                audit_y + finite_difference_step * basis[j], audit_sigma
            )
            - audit_model.denoise(
                audit_y - finite_difference_step * basis[j], audit_sigma
            )
        )
        / (2.0 * finite_difference_step)
        for j in range(audit_y.size)
    ])
    jacobian_difference = analytic - finite_difference

    sampler_seed = 2026081997
    sampler_size = 4_000_000
    config = TrainConfig()
    sampler_rng = np.random.default_rng(sampler_seed)
    latent = _plane_latents(
        sampler_rng,
        sampler_size,
        config.plane_order,
        config.plane_anisotropy,
        config.plane_tau,
    )
    empirical_moments = np.mean(latent * latent, axis=0)
    m = config.plane_order
    eps = config.plane_anisotropy
    theory_moments = config.plane_tau**2 * np.asarray([
        (m + 1.0 + eps) / (1.0 + eps),
        (1.0 + eps * (m + 1.0)) / (1.0 + eps),
    ])
    relative_errors = np.abs(empirical_moments - theory_moments) / theory_moments
    return {
        "jacobian": {
            "seed": jacobian_seed,
            "ambient_dimension": int(audit_y.size),
            "hidden_width": 17,
            "sigma": audit_sigma,
            "centered_difference_step": finite_difference_step,
            "operator_norm_error": float(np.linalg.norm(jacobian_difference, ord=2)),
            "max_absolute_error": float(np.max(np.abs(jacobian_difference))),
        },
        "plane_sampler": {
            "seed": sampler_seed,
            "sample_size": sampler_size,
            "empirical_second_moments": empirical_moments.tolist(),
            "theoretical_second_moments": theory_moments.tolist(),
            "relative_errors": relative_errors.tolist(),
            "max_relative_error": float(np.max(relative_errors)),
        },
    }


def _map(value, lo, hi, start, end, log=False):
    value = np.asarray(value, dtype=float)
    if log:
        value = np.log10(np.maximum(value, lo))
        lo, hi = math.log10(lo), math.log10(hi)
    return start + (value - lo) * (end - start) / (hi - lo)


def _tick_label(value):
    value = float(value)
    if value and (abs(value) < 1e-3 or abs(value) >= 1e4):
        return f"{value:.0e}".replace("e+0", "e").replace("e-0", "e-")
    return f"{value:g}"


def _panel_box(canvas, x, y, w, h, radius=5.0):
    """Rounded frame around one subfigure."""
    canvas.saveState()
    canvas.setStrokeColor(LIGHT)
    canvas.setLineWidth(0.6)
    canvas.roundRect(x, y, w, h, radius, stroke=1, fill=0)
    canvas.restoreState()


def _auto_box(canvas, x0, y0, w, h, yticks=None, yticklabels=None, top_pad=10.0):
    """Frame one panel, wide enough to contain its axis labels."""
    labels = yticklabels if yticklabels else (
        [_tick_label(t) for t in yticks] if yticks is not None else [])
    widest = max((pdfmetrics.stringWidth(str(t), "Helvetica", 10.2)
                  for t in labels), default=0.0)
    offset = max(38.0, widest + 20.0)
    _panel_box(canvas, x0 - offset - 8.0, y0 - 36.0,
               w + offset + 18.0, h + 36.0 + top_pad)


def _axes(canvas, x0, y0, width, height, xlim, ylim, xlabel, ylabel,
          xlog=False, ylog=False, xticks=None, yticks=None, xticklabels=None,
          yticklabels=None):
    canvas.setFont("Helvetica", 10.2)
    if xticks is None:
        xticks = np.geomspace(*xlim, 4) if xlog else np.linspace(*xlim, 4)
    if yticks is None:
        yticks = np.geomspace(*ylim, 4) if ylog else np.linspace(*ylim, 4)
    canvas.setStrokeColor(LIGHT)
    canvas.setLineWidth(0.45)
    for i, tick in enumerate(xticks):
        xx = float(_map(tick, *xlim, x0, x0 + width, log=xlog))
        canvas.line(xx, y0, xx, y0 + height)
    for tick in yticks:
        yy = float(_map(tick, *ylim, y0, y0 + height, log=ylog))
        canvas.line(x0, yy, x0 + width, yy)
    canvas.setStrokeColor(BLACK)
    canvas.setLineWidth(0.75)
    canvas.line(x0, y0, x0 + width, y0)
    canvas.line(x0, y0, x0, y0 + height)
    canvas.setFillColor(BLACK)
    for i, tick in enumerate(xticks):
        xx = float(_map(tick, *xlim, x0, x0 + width, log=xlog))
        canvas.line(xx, y0 - 2.3, xx, y0)
        label = xticklabels[i] if xticklabels else _tick_label(tick)
        canvas.drawCentredString(xx, y0 - 12, label)
    for i, tick in enumerate(yticks):
        yy = float(_map(tick, *ylim, y0, y0 + height, log=ylog))
        canvas.line(x0 - 2.3, yy, x0, yy)
        label = yticklabels[i] if yticklabels else _tick_label(tick)
        canvas.drawRightString(x0 - 4.5, yy - 2.5, label)
    canvas.setFont("Helvetica", 11.0)
    canvas.drawCentredString(x0 + width / 2, y0 - 27, xlabel)
    canvas.saveState()
    drawn = yticklabels if yticklabels else [_tick_label(t) for t in yticks]
    widest = max((pdfmetrics.stringWidth(str(t), "Helvetica", 10.2)
                  for t in drawn), default=0.0)
    canvas.translate(x0 - max(38.0, widest + 20.0), y0 + height / 2)
    canvas.rotate(90)
    canvas.drawCentredString(0, 0, ylabel)
    canvas.restoreState()


def _curve(canvas, x, y, x0, y0, width, height, xlim, ylim, color,
           xlog=False, ylog=False, line_width=1.5, dash=None, marker=None):
    xp = _map(x, *xlim, x0, x0 + width, log=xlog)
    yp = _map(y, *ylim, y0, y0 + height, log=ylog)
    path = canvas.beginPath()
    path.moveTo(float(xp[0]), float(yp[0]))
    for xx, yy in zip(xp[1:], yp[1:]):
        path.lineTo(float(xx), float(yy))
    canvas.saveState()
    clip = canvas.beginPath()
    clip.rect(x0, y0, width, height)
    canvas.clipPath(clip, stroke=0, fill=0)
    canvas.setStrokeColor(color)
    canvas.setLineWidth(line_width)
    canvas.setLineCap(1)
    canvas.setLineJoin(1)
    canvas.setDash(dash or [])
    canvas.drawPath(path, fill=0, stroke=1)
    canvas.setDash([])
    if marker:
        for xx, yy in zip(xp, yp):
            _marker(canvas, float(xx), float(yy), marker, color, size=2.2)
    canvas.restoreState()


def _band(canvas, x, low, high, x0, y0, width, height, xlim, ylim, color,
          xlog=False, ylog=False):
    xp = _map(x, *xlim, x0, x0 + width, log=xlog)
    lo = _map(low, *ylim, y0, y0 + height, log=ylog)
    hi = _map(high, *ylim, y0, y0 + height, log=ylog)
    path = canvas.beginPath()
    path.moveTo(float(xp[0]), float(lo[0]))
    for xx, yy in zip(xp[1:], lo[1:]):
        path.lineTo(float(xx), float(yy))
    for xx, yy in zip(xp[::-1], hi[::-1]):
        path.lineTo(float(xx), float(yy))
    path.close()
    canvas.saveState()
    clip = canvas.beginPath()
    clip.rect(x0, y0, width, height)
    canvas.clipPath(clip, stroke=0, fill=0)
    canvas.setFillColor(color)
    canvas.setFillAlpha(0.13)
    canvas.drawPath(path, fill=1, stroke=0)
    canvas.restoreState()


def _marker(canvas, x, y, marker, color, size=2.2, filled=True):
    canvas.setStrokeColor(color)
    canvas.setFillColor(color if filled else HexColor("#FFFFFF"))
    canvas.setLineWidth(0.9)
    if marker == "circle":
        canvas.circle(x, y, size, fill=1, stroke=1)
    elif marker == "square":
        canvas.rect(x - size, y - size, 2 * size, 2 * size, fill=1, stroke=1)
    elif marker == "triangle":
        path = canvas.beginPath()
        path.moveTo(x, y + 1.2 * size)
        path.lineTo(x - 1.1 * size, y - .9 * size)
        path.lineTo(x + 1.1 * size, y - .9 * size)
        path.close()
        canvas.drawPath(path, fill=1, stroke=1)
    elif marker == "triangle_down":
        path = canvas.beginPath()
        path.moveTo(x, y - 1.2 * size)
        path.lineTo(x - 1.1 * size, y + .9 * size)
        path.lineTo(x + 1.1 * size, y + .9 * size)
        path.close()
        canvas.drawPath(path, fill=1, stroke=1)
    elif marker == "diamond":
        path = canvas.beginPath()
        path.moveTo(x, y + 1.2 * size)
        path.lineTo(x - 1.1 * size, y)
        path.lineTo(x, y - 1.2 * size)
        path.lineTo(x + 1.1 * size, y)
        path.close()
        canvas.drawPath(path, fill=1, stroke=1)


def _legend_grid(canvas, entries, x, y, columns, column_width,
                 row_height=13, font_size=9.2):
    canvas.setFont("Helvetica", font_size)
    for i, entry in enumerate(entries):
        label, color, dash = entry[:3]
        marker = entry[3] if len(entry) > 3 else None
        column, row = i % columns, i // columns
        xx = x + column * column_width
        yy = y - row * row_height
        canvas.setStrokeColor(color)
        canvas.setLineWidth(1.7)
        canvas.setDash(dash or [])
        canvas.line(xx, yy, xx + 16, yy)
        canvas.setDash([])
        if marker:
            _marker(canvas, xx + 8, yy, marker, color, size=2.0)
        canvas.setFillColor(BLACK)
        canvas.drawString(xx + 21, yy - 2.5, label)


def _panel_title(canvas, x, y, text):
    canvas.setFillColor(BLACK)
    canvas.setFont("Helvetica-Bold", 11.5)
    canvas.drawString(x, y, text)


def _summary_lookup(summaries, problem, field):
    selected = sorted(
        [row for row in summaries if row["problem"] == problem],
        key=lambda row: float(row["sigma"]),
    )
    x = np.asarray([float(row["sigma"]) for row in selected])
    med = np.asarray([float(row[f"{field}_median"]) for row in selected])
    low = np.asarray([float(row[f"{field}_q1"]) for row in selected])
    high = np.asarray([float(row[f"{field}_q3"]) for row in selected])
    return x, med, low, high


def make_learned_geometry_figure(summaries):
    canvas = Canvas(str(FIG / "learned_geometry.pdf"), pagesize=(590, 315))
    x1, x2, y0, width, height = 58, 365, 45, 205, 180
    xlim = (0.075, 0.19)
    _auto_box(canvas, x1, y0, width, height, yticks=[0, 1, 2, 3], top_pad=10.0)
    _axes(canvas, x1, y0, width, height, xlim, (0.0, 3.4),
          "noise level \u03c3", "tangent eigenvalue", xticks=[0.08, 0.12, 0.18],
          yticks=[0, 1, 2, 3])
    for problem, color, marker in (("circle_zero", ORANGE, "circle"),
                                   ("circle_regular", BLUE, "square")):
        x, med, low, high = _summary_lookup(summaries, problem, "top_eigenvalue")
        _band(canvas, x, low, high, x1, y0, width, height, xlim, (0.0, 3.4), color)
        _curve(canvas, x, med, x1, y0, width, height, xlim, (0.0, 3.4), color,
               marker=marker)
        _, truth, _, _ = _summary_lookup(summaries, problem, "true_top_eigenvalue")
        _curve(canvas, x, truth, x1, y0, width, height, xlim, (0.0, 3.4),
               color, dash=[4, 2], line_width=1.1)
    _panel_title(canvas, x1, 301, "(a) Learned circle spectrum")
    _legend_grid(canvas, [("quadratic zero", ORANGE, None),
                          ("positive density", BLUE, None),
                          ("learned median", BLACK, None, "circle"),
                          ("population", BLACK, [4, 2])],
                 x1, 280, columns=2, column_width=102)

    _auto_box(canvas, x2, y0, width, height, yticklabels=["0.001", "0.01", "0.1", "1"], top_pad=10.0)
    _axes(canvas, x2, y0, width, height, xlim, (1e-3, 5.0),
          "noise level \u03c3", "operator-norm error", ylog=True,
          xticks=[0.08, 0.12, 0.18], yticks=[1e-3, 1e-2, 1e-1, 1],
          yticklabels=["0.001", "0.01", "0.1", "1"])
    for field, color, dash, marker in (
        ("raw_projector_error", CRIMSON, None, "circle"),
        ("hard_projector_error", TEAL, [5, 2], "square"),
        ("population_jacobian_error", PURPLE, [1.2, 2], "triangle"),
        ("coupling_block_norm", GRAY, [4, 1.5, 1, 1.5], "diamond"),
    ):
        x, med, low, high = _summary_lookup(summaries, "circle_zero", field)
        low = np.maximum(low, 1e-3)
        high = np.maximum(high, 1e-3)
        med = np.maximum(med, 1e-3)
        _band(canvas, x, low, high, x2, y0, width, height, xlim, (1e-3, 5.0),
              color, ylog=True)
        _curve(canvas, x, med, x2, y0, width, height, xlim, (1e-3, 5.0),
               color, ylog=True, dash=dash, marker=marker)
    _panel_title(canvas, x2, 301, "(b) Learned range recovery")
    _legend_grid(canvas, [("raw link vs tangent", CRIMSON, None, "circle"),
                          ("hard-calibrated range", TEAL, [5, 2], "square"),
                          ("learned vs population", PURPLE, [1.2, 2], "triangle"),
                          ("coupling block", GRAY, [4, 1.5, 1, 1.5], "diamond")],
                 x2, 280, columns=1, column_width=200, row_height=13)
    canvas.save()


def make_learned_plane_figure(summaries):
    canvas = Canvas(str(FIG / "learned_plane.pdf"), pagesize=(590, 345))
    x1, x2, y0, width, height = 58, 365, 45, 205, 175
    xlim = (0.11, 0.26)
    _auto_box(canvas, x1, y0, width, height, yticks=[0, 1, 2, 3, 4, 5], top_pad=10.0)
    _axes(canvas, x1, y0, width, height, xlim, (0.0, 5.2),
          "noise level \u03c3", "tangent eigenvalue", xticks=[0.12, 0.18, 0.25],
          yticks=[0, 1, 2, 3, 4, 5])
    x, top, top_lo, top_hi = _summary_lookup(summaries, "plane_zero", "top_eigenvalue")
    _, second, second_lo, second_hi = _summary_lookup(summaries, "plane_zero", "second_eigenvalue")
    _, true_top, _, _ = _summary_lookup(summaries, "plane_zero", "true_top_eigenvalue")
    _, true_second, _, _ = _summary_lookup(summaries, "plane_zero", "true_second_eigenvalue")
    for med, low, high, truth, color, marker in (
        (top, top_lo, top_hi, true_top, ORANGE, "circle"),
        (second, second_lo, second_hi, true_second, TEAL, "square"),
    ):
        _band(canvas, x, low, high, x1, y0, width, height, xlim, (0.0, 5.2), color)
        _curve(canvas, x, med, x1, y0, width, height, xlim, (0.0, 5.2),
               color, marker=marker)
        _curve(canvas, x, truth, x1, y0, width, height, xlim, (0.0, 5.2),
               color, dash=[4, 2], line_width=1.1)
    _panel_title(canvas, x1, 330, "(a) Learned spectrum (d = 16)")
    _legend_grid(canvas, [("first eigenvalue", ORANGE, None),
                          ("second eigenvalue", TEAL, None),
                          ("learned median", BLACK, None, "circle"),
                          ("population", BLACK, [4, 2])],
                 x1, 304, columns=2, column_width=102, row_height=18)

    _auto_box(canvas, x2, y0, width, height, yticklabels=["0.001", "0.01", "0.1", "1"], top_pad=10.0)
    _axes(canvas, x2, y0, width, height, xlim, (1e-3, 6.0),
          "noise level \u03c3", "operator-norm error", ylog=True,
          xticks=[0.12, 0.18, 0.25], yticks=[1e-3, 1e-2, 1e-1, 1],
          yticklabels=["0.001", "0.01", "0.1", "1"])
    for field, color, dash, marker in (
            ("raw_projector_error", CRIMSON, None, "circle"),
            ("hard_projector_error", BLUE, [5, 2], "square"),
            ("soft_projector_error", GREEN, [2.5, 1.5], "triangle_down"),
            ("population_jacobian_error", PURPLE, [1.2, 2], "triangle"),
            ("coupling_block_norm", GRAY, [4, 1.5, 1, 1.5], "diamond")):
        x, med, low, high = _summary_lookup(summaries, "plane_zero", field)
        med, low, high = [np.maximum(v, 1e-3) for v in (med, low, high)]
        _band(canvas, x, low, high, x2, y0, width, height, xlim, (1e-3, 6.0),
              color, ylog=True)
        _curve(canvas, x, med, x2, y0, width, height, xlim, (1e-3, 6.0),
               color, ylog=True, dash=dash, marker=marker)
    _panel_title(canvas, x2, 330, "(b) Tangent-range recovery")
    _legend_grid(canvas, [("raw link", CRIMSON, None, "circle"),
                          ("hard top-2", BLUE, [5, 2], "square"),
                          ("soft calibration", GREEN, [2.5, 1.5], "triangle_down"),
                          ("learned vs population", PURPLE, [1.2, 2], "triangle"),
                          ("coupling block", GRAY, [4, 1.5, 1, 1.5], "diamond")],
                 x2, 304, columns=1, column_width=200, row_height=18)
    canvas.save()


def _population_spectrum(problem, sigma, config=None, frame=None):
    """Exact finite-noise population tangent eigenvalues at the query point."""
    config = config or TrainConfig()
    if problem.startswith("circle"):
        _, hessian = _circle_population(sigma, 1)
        return [float(hessian[1, 1])]
    frame = fixed_plane_frame(config.plane_dim) if frame is None else frame
    _, eigenvalues = _plane_population(sigma, frame, config)
    return [float(value) for value in eigenvalues]


def _stable_intervals(eigenvalues):
    """Step-size intervals on which each update has spectral radius below one.

    The raw update is C(I - gamma C), inner-only is C(I - gamma I), and fully
    calibrated is I - gamma I; intervals are intersected over eigenvalues.
    """
    raw_lo = max((c - 1.0) / (c * c) for c in eigenvalues)
    raw_hi = min((c + 1.0) / (c * c) for c in eigenvalues)
    inner_lo = 1.0 - 1.0 / max(eigenvalues)
    inner_hi = 1.0 + 1.0 / max(eigenvalues)
    return {"raw": (raw_lo, raw_hi), "inner_only": (inner_lo, inner_hi),
            "calibrated": (0.0, 2.0)}


def _interval_bar(canvas, lo, hi, x0, width, xlim, y, color, thickness=3.0):
    """Draw a stability interval as a bar above the axes."""
    left = float(_map(np.asarray([max(lo, xlim[0])]), *xlim, x0, x0 + width)[0])
    right = float(_map(np.asarray([min(hi, xlim[1])]), *xlim, x0, x0 + width)[0])
    if right <= left:
        return
    canvas.saveState()
    canvas.setFillColor(color)
    canvas.setStrokeColor(color)
    canvas.setLineWidth(0.8)
    canvas.rect(left, y, right - left, thickness, stroke=0, fill=1)
    for edge in (left, right):
        canvas.line(edge, y - 1.6, edge, y + thickness + 1.6)
    canvas.restoreState()


def make_learned_optimization_figure(summaries):
    canvas = Canvas(str(FIG / "learned_optimization.pdf"), pagesize=(590, 300))
    panels = [("circle_zero", 0.12, 58, (0.05, 1.0), 35, "Circle (35 iterations)"),
              ("plane_zero", 0.18, 365, (0.05, 1.2), 30, "Rotated 2-plane (30 iterations)")]
    y0, width, height = 58, 205, 166
    colors = {"oracle": GRAY, "raw": ORANGE, "inner_only": PURPLE,
              "calibrated": BLUE}
    labels = {"oracle": "Riemannian oracle",
              "raw": "fully raw",
              "inner_only": "inner calibrated only",
              "calibrated": "SC-DRGD"}
    styles = {"oracle": ([4, 2], "diamond"),
              "raw": ([7, 2], "square"),
              "inner_only": ([1.2, 2], "triangle"),
              "calibrated": (None, "circle")}
    for panel_index, (problem, panel_sigma, x0, xlim, final_iteration, title) in enumerate(panels):
        ticks = ([0.05, .25, .5, .75, 1.0] if problem == "circle_zero"
                 else [0.05, .3, .6, .9, 1.2])
        _auto_box(canvas, x0, y0, width, height, yticklabels=["1e-8", "1e-6", "1e-4", "1e-2", "1"], top_pad=18.0)
        _axes(canvas, x0, y0, width, height, xlim, (1e-8, 1.0),
              "step size \u03b3", "final objective gap", ylog=True,
              xticks=ticks, yticks=[1e-8, 1e-6, 1e-4, 1e-2, 1],
              yticklabels=["1e-8", "1e-6", "1e-4", "1e-2", "1"])
        intervals = _stable_intervals(_population_spectrum(problem, panel_sigma))
        # Stack the bars only if they overlap, so height carries no meaning.
        bars = [(method,) + intervals[method] for method in ("raw", "inner_only")]
        overlap = any(a[2] > b[1] and b[2] > a[1]
                      for a in bars for b in bars if a[0] != b[0])
        for row, (method, lo, hi) in enumerate(bars):
            canvas_y = y0 + height + 5.0 + (row * 7.5 if overlap else 0.0)
            _interval_bar(canvas, lo, hi, x0, width, xlim, canvas_y,
                          colors[method])
        canvas.setFillColor(GRAY)
        canvas.setFont("Helvetica", 7.6)
        canvas.drawRightString(x0 - 4.5, y0 + height + 9.0, "stable \u03b3")
        for method in ("raw", "inner_only", "calibrated", "oracle"):
            selected = sorted(
                [row for row in summaries if row["problem"] == problem
                 and row["method"] == method
                 and abs(float(row["sigma"]) - panel_sigma) < 1e-12
                 and int(row["iteration"]) == final_iteration],
                key=lambda row: float(row["gamma"]),
            )
            x = np.asarray([float(row["gamma"]) for row in selected])
            med = np.maximum([float(row["objective_median"]) for row in selected], 1e-8)
            low = np.maximum([float(row["objective_q1"]) for row in selected], 1e-8)
            high = np.maximum([float(row["objective_q3"]) for row in selected], 1e-8)
            med, low, high = [np.minimum(v, 1.0) for v in (med, low, high)]
            _band(canvas, x, low, high, x0, y0, width, height, xlim, (1e-8, 1.0),
                  colors[method], ylog=True)
            dash, _ = styles[method]
            _curve(canvas, x, med, x0, y0, width, height, xlim, (1e-8, 1.0),
                   colors[method], ylog=True, dash=dash, marker=None)
        _panel_title(canvas, x0, 250, f"({chr(97 + panel_index)}) {title}")
    legend_entries = [(labels[m], colors[m], styles[m][0])
                      for m in ("calibrated", "oracle", "inner_only", "raw")]
    _legend_grid(canvas, legend_entries, 58, 290, columns=4, column_width=130)
    canvas.setFillColor(GRAY)
    canvas.setFont("Helvetica", 8.2)
    canvas.drawRightString(570, 9, "values at or below 1e-8 are shown at the display floor")
    canvas.save()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--steps", type=int, default=3500)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--quick", action="store_true",
                        help="Run a one-seed, short smoke test.")
    parser.add_argument("--plots-only", action="store_true",
                        help="Regenerate figures from existing summary CSV files.")
    parser.add_argument("--refresh-oracle", action="store_true",
                        help="Add deterministic oracle optimization baselines to existing runs.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.refresh_oracle:
        with (DATA / "learned_optimization.csv").open() as stream:
            rows = [row for row in csv.DictReader(stream) if row["method"] != "oracle"]
        seeds = sorted({int(row["seed"]) for row in rows})
        frame = fixed_plane_frame(TrainConfig().plane_dim)
        specifications = [
            ("circle_zero", 0.12, np.linspace(0.05, 1.0, 20), 35),
            ("plane_zero", 0.18, np.linspace(0.05, 1.2, 24), 30),
        ]
        for seed in seeds:
            for problem, sigma, gammas, steps in specifications:
                for gamma in gammas:
                    for iteration, gap, feasibility in _run_optimizer(
                        None, problem, sigma, "oracle", steps, float(gamma), frame
                    ):
                        rows.append({
                            "seed": seed, "problem": problem, "sigma": sigma,
                            "gamma": float(gamma), "method": "oracle",
                            "iteration": iteration, "objective_gap": gap,
                            "feasibility_error": feasibility,
                        })
        _write_csv(DATA / "learned_optimization.csv", rows)
        summaries = summarize_optimization(rows)
        _write_csv(DATA / "learned_optimization_summary.csv", summaries)
        make_learned_optimization_figure(summaries)
        print("refreshed deterministic oracle optimization baselines")
        return
    if args.plots_only:
        with (DATA / "learned_score_summary.csv").open() as stream:
            metric_summaries = list(csv.DictReader(stream))
        with (DATA / "learned_optimization_summary.csv").open() as stream:
            optimization_summaries = list(csv.DictReader(stream))
        make_learned_geometry_figure(metric_summaries)
        make_learned_plane_figure(metric_summaries)
        make_learned_optimization_figure(optimization_summaries)
        print(f"regenerated learned-model figures in {FIG}")
        return
    if args.quick:
        config = TrainConfig(seeds=1, steps=120, hidden=32, batch_size=256,
                             validation_size=2_000)
    else:
        config = TrainConfig(seeds=args.seeds, steps=args.steps,
                             hidden=args.hidden, batch_size=args.batch_size)

    metric_rows, optimization_rows = [], []
    wall_start = time.perf_counter()
    for index in range(config.seeds):
        start = time.perf_counter()
        metrics, optimization = run_replication(index, config)
        metric_rows.extend(metrics)
        optimization_rows.extend(optimization)
        print(
            f"replication {index + 1}/{config.seeds} complete "
            f"in {time.perf_counter() - start:.1f}s",
            flush=True,
        )

    metric_summaries = summarize_metrics(metric_rows)
    optimization_summaries = summarize_optimization(optimization_rows)
    contrasts = [
        paired_bootstrap(metric_rows, "circle_zero", 0.12,
                         "raw_projector_error", "hard_projector_error"),
        paired_bootstrap(metric_rows, "plane_zero", 0.18,
                         "raw_projector_error", "hard_projector_error"),
    ]
    _write_csv(DATA / "learned_score_metrics.csv", metric_rows)
    _write_csv(DATA / "learned_score_summary.csv", metric_summaries)
    _write_csv(DATA / "learned_optimization.csv", optimization_rows)
    _write_csv(DATA / "learned_optimization_summary.csv", optimization_summaries)
    _write_csv(DATA / "learned_paired_bootstrap.csv", contrasts)

    make_learned_geometry_figure(metric_summaries)
    make_learned_plane_figure(metric_summaries)
    make_learned_optimization_figure(optimization_summaries)

    checks = numerical_self_checks()
    with (DATA / "learned_numerical_checks.json").open("w") as stream:
        json.dump(checks, stream, indent=2)

    metadata = {
        "configuration": asdict(config),
        "base_seed": 2026081900,
        "frame_seed": 2026081901,
        "bootstrap_seed": 2026081999,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "total_wall_seconds": time.perf_counter() - wall_start,
        "notes": [
            "CPU-only NumPy implementation; no accelerator was used.",
            "Every MLP parameter is optimized; Jacobians are analytic.",
            "Focused resampling depends only on Y and therefore preserves E[X|Y=y].",
        ],
    }
    with (DATA / "learned_run_metadata.json").open("w") as stream:
        json.dump(metadata, stream, indent=2)
    print(f"wrote learned-model figures to {FIG}")
    print(f"wrote learned-model results to {DATA}")


if __name__ == "__main__":
    main()
