#!/usr/bin/env python3
"""Integrity checks on the manuscript source and the recorded results."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_latex_links() -> None:
    sources = "\n".join(
        (ROOT / name).read_text(encoding="utf-8")
        for name in ("main.tex", "appendix.tex")
    )
    labels = re.findall(r"\\label\{([^}]+)\}", sources)
    require(len(labels) == len(set(labels)), "duplicate LaTeX labels found")
    missing_labels = sorted(set(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", sources))
                            - set(labels))
    require(not missing_labels, "undefined LaTeX references: " + ", ".join(missing_labels))

    bibliography = (ROOT / "references.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@[A-Za-z]+\s*\{\s*([^,\s]+)", bibliography))
    citation_groups = re.findall(r"\\cite[a-zA-Z]*\{([^}]+)\}", sources)
    cited = {key.strip() for group in citation_groups for key in group.split(",")}
    missing_citations = sorted(cited - bib_keys)
    require(not missing_citations,
            "citation keys absent from references.bib: " + ", ".join(missing_citations))

    graphics = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", sources)
    missing_graphics = []
    for graphic in graphics:
        candidate = ROOT / graphic
        if candidate.suffix:
            exists = candidate.is_file()
        else:
            exists = any(candidate.with_suffix(suffix).is_file()
                         for suffix in (".pdf", ".png", ".jpg"))
        if not exists:
            missing_graphics.append(graphic)
    require(not missing_graphics,
            "included graphics are missing: " + ", ".join(sorted(missing_graphics)))


def read_csv(name: str) -> list[dict[str, str]]:
    path = ROOT / "results" / name
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(bool(rows), f"{path.relative_to(ROOT)} has no data rows")
    require(None not in rows[0], f"{path.relative_to(ROOT)} has malformed columns")
    return rows


def check_recorded_evidence() -> None:
    kde = read_csv("finite_sample_kde.csv")
    require(len(kde) == 48 * 3 * 5,
            f"finite_sample_kde.csv has {len(kde)} rows; expected 720")
    require(len({row["seed"] for row in kde}) == 48,
            "finite_sample_kde.csv does not contain 48 independent seeds")

    learned = read_csv("learned_score_metrics.csv")
    require(len({row["seed"] for row in learned}) == 12,
            "learned_score_metrics.csv does not contain 12 trained seeds")
    problems = {row["problem"] for row in learned}
    require(problems == {"circle_regular", "circle_zero", "plane_zero"},
            f"unexpected learned-score problem set: {sorted(problems)}")

    paired = read_csv("learned_paired_bootstrap.csv")
    require(len(paired) == 2, "learned_paired_bootstrap.csv must contain two contrasts")
    require(all(int(float(row["replications"])) == 12 for row in paired),
            "paired bootstrap contrasts must use all 12 paired runs")

    optimization = read_csv("learned_optimization.csv")
    methods = {row["method"] for row in optimization}
    require(methods == {"oracle", "raw", "inner_only", "calibrated"},
            f"unexpected optimizer method set: {sorted(methods)}")

    metadata = json.loads((ROOT / "results" / "learned_run_metadata.json").read_text())
    require(metadata["configuration"]["seeds"] == 12,
            "learned metadata does not describe the reported 12-seed run")
    require(metadata["configuration"]["plane_dim"] == 16,
            "learned metadata has an unexpected ambient dimension")
    require(float(metadata["total_wall_seconds"]) > 0,
            "learned metadata has no positive wall-clock measurement")

    numerical = json.loads(
        (ROOT / "results" / "learned_numerical_checks.json").read_text()
    )
    require(numerical["jacobian"]["operator_norm_error"] < 1e-8,
            "analytic denoiser Jacobian failed its finite-difference audit")
    require(numerical["plane_sampler"]["sample_size"] == 4_000_000,
            "plane sampler audit has an unexpected sample size")
    require(numerical["plane_sampler"]["max_relative_error"] < 1e-3,
            "plane sampler moment audit failed")

    for name in (
        "circle_spectrum.csv", "conditioning.csv", "conditioning_stability_width.csv",
        "crossover.csv", "finite_sample_kde_summary.csv", "landing_error.csv",
        "learned_optimization_summary.csv", "learned_score_summary.csv",
        "normal_offset_audit.csv", "spatial_boundary_layer.csv", "spiral.csv",
        "tilted_posterior.csv",
    ):
        read_csv(name)


def check_figures() -> None:
    figures = sorted((ROOT / "figures").glob("*.pdf"))
    require(len(figures) == 11, f"expected 11 figures, found {len(figures)}")
    for path in figures:
        data = path.read_bytes()
        require(data.startswith(b"%PDF-"), f"{path.relative_to(ROOT)} is not a PDF")
        require(len(data) > 1_000, f"{path.relative_to(ROOT)} is unexpectedly small")


def main() -> None:
    check_latex_links()
    check_recorded_evidence()
    check_figures()
    print("reproducibility checks passed")


if __name__ == "__main__":
    main()
