from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median

import numpy as np

from evox.descriptors import behavior_matrix, generic_unlabeled_tape, in_manifold_unlabeled_tape
from evox.evolution import evolve
from evox.modes import audit_reference_purity, extract_modes
from evox.worlds import counterfactual_pool, visible_examples
from experiments.run_v0 import Config as V0Config
from experiments.run_v0 import _ancestry_transition_rate, _run_hidden_trial, _transfer_summary

STRATEGIES = (
    "baseline",
    "structural_novelty",
    "behavioral_generic",
    "behavioral_manifold",
)


@dataclass(frozen=True)
class V2Config(V0Config):
    population_size: int = 128
    generations: int = 16
    elite_count: int = 24
    canonical_seed_start: int = 600
    canonical_seed_count: int = 32
    novelty_fraction: float = 0.75
    descriptor_length: int = 32
    generic_descriptor_seed: int = 271828
    manifold_descriptor_seed: int = 314159


THRESHOLDS = {
    "valid_seed_fraction": 0.85,
    "minority_mode_fraction": 0.15,
    "mode_purity": 0.95,
    "visible_mse_slack": 1e-12,
    "active_probe_advantage": 0.50,
    "transfer_eval_ratio": 0.75,
    "transfer_success_slack": 0.05,
    "control_valid_advantage": 0.10,
    "control_mass_advantage": 0.05,
}


def _descriptor_tape(strategy: str, config: V2Config):
    if strategy == "behavioral_manifold":
        return in_manifold_unlabeled_tape(
            seed=config.manifold_descriptor_seed, length=config.descriptor_length
        )
    return generic_unlabeled_tape(
        seed=config.generic_descriptor_seed, length=config.descriptor_length
    )


def _descriptor_audit(programs, tape) -> tuple[float, float]:
    if len(programs) < 2:
        return 0.0, 0.0
    matrix = behavior_matrix(programs, tape)
    distances = []
    for i in range(len(programs)):
        for j in range(i + 1, len(programs)):
            distances.append(float(np.linalg.norm(matrix[i] - matrix[j])))
    nonzero_fraction = float(np.mean(np.std(matrix, axis=0) > 0.0)) if matrix.shape[1] else 0.0
    return (mean(distances) if distances else 0.0, nonzero_fraction)


def run_strategy_seed(strategy: str, seed: int, config: V2Config) -> dict[str, object]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")

    tape = _descriptor_tape(strategy, config)
    if strategy == "baseline":
        evolution_strategy = "baseline"
        descriptor_inputs = None
    elif strategy == "structural_novelty":
        evolution_strategy = "structural_novelty"
        descriptor_inputs = None
    else:
        evolution_strategy = "behavioral_novelty"
        descriptor_inputs = tape

    evolution = evolve(
        visible_examples(),
        seed=seed,
        population_size=config.population_size,
        generations=config.generations,
        elite_count=config.elite_count,
        max_depth=config.max_depth,
        viable_mse=config.viable_mse,
        strategy=evolution_strategy,
        novelty_fraction=config.novelty_fraction,
        descriptor_inputs=descriptor_inputs,
    )

    final_mses = [float(ind.mse) for ind in evolution.final_population]
    audit_distance, audit_nonzero = _descriptor_audit(
        [ind.tree for ind in evolution.viable], tape
    )
    base: dict[str, object] = {
        "strategy": strategy,
        "seed": seed,
        "evolution_evaluations": evolution.evaluations,
        "descriptor_executions": evolution.descriptor_executions,
        "descriptor_tape_kind": "manifold" if strategy == "behavioral_manifold" else "generic",
        "viable_count": len(evolution.viable),
        "best_visible_mse": min(final_mses),
        "median_visible_mse": median(final_mses),
        "mean_pairwise_descriptor_distance": audit_distance,
        "descriptor_nonzero_coordinate_fraction": audit_nonzero,
    }
    if len(evolution.viable) < 4:
        base.update(
            {
                "status": "INSUFFICIENT_DIVERSITY",
                "reason": "fewer than four viable programs",
                "trials": {},
            }
        )
        return base

    probes = counterfactual_pool()[: config.mode_probe_count]
    try:
        model = extract_modes([ind.tree for ind in evolution.viable], probes)
    except ValueError as exc:
        base.update(
            {
                "status": "INSUFFICIENT_DIVERSITY",
                "reason": str(exc),
                "trials": {},
            }
        )
        return base

    audit = audit_reference_purity(model)
    purity = float(audit["purity"])
    mapping = audit["cluster_to_reference"]
    cluster_sizes = [int(np.sum(model.labels == k)) for k in (0, 1)]
    minority_fraction = min(cluster_sizes) / sum(cluster_sizes)
    if purity < 0.90:
        base.update(
            {
                "status": "NO_MODE_SEPARATION",
                "reason": f"post-hoc purity {purity:.6f} < 0.90",
                "retained_rank": model.retained_rank,
                "cluster_sizes": cluster_sizes,
                "minority_mode_fraction": minority_fraction,
                "mode_purity": purity,
                "trials": {},
            }
        )
        return base

    ancestry = _ancestry_transition_rate(
        model, evolution.viable, evolution.ledger, config.viable_mse
    )
    base.update(
        {
            "status": "OK",
            "retained_rank": model.retained_rank,
            "explained_variance": model.explained_variance,
            "cluster_sizes": cluster_sizes,
            "minority_mode_fraction": minority_fraction,
            "mode_purity": purity,
            "cluster_to_reference": {str(k): v for k, v in mapping.items()},
            "ancestry": ancestry,
            "trials": {
                mode: _run_hidden_trial(seed, mode, model, mapping, config)
                for mode in ("direct", "relational")
            },
        }
    )
    return base


def summarize_strategy(rows: list[dict[str, object]], config: V2Config) -> dict[str, object]:
    valid = [row for row in rows if row["status"] == "OK"]
    trials = [trial for row in valid for trial in row["trials"].values()]
    return {
        "seed_count": len(rows),
        "valid_seed_count": len(valid),
        "valid_seed_fraction": 0.0 if not rows else len(valid) / len(rows),
        "mean_viable_count": mean(int(row["viable_count"]) for row in rows) if rows else 0.0,
        "mean_minority_mode_fraction": mean(float(row["minority_mode_fraction"]) for row in valid) if valid else 0.0,
        "median_minority_mode_fraction": median(float(row["minority_mode_fraction"]) for row in valid) if valid else 0.0,
        "mean_mode_purity": mean(float(row["mode_purity"]) for row in valid) if valid else 0.0,
        "median_best_visible_mse": median(float(row["best_visible_mse"]) for row in rows) if rows else float("inf"),
        "median_visible_mse": median(float(row["median_visible_mse"]) for row in rows) if rows else float("inf"),
        "mean_descriptor_executions": mean(int(row["descriptor_executions"]) for row in rows) if rows else 0.0,
        "mean_pairwise_descriptor_distance": mean(float(row["mean_pairwise_descriptor_distance"]) for row in rows) if rows else 0.0,
        "mean_descriptor_nonzero_coordinate_fraction": mean(float(row["descriptor_nonzero_coordinate_fraction"]) for row in rows) if rows else 0.0,
        "active_accuracy": mean(float(bool(t["active"]["identification_correct"])) for t in trials) if trials else 0.0,
        "random_accuracy": mean(float(bool(t["random"]["identification_correct"])) for t in trials) if trials else 0.0,
        "active_mean_censored_probes": mean(int(t["active"]["censored_probes_to_confidence"]) for t in trials) if trials else config.max_probes + 1,
        "random_mean_censored_probes": mean(int(t["random"]["censored_probes_to_confidence"]) for t in trials) if trials else config.max_probes + 1,
        "transfer": {
            condition: _transfer_summary(trials, condition)
            for condition in ("correct", "wrong", "scrambled", "restart")
        },
    }


def _core_conditions(summary: dict[str, object], baseline: dict[str, object]) -> tuple[bool, bool, bool, bool, bool, bool]:
    valid_ok = float(summary.get("valid_seed_fraction", 0.0)) >= THRESHOLDS["valid_seed_fraction"]
    mass_ok = float(summary.get("mean_minority_mode_fraction", 0.0)) >= THRESHOLDS["minority_mode_fraction"]
    purity_ok = float(summary.get("mean_mode_purity", 0.0)) >= THRESHOLDS["mode_purity"]
    fitness_ok = float(summary.get("median_best_visible_mse", float("inf"))) <= (
        float(baseline.get("median_best_visible_mse", float("inf"))) + THRESHOLDS["visible_mse_slack"]
    )
    active_ok = (
        float(summary.get("active_accuracy", 0.0)) + 1e-12 >= float(summary.get("random_accuracy", 0.0))
        and float(summary.get("active_mean_censored_probes", float("inf")))
        <= float(summary.get("random_mean_censored_probes", float("inf"))) - THRESHOLDS["active_probe_advantage"]
    )
    transfer = summary.get("transfer") or {}
    correct = transfer.get("correct") if isinstance(transfer, dict) else None
    restart = transfer.get("restart") if isinstance(transfer, dict) else None
    transfer_ok = False
    if correct and restart:
        transfer_ok = (
            float(correct["success_rate"]) + THRESHOLDS["transfer_success_slack"] >= float(restart["success_rate"])
            and float(correct["mean_censored_evaluations"])
            <= THRESHOLDS["transfer_eval_ratio"] * float(restart["mean_censored_evaluations"])
        )
    return valid_ok, mass_ok, purity_ok, fitness_ok, active_ok, transfer_ok


def classify_generic(
    generic: dict[str, object],
    baseline: dict[str, object],
    manifold_control: dict[str, object],
) -> str:
    valid_ok, mass_ok, purity_ok, fitness_ok, active_ok, transfer_ok = _core_conditions(
        generic, baseline
    )
    control_advantage = (
        float(generic.get("valid_seed_fraction", 0.0))
        - float(manifold_control.get("valid_seed_fraction", 0.0))
        >= THRESHOLDS["control_valid_advantage"]
        or float(generic.get("mean_minority_mode_fraction", 0.0))
        - float(manifold_control.get("mean_minority_mode_fraction", 0.0))
        >= THRESHOLDS["control_mass_advantage"]
    )
    essentials = valid_ok and purity_ok and fitness_ok and active_ok and transfer_ok
    if essentials and not control_advantage:
        return "LEAK_OR_CONTROL_FAILURE"
    if essentials and control_advantage:
        return "PRESERVES_FUNCTIONAL_MODES" if mass_ok else "FUNCTIONAL_BUT_THIN"
    return "NO_FUNCTIONAL_PRESERVATION_GAIN"


def run_sweep(
    config: V2Config,
    seed_start: int,
    seed_count: int,
    strategies: tuple[str, ...] = STRATEGIES,
) -> dict[str, object]:
    seeds = list(range(seed_start, seed_start + seed_count))
    per_strategy = {
        strategy: [run_strategy_seed(strategy, seed, config) for seed in seeds]
        for strategy in strategies
    }
    summaries = {
        strategy: summarize_strategy(rows, config) for strategy, rows in per_strategy.items()
    }
    classifications: dict[str, str] = {}
    if "baseline" in summaries:
        classifications["baseline"] = "BASELINE"
    if "structural_novelty" in summaries:
        classifications["structural_novelty"] = "V1_CONTROL"
    if "behavioral_manifold" in summaries:
        classifications["behavioral_manifold"] = "MANIFOLD_CONTROL"
    if all(name in summaries for name in ("baseline", "behavioral_generic", "behavioral_manifold")):
        classifications["behavioral_generic"] = classify_generic(
            summaries["behavioral_generic"],
            summaries["baseline"],
            summaries["behavioral_manifold"],
        )
    return {
        "config": asdict(config),
        "seed_range": [seed_start, seed_start + seed_count - 1],
        "thresholds": THRESHOLDS,
        "classifications": classifications,
        "summaries": summaries,
        "strategies": per_strategy,
    }


def run_canonical(config: V2Config) -> dict[str, object]:
    result = run_sweep(config, config.canonical_seed_start, config.canonical_seed_count)
    result.update(
        {
            "experiment": "EvoX v2 unlabeled behavioral novelty",
            "pilot_seed_range": [500, 507],
            "canonical_seed_range": [
                config.canonical_seed_start,
                config.canonical_seed_start + config.canonical_seed_count - 1,
            ],
            "overall_positive": result["classifications"].get("behavioral_generic")
            == "PRESERVES_FUNCTIONAL_MODES",
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/v2_behavioral.json"))
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument(
        "--strategies",
        type=str,
        default=",".join(STRATEGIES),
        help="comma-separated subset of strategies",
    )
    args = parser.parse_args()
    config = V2Config(
        canonical_seed_start=V2Config.canonical_seed_start if args.seed_start is None else args.seed_start,
        canonical_seed_count=V2Config.canonical_seed_count if args.seed_count is None else args.seed_count,
    )
    strategies = tuple(part.strip() for part in args.strategies.split(",") if part.strip())
    unknown = set(strategies) - set(STRATEGIES)
    if unknown:
        raise ValueError(f"unknown strategies: {sorted(unknown)}")
    receipt = run_sweep(config, config.canonical_seed_start, config.canonical_seed_count, strategies)
    receipt.update(
        {
            "experiment": "EvoX v2 unlabeled behavioral novelty",
            "pilot_seed_range": [500, 507],
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt["summaries"], indent=2, sort_keys=True))
    print(json.dumps(receipt["classifications"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
