from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median

import numpy as np

from evox.evolution import Individual, evolve
from evox.modes import ModeModel, audit_reference_purity, extract_modes, mode_predictions
from evox.probes import choose_active_probe, entropy_bits, posterior_update
from evox.transfer import TransferResult, run_transfer_condition
from evox.worlds import counterfactual_pool, followup_examples, target_value, visible_examples


@dataclass(frozen=True)
class Config:
    population_size: int = 128
    generations: int = 16
    elite_count: int = 24
    max_depth: int = 3
    viable_mse: float = 1e-12
    mode_probe_count: int = 120
    max_probes: int = 3
    posterior_confidence: float = 0.95
    probe_sigma: float = 2.0
    transfer_population: int = 64
    transfer_generations: int = 8
    transfer_elites: int = 12
    canonical_seed_start: int = 100
    canonical_seed_count: int = 16


THRESHOLDS = {
    "valid_seed_fraction": 0.70,
    "mode_purity": 0.95,
    "active_probe_advantage": 0.50,
    "transfer_eval_ratio": 0.75,
    "transfer_success_slack": 0.05,
}


def _reference_cluster(mapping: dict[int, str] | dict[str, str], mode: str) -> int:
    for key, value in mapping.items():
        if value == mode:
            return int(key)
    raise ValueError(f"no cluster maps to {mode}")


def _run_probe_policy(
    model: ModeModel,
    hidden_mode: str,
    mapping: dict[int, str] | dict[str, str],
    candidates: list[tuple[int, int, int]],
    config: Config,
    *,
    active: bool,
    rng: np.random.Generator,
) -> dict[str, object]:
    prior = np.asarray([0.5, 0.5], dtype=float)
    used: set[tuple[int, int, int]] = set()
    trace: list[dict[str, object]] = []
    reached = False

    for _ in range(config.max_probes):
        if active:
            x, gain = choose_active_probe(model, candidates, prior, used=used, sigma=config.probe_sigma)
        else:
            available = [item for item in candidates if item not in used]
            x = available[int(rng.integers(len(available)))]
            gain = None
        used.add(x)
        predictions = mode_predictions(model, x)
        observation = target_value(hidden_mode, x)
        prior = posterior_update(prior, predictions, observation, config.probe_sigma)
        trace.append(
            {
                "x": list(x),
                "predictions": [float(v) for v in predictions],
                "observation": float(observation),
                "information_gain": None if gain is None else float(gain),
                "posterior": [float(v) for v in prior],
            }
        )
        if float(np.max(prior)) >= config.posterior_confidence:
            reached = True
            break

    chosen_cluster = int(np.argmax(prior))
    chosen_reference = mapping[chosen_cluster] if chosen_cluster in mapping else mapping[str(chosen_cluster)]
    return {
        "probes_used": len(trace),
        "censored_probes_to_confidence": len(trace) if reached else config.max_probes + 1,
        "confidence_reached": reached,
        "final_posterior": [float(v) for v in prior],
        "final_entropy_bits": entropy_bits(prior),
        "chosen_cluster": chosen_cluster,
        "chosen_reference_mode": chosen_reference,
        "identification_correct": chosen_reference == hidden_mode,
        "trace": trace,
    }


def _ancestry_transition_rate(model: ModeModel, viable: tuple[Individual, ...], ledger: tuple[Individual, ...], viable_mse: float) -> dict[str, object]:
    if not viable:
        return {"links": 0, "transition_rate": None}
    child_labels = {ind.id: int(label) for ind, label in zip(viable, model.labels)}
    by_id = {ind.id: ind for ind in ledger}
    centroids = np.stack([np.median(model.behavior[model.labels == k], axis=0) for k in (0, 1)])
    links = 0
    transitions = 0
    for child in viable:
        if child.parent_id is None:
            continue
        parent = by_id.get(child.parent_id)
        if parent is None or parent.mse > viable_mse:
            continue
        parent_behavior = np.asarray([float(__import__("evox.program", fromlist=["evaluate"]).evaluate(parent.tree, x)) for x in model.probes])
        distances = np.sum((centroids - parent_behavior[None, :]) ** 2, axis=1)
        parent_label = int(np.argmin(distances))
        links += 1
        transitions += int(parent_label != child_labels[child.id])
    return {
        "links": links,
        "transition_rate": None if links == 0 else transitions / links,
    }


def _transfer_dict(result: TransferResult) -> dict[str, object]:
    return {
        "condition": result.condition,
        "total_evaluations": result.total_evaluations,
        "evaluations_to_exact": result.evaluations_to_exact,
        "success": result.success,
        "best_mse": result.best_mse,
    }


def _run_hidden_trial(
    seed: int,
    hidden_mode: str,
    model: ModeModel,
    mapping: dict[int, str] | dict[str, str],
    config: Config,
) -> dict[str, object]:
    candidates = counterfactual_pool()
    random_rng = np.random.default_rng(seed * 1009 + (17 if hidden_mode == "direct" else 29))
    active = _run_probe_policy(model, hidden_mode, mapping, candidates, config, active=True, rng=random_rng)
    random_control = _run_probe_policy(model, hidden_mode, mapping, candidates, config, active=False, rng=random_rng)

    correct_cluster = _reference_cluster(mapping, hidden_mode)
    correct_programs = [p for p, label in zip(model.programs, model.labels) if int(label) == correct_cluster]
    wrong_programs = [p for p, label in zip(model.programs, model.labels) if int(label) != correct_cluster]

    scramble_rng = np.random.default_rng(seed * 2003 + (101 if hidden_mode == "direct" else 211))
    shuffled = model.labels.copy()
    scramble_rng.shuffle(shuffled)
    scrambled_programs = [p for p, label in zip(model.programs, shuffled) if int(label) == correct_cluster]

    transfer_seed = seed * 3001 + (313 if hidden_mode == "direct" else 617)
    common = dict(
        examples=followup_examples(hidden_mode),
        seed=transfer_seed,
        population_size=config.transfer_population,
        generations=config.transfer_generations,
        elite_count=config.transfer_elites,
        max_depth=config.max_depth,
        viable_mse=config.viable_mse,
    )
    transfer = {
        "correct": _transfer_dict(run_transfer_condition("correct", source_programs=correct_programs, **common)),
        "wrong": _transfer_dict(run_transfer_condition("wrong", source_programs=wrong_programs, **common)),
        "scrambled": _transfer_dict(run_transfer_condition("scrambled", source_programs=scrambled_programs, **common)),
        "restart": _transfer_dict(run_transfer_condition("restart", source_programs=None, **common)),
    }
    return {
        "hidden_mode": hidden_mode,
        "active": active,
        "random": random_control,
        "transfer": transfer,
    }


def run_seed(seed: int, config: Config) -> dict[str, object]:
    evolution = evolve(
        visible_examples(),
        seed=seed,
        population_size=config.population_size,
        generations=config.generations,
        elite_count=config.elite_count,
        max_depth=config.max_depth,
        viable_mse=config.viable_mse,
    )
    base: dict[str, object] = {
        "seed": seed,
        "evolution_evaluations": evolution.evaluations,
        "viable_count": len(evolution.viable),
    }
    if len(evolution.viable) < 4:
        base.update({"status": "INSUFFICIENT_DIVERSITY", "reason": "fewer than four viable programs", "trials": {}})
        return base

    probes = counterfactual_pool()[: config.mode_probe_count]
    try:
        model = extract_modes([ind.tree for ind in evolution.viable], probes)
    except ValueError as exc:
        base.update({"status": "INSUFFICIENT_DIVERSITY", "reason": str(exc), "trials": {}})
        return base

    audit = audit_reference_purity(model)
    purity = float(audit["purity"])
    mapping = audit["cluster_to_reference"]
    if purity < 0.90:
        base.update(
            {
                "status": "NO_MODE_SEPARATION",
                "reason": f"post-hoc purity {purity:.6f} < 0.90",
                "retained_rank": model.retained_rank,
                "mode_purity": purity,
                "trials": {},
            }
        )
        return base

    ancestry = _ancestry_transition_rate(model, evolution.viable, evolution.ledger, config.viable_mse)
    base.update(
        {
            "status": "OK",
            "retained_rank": model.retained_rank,
            "explained_variance": model.explained_variance,
            "singular_values": [float(v) for v in model.singular_values],
            "cluster_sizes": [int(np.sum(model.labels == k)) for k in (0, 1)],
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


def _transfer_summary(trials: list[dict[str, object]], condition: str) -> dict[str, float]:
    rows = [trial["transfer"][condition] for trial in trials]
    if not rows:
        return {"success_rate": 0.0, "mean_censored_evaluations": float("inf"), "median_censored_evaluations": float("inf"), "mean_best_mse": float("inf")}
    budget = int(rows[0]["total_evaluations"])
    censored = [int(row["evaluations_to_exact"]) if row["evaluations_to_exact"] is not None else budget + 1 for row in rows]
    return {
        "success_rate": mean(float(bool(row["success"])) for row in rows),
        "mean_censored_evaluations": mean(censored),
        "median_censored_evaluations": median(censored),
        "mean_best_mse": mean(float(row["best_mse"]) for row in rows),
    }


def summarize(seeds: list[dict[str, object]], config: Config) -> dict[str, object]:
    valid = [row for row in seeds if row["status"] == "OK"]
    trials = [trial for row in valid for trial in row["trials"].values()]
    summary: dict[str, object] = {
        "seed_count": len(seeds),
        "valid_seed_count": len(valid),
        "valid_seed_fraction": 0.0 if not seeds else len(valid) / len(seeds),
        "mean_viable_count": mean(int(row["viable_count"]) for row in seeds) if seeds else 0.0,
        "mean_mode_purity": mean(float(row["mode_purity"]) for row in valid) if valid else 0.0,
        "mean_retained_rank": mean(int(row["retained_rank"]) for row in valid) if valid else 0.0,
        "active_accuracy": mean(float(bool(t["active"]["identification_correct"])) for t in trials) if trials else 0.0,
        "random_accuracy": mean(float(bool(t["random"]["identification_correct"])) for t in trials) if trials else 0.0,
        "active_mean_censored_probes": mean(int(t["active"]["censored_probes_to_confidence"]) for t in trials) if trials else config.max_probes + 1,
        "random_mean_censored_probes": mean(int(t["random"]["censored_probes_to_confidence"]) for t in trials) if trials else config.max_probes + 1,
        "active_mean_final_entropy_bits": mean(float(t["active"]["final_entropy_bits"]) for t in trials) if trials else 1.0,
        "random_mean_final_entropy_bits": mean(float(t["random"]["final_entropy_bits"]) for t in trials) if trials else 1.0,
        "transfer": {condition: _transfer_summary(trials, condition) for condition in ("correct", "wrong", "scrambled", "restart")},
    }
    return summary


def classify(summary: dict[str, object]) -> str:
    if float(summary.get("valid_seed_fraction", 0.0)) < THRESHOLDS["valid_seed_fraction"]:
        return "INSUFFICIENT_DIVERSITY"
    if float(summary.get("mean_mode_purity", 0.0)) < THRESHOLDS["mode_purity"]:
        return "NO_MODE_SEPARATION"

    transfer = summary.get("transfer") or {}
    if not all(name in transfer for name in ("correct", "wrong", "scrambled", "restart")):
        return "MODES_ONLY"

    active_ok = (
        float(summary["active_accuracy"]) + 1e-12 >= float(summary["random_accuracy"])
        and float(summary["active_mean_censored_probes"])
        <= float(summary["random_mean_censored_probes"]) - THRESHOLDS["active_probe_advantage"]
    )

    correct = transfer["correct"]
    comparators = [transfer[name] for name in ("wrong", "scrambled", "restart")]
    success_ok = all(
        float(correct["success_rate"]) + THRESHOLDS["transfer_success_slack"] >= float(other["success_rate"])
        for other in comparators
    )
    eval_ok = all(
        float(correct["mean_censored_evaluations"])
        <= THRESHOLDS["transfer_eval_ratio"] * float(other["mean_censored_evaluations"])
        for other in comparators
    )
    return "PASS_ACTIVE_MODES" if active_ok and success_ok and eval_ok else "MODES_ONLY"


def run_canonical(config: Config) -> dict[str, object]:
    seeds = [run_seed(seed, config) for seed in range(config.canonical_seed_start, config.canonical_seed_start + config.canonical_seed_count)]
    summary = summarize(seeds, config)
    return {
        "experiment": "EvoX v0 active lineage modes",
        "config": asdict(config),
        "thresholds_frozen_before_canonical_run": THRESHOLDS,
        "pilot_seed_range_used_for_design": [0, 7],
        "canonical_seed_range": [config.canonical_seed_start, config.canonical_seed_start + config.canonical_seed_count - 1],
        "classification": classify(summary),
        "summary": summary,
        "seeds": seeds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/v0.json"))
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    args = parser.parse_args()
    config = Config(
        canonical_seed_count=Config.canonical_seed_count if args.seed_count is None else args.seed_count,
        canonical_seed_start=Config.canonical_seed_start if args.seed_start is None else args.seed_start,
    )
    receipt = run_canonical(config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt["summary"], indent=2, sort_keys=True))
    print("classification:", receipt["classification"])
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
