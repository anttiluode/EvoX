from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

from evox.active import IdentificationResult, entropy_bits, run_identification
from evox.domains.causal_circuit import build_case as build_causal_case
from evox.domains.common import AdaptationResult, DomainCase
from evox.domains.dynamics import build_case as build_dynamics_case
from evox.domains.program_transform import build_case as build_program_transform_case
from evox.domains.regression import build_case as build_regression_case
from evox.domains.state_machine import build_case as build_state_machine_case


@dataclass(frozen=True)
class V3Config:
    confidence: float = 0.95
    pilot_seed_start: int = 700
    pilot_seed_count: int = 16
    canonical_seed_start: int = 800
    canonical_seed_count: int = 128


THRESHOLDS = {
    "probe_advantage": 0.40,
    "accuracy_advantage": 0.10,
    "reuse_success_slack": 0.05,
    "restart_cost_ratio": 0.75,
    "random_cost_ratio": 1.00,
    "oracle_success": 0.95,
}


def build_cases() -> tuple[DomainCase, ...]:
    return (
        build_regression_case(),
        build_state_machine_case(),
        build_program_transform_case(),
        build_dynamics_case(),
        build_causal_case(),
    )


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(repr(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _serialize_identification(
    result: IdentificationResult,
    truth: object,
    probe_budget: int,
) -> dict[str, Any]:
    censored = result.probes_used if result.reached_confidence else probe_budget + 1
    return {
        "map_hypothesis": _jsonable(result.map_hypothesis),
        "correct": result.map_hypothesis == truth,
        "posterior": _jsonable(result.posterior),
        "probes_used": result.probes_used,
        "censored_probes_to_confidence": censored,
        "reached_confidence": result.reached_confidence,
        "final_entropy_bits": entropy_bits(result.posterior),
        "steps": [
            {
                "intervention": _jsonable(step.intervention),
                "information_gain": step.information_gain,
                "observation": _jsonable(step.observation),
                "prior": _jsonable(step.prior),
                "posterior": _jsonable(step.posterior),
                "confidence": step.confidence,
            }
            for step in result.steps
        ],
    }


def _serialize_adaptation(result: AdaptationResult, budget: int) -> dict[str, Any]:
    return {
        "success": result.success,
        "executions": result.executions,
        "censored_executions": result.executions if result.success else budget + 1,
        "error": result.error,
        "detail": result.detail,
    }


def _observation_callback(case: DomainCase, truth: object, seed: int):
    def observe(intervention):
        rng = np.random.default_rng(
            _stable_seed("observation", case.name, seed, intervention)
        )
        return case.observe(truth, intervention, rng)

    return observe


def _run_trial(case: DomainCase, seed: int, config: V3Config) -> dict[str, Any]:
    truth = case.hypotheses[seed % len(case.hypotheses)]
    diagnostic_interventions = tuple(
        intervention
        for intervention in case.interventions
        if intervention != case.passive_intervention
    )
    observe = _observation_callback(case, truth, seed)

    active = run_identification(
        case.hypotheses,
        diagnostic_interventions,
        case.predict,
        observe,
        budget=case.probe_budget,
        confidence=config.confidence,
        policy="active",
        rng=np.random.default_rng(_stable_seed("active-policy", case.name, seed)),
    )
    random = run_identification(
        case.hypotheses,
        diagnostic_interventions,
        case.predict,
        observe,
        budget=case.probe_budget,
        confidence=config.confidence,
        policy="random",
        rng=np.random.default_rng(_stable_seed("random-policy", case.name, seed)),
    )

    adaptation_rng = lambda label: np.random.default_rng(
        _stable_seed("adaptation", case.name, seed, label)
    )
    active_reuse = case.adapt(
        active.map_hypothesis,
        truth,
        case.adapt_budget,
        adaptation_rng("active_reuse"),
    )
    random_reuse = case.adapt(
        random.map_hypothesis,
        truth,
        case.adapt_budget,
        adaptation_rng("random_reuse"),
    )
    restart = case.restart(
        truth,
        case.adapt_budget,
        adaptation_rng("restart"),
    )
    oracle = case.adapt(
        truth,
        truth,
        case.adapt_budget,
        adaptation_rng("oracle"),
    )

    return {
        "seed": seed,
        "truth": _jsonable(truth),
        "active": _serialize_identification(active, truth, case.probe_budget),
        "random": _serialize_identification(random, truth, case.probe_budget),
        "active_reuse": _serialize_adaptation(active_reuse, case.adapt_budget),
        "random_reuse": _serialize_adaptation(random_reuse, case.adapt_budget),
        "restart": _serialize_adaptation(restart, case.adapt_budget),
        "oracle": _serialize_adaptation(oracle, case.adapt_budget),
    }


def _summarize_identification(trials: list[dict[str, Any]], policy: str) -> dict[str, float]:
    rows = [trial[policy] for trial in trials]
    return {
        "accuracy": mean(float(bool(row["correct"])) for row in rows),
        "mean_censored_probes": mean(float(row["censored_probes_to_confidence"]) for row in rows),
        "mean_probes_used": mean(float(row["probes_used"]) for row in rows),
        "confidence_rate": mean(float(bool(row["reached_confidence"])) for row in rows),
        "mean_final_entropy_bits": mean(float(row["final_entropy_bits"]) for row in rows),
    }


def _summarize_adaptation(trials: list[dict[str, Any]], condition: str) -> dict[str, float]:
    rows = [trial[condition] for trial in trials]
    return {
        "success_rate": mean(float(bool(row["success"])) for row in rows),
        "mean_censored_executions": mean(float(row["censored_executions"]) for row in rows),
        "mean_executions": mean(float(row["executions"]) for row in rows),
    }


def _paired_identification(trials: list[dict[str, Any]]) -> dict[str, int]:
    wins = losses = ties = 0
    for trial in trials:
        active = trial["active"]
        random = trial["random"]
        active_key = (
            int(bool(active["correct"])),
            -int(active["censored_probes_to_confidence"]),
        )
        random_key = (
            int(bool(random["correct"])),
            -int(random["censored_probes_to_confidence"]),
        )
        if active_key > random_key:
            wins += 1
        elif active_key < random_key:
            losses += 1
        else:
            ties += 1
    return {"active_wins": wins, "active_losses": losses, "ties": ties}


def summarize_domain(trials: list[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        raise ValueError("domain summary requires at least one trial")
    return {
        "trial_count": len(trials),
        "identification": {
            "active": _summarize_identification(trials, "active"),
            "random": _summarize_identification(trials, "random"),
            "paired": _paired_identification(trials),
        },
        "adaptation": {
            condition: _summarize_adaptation(trials, condition)
            for condition in ("active_reuse", "random_reuse", "restart", "oracle")
        },
    }


def classify_domain(summary: dict[str, Any]) -> str:
    identification = summary["identification"]
    active = identification["active"]
    random = identification["random"]
    adaptation = summary["adaptation"]
    active_reuse = adaptation["active_reuse"]
    random_reuse = adaptation["random_reuse"]
    restart = adaptation["restart"]
    oracle = adaptation["oracle"]

    if active["accuracy"] + 1e-12 < random["accuracy"]:
        return "FAIL_IDENTIFICATION_ACCURACY"

    probe_advantage = random["mean_censored_probes"] - active["mean_censored_probes"]
    accuracy_advantage = active["accuracy"] - random["accuracy"]
    if (
        probe_advantage + 1e-12 < THRESHOLDS["probe_advantage"]
        and accuracy_advantage + 1e-12 < THRESHOLDS["accuracy_advantage"]
    ):
        return "FAIL_IDENTIFICATION_EFFICIENCY"

    if (
        active_reuse["success_rate"] + THRESHOLDS["reuse_success_slack"] + 1e-12
        < restart["success_rate"]
    ):
        return "FAIL_REUSE_SUCCESS"

    if (
        active_reuse["mean_censored_executions"]
        > THRESHOLDS["restart_cost_ratio"] * restart["mean_censored_executions"] + 1e-12
    ):
        return "FAIL_REUSE_VS_RESTART_COST"

    if (
        active_reuse["mean_censored_executions"]
        > THRESHOLDS["random_cost_ratio"] * random_reuse["mean_censored_executions"] + 1e-12
    ):
        return "FAIL_REUSE_VS_RANDOM_COST"

    if oracle["success_rate"] + 1e-12 < THRESHOLDS["oracle_success"]:
        return "FAIL_ORACLE"

    return "PASS"


def overall_pass(classifications: dict[str, str]) -> bool:
    return bool(classifications) and all(value == "PASS" for value in classifications.values())


def run_domain(case: DomainCase, seeds: Iterable[int], config: V3Config) -> dict[str, Any]:
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("seed tape must not be empty")
    trials = [_run_trial(case, seed, config) for seed in seed_list]
    summary = summarize_domain(trials)
    classification = classify_domain(summary)
    return {
        "name": case.name,
        "probe_budget": case.probe_budget,
        "adapt_budget": case.adapt_budget,
        "seeds": seed_list,
        "classification": classification,
        "summary": summary,
        "trials": trials,
    }


def run_sweep(
    *,
    seed_start: int,
    seed_count: int,
    config: V3Config | None = None,
    domain_names: set[str] | None = None,
) -> dict[str, Any]:
    config = V3Config() if config is None else config
    seeds = list(range(seed_start, seed_start + seed_count))
    cases = tuple(
        case for case in build_cases()
        if domain_names is None or case.name in domain_names
    )
    if not cases:
        raise ValueError("no domains selected")
    domains = {case.name: run_domain(case, seeds, config) for case in cases}
    classifications = {name: domain["classification"] for name, domain in domains.items()}
    return {
        "experiment": "EvoX V3 generic active mechanism selection and reuse",
        "config": asdict(config),
        "thresholds": THRESHOLDS,
        "seed_range": [seed_start, seed_start + seed_count - 1],
        "classifications": classifications,
        "overall_pass": overall_pass(classifications),
        "domains": domains,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/v3_active_mechanisms.json"))
    parser.add_argument("--seed-start", type=int, default=V3Config.canonical_seed_start)
    parser.add_argument("--seed-count", type=int, default=V3Config.canonical_seed_count)
    parser.add_argument(
        "--domains",
        type=str,
        default="",
        help="comma-separated domain names; empty means all domains",
    )
    args = parser.parse_args()
    selected = {item.strip() for item in args.domains.split(",") if item.strip()} or None
    receipt = run_sweep(
        seed_start=args.seed_start,
        seed_count=args.seed_count,
        domain_names=selected,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compact = {
        name: {
            "classification": domain["classification"],
            "summary": domain["summary"],
        }
        for name, domain in receipt["domains"].items()
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    print(json.dumps({"overall_pass": receipt["overall_pass"]}, indent=2))


if __name__ == "__main__":
    main()
