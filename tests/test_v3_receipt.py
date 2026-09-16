import numpy as np

from evox.domains.regression import build_case as build_regression_case
from experiments.run_v3_active_mechanisms import (
    V3Config,
    classify_domain,
    overall_pass,
    run_domain,
)


def _passing_summary():
    return {
        "identification": {
            "active": {
                "accuracy": 1.0,
                "mean_censored_probes": 1.0,
                "mean_final_entropy_bits": 0.0,
            },
            "random": {
                "accuracy": 0.8,
                "mean_censored_probes": 1.6,
                "mean_final_entropy_bits": 0.4,
            },
        },
        "adaptation": {
            "active_reuse": {"success_rate": 1.0, "mean_censored_executions": 1.5},
            "random_reuse": {"success_rate": 1.0, "mean_censored_executions": 2.0},
            "restart": {"success_rate": 1.0, "mean_censored_executions": 3.0},
            "oracle": {"success_rate": 1.0, "mean_censored_executions": 1.0},
        },
    }


def test_classifier_accepts_summary_that_clears_every_gate():
    assert classify_domain(_passing_summary()) == "PASS"


def test_each_frozen_gate_can_independently_fail_the_domain():
    cases = []

    summary = _passing_summary()
    summary["identification"]["active"]["accuracy"] = 0.7
    cases.append(summary)

    summary = _passing_summary()
    summary["identification"]["active"]["accuracy"] = 0.8
    summary["identification"]["random"]["accuracy"] = 0.8
    summary["identification"]["active"]["mean_censored_probes"] = 1.3
    summary["identification"]["random"]["mean_censored_probes"] = 1.6
    cases.append(summary)

    summary = _passing_summary()
    summary["adaptation"]["active_reuse"]["success_rate"] = 0.94
    summary["adaptation"]["restart"]["success_rate"] = 1.0
    cases.append(summary)

    summary = _passing_summary()
    summary["adaptation"]["active_reuse"]["mean_censored_executions"] = 2.3
    summary["adaptation"]["restart"]["mean_censored_executions"] = 3.0
    cases.append(summary)

    summary = _passing_summary()
    summary["adaptation"]["active_reuse"]["mean_censored_executions"] = 2.1
    summary["adaptation"]["random_reuse"]["mean_censored_executions"] = 2.0
    summary["adaptation"]["restart"]["mean_censored_executions"] = 3.0
    cases.append(summary)

    summary = _passing_summary()
    summary["adaptation"]["oracle"]["success_rate"] = 0.94
    cases.append(summary)

    assert all(classify_domain(summary) != "PASS" for summary in cases)


def test_overall_pass_requires_every_domain_to_pass():
    assert overall_pass({"a": "PASS", "b": "PASS"})
    assert not overall_pass({"a": "PASS", "b": "FAIL_IDENTIFICATION"})
    assert not overall_pass({})


def test_run_domain_enforces_matched_hard_budgets():
    case = build_regression_case()
    config = V3Config(confidence=0.95)
    receipt = run_domain(case, [700, 701, 702, 703], config)
    assert len(receipt["trials"]) == 4
    for trial in receipt["trials"]:
        assert trial["active"]["probes_used"] <= case.probe_budget
        assert trial["random"]["probes_used"] <= case.probe_budget
        for condition in ("active_reuse", "random_reuse", "restart", "oracle"):
            assert 1 <= trial[condition]["executions"] <= case.adapt_budget


def test_truth_cycle_is_balanced_over_four_consecutive_seeds():
    case = build_regression_case()
    receipt = run_domain(case, [800, 801, 802, 803], V3Config())
    truths = [trial["truth"] for trial in receipt["trials"]]
    assert sorted(truths) == sorted(case.hypotheses)


def test_run_domain_is_deterministic_for_same_seed_tape():
    case = build_regression_case()
    config = V3Config()
    a = run_domain(case, [812, 813], config)
    b = run_domain(case, [812, 813], config)
    assert a == b
