from experiments.run_v2_behavioral import V2Config, classify_generic, run_strategy_seed


def test_v2_smoke_has_equal_labeled_budget_and_explicit_descriptor_cost():
    config = V2Config(
        population_size=32,
        generations=4,
        elite_count=8,
        descriptor_length=12,
        canonical_seed_count=1,
    )
    rows = [
        run_strategy_seed(strategy, 701, config)
        for strategy in (
            "baseline",
            "structural_novelty",
            "behavioral_generic",
            "behavioral_manifold",
        )
    ]
    assert {row["evolution_evaluations"] for row in rows} == {32 * 4}
    by_strategy = {row["strategy"]: row for row in rows}
    assert by_strategy["baseline"]["descriptor_executions"] == 0
    assert by_strategy["structural_novelty"]["descriptor_executions"] == 0
    assert by_strategy["behavioral_generic"]["descriptor_executions"] == 32 * 12 * 3
    assert by_strategy["behavioral_manifold"]["descriptor_executions"] == 32 * 12 * 3


def _good_summary(*, valid=0.9, mass=0.2):
    return {
        "valid_seed_fraction": valid,
        "mean_minority_mode_fraction": mass,
        "mean_mode_purity": 1.0,
        "median_best_visible_mse": 0.0,
        "active_accuracy": 1.0,
        "random_accuracy": 1.0,
        "active_mean_censored_probes": 1.0,
        "random_mean_censored_probes": 2.0,
        "transfer": {
            "correct": {"mean_censored_evaluations": 20.0, "success_rate": 1.0},
            "restart": {"mean_censored_evaluations": 100.0, "success_rate": 0.8},
        },
    }


def test_v2_classification_requires_generic_advantage_over_manifold_control():
    baseline = {"median_best_visible_mse": 0.0}
    generic = _good_summary(valid=0.9, mass=0.2)
    weak_control = _good_summary(valid=0.7, mass=0.08)
    assert classify_generic(generic, baseline, weak_control) == "PRESERVES_FUNCTIONAL_MODES"

    equally_good_control = _good_summary(valid=0.88, mass=0.18)
    assert classify_generic(generic, baseline, equally_good_control) == "LEAK_OR_CONTROL_FAILURE"


def test_v2_classification_keeps_thin_result_distinct():
    baseline = {"median_best_visible_mse": 0.0}
    generic = _good_summary(valid=0.9, mass=0.1)
    control = _good_summary(valid=0.7, mass=0.02)
    assert classify_generic(generic, baseline, control) == "FUNCTIONAL_BUT_THIN"
