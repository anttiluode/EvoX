from experiments.run_v0 import Config, classify, run_seed


def test_v0_seed_receipt_has_modes_probes_and_four_transfer_controls():
    config = Config(
        population_size=96,
        generations=12,
        elite_count=18,
        mode_probe_count=80,
        transfer_population=48,
        transfer_generations=6,
        transfer_elites=10,
    )
    receipt = run_seed(0, config)
    assert receipt["seed"] == 0
    assert receipt["viable_count"] >= 0
    assert receipt["status"] in {"OK", "INSUFFICIENT_DIVERSITY", "NO_MODE_SEPARATION"}
    if receipt["status"] == "OK":
        assert receipt["retained_rank"] >= 1
        assert 0.0 <= receipt["mode_purity"] <= 1.0
        assert set(receipt["trials"]) == {"direct", "relational"}
        for trial in receipt["trials"].values():
            assert "active" in trial and "random" in trial
            assert set(trial["transfer"]) == {"correct", "wrong", "scrambled", "restart"}
            totals = {item["total_evaluations"] for item in trial["transfer"].values()}
            assert len(totals) == 1


def test_classification_strings_are_declared():
    summary = {
        "valid_seed_fraction": 0.0,
        "mean_mode_purity": 0.0,
        "active_accuracy": 0.0,
        "random_accuracy": 0.0,
        "active_mean_censored_probes": 4.0,
        "random_mean_censored_probes": 4.0,
        "transfer": {},
    }
    assert classify(summary) == "INSUFFICIENT_DIVERSITY"
