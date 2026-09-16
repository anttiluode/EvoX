from evox.domains import dynamics, program_transform
from experiments.run_v3_active_mechanisms import THRESHOLDS, V3Config


def test_v3_pilot_seed_ranges_and_thresholds_are_frozen():
    config = V3Config()
    assert (config.pilot_seed_start, config.pilot_seed_count) == (700, 16)
    assert (config.canonical_seed_start, config.canonical_seed_count) == (800, 128)
    assert config.confidence == 0.95
    assert THRESHOLDS == {
        "probe_advantage": 0.40,
        "accuracy_advantage": 0.10,
        "reuse_success_slack": 0.05,
        "restart_cost_ratio": 0.75,
        "random_cost_ratio": 1.00,
        "oracle_success": 0.95,
    }


def test_program_transform_pilot_interventions_are_frozen():
    assert program_transform.INTERVENTIONS == (
        program_transform.PASSIVE,
        (0, 0, 0, 1),
        (0, 0, 0, 2),
        (0, 0, 1, 0),
        (0, 0, 2, 0),
        (0, 1, 2, 3),
    )


def test_dynamics_pilot_interventions_and_noise_are_frozen():
    assert dynamics.INTERVENTIONS == (
        dynamics.PASSIVE,
        (0.2,),
        (-0.2,),
        (0.2, 0.2),
        (1.0, -1.0),
        (1.0, 1.0),
        (0.0, 1.0),
    )
    assert dynamics.OBSERVATION_SIGMA == 0.15
    assert dynamics.TARGET == 1.2
    assert dynamics.TARGET_TOLERANCE == 0.08
