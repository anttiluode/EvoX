import numpy as np

from evox.active import GaussianDistribution
from evox.domains.causal_circuit import build_case as build_causal_case
from evox.domains.dynamics import build_case as build_dynamics_case


def test_continuous_domains_are_passively_ambiguous_and_have_informative_interventions():
    for build in (build_dynamics_case, build_causal_case):
        case = build()
        assert len(case.hypotheses) == 4
        passive = [case.predict(h, case.passive_intervention) for h in case.hypotheses]
        assert all(isinstance(dist, GaussianDistribution) for dist in passive)
        assert max(dist.mean for dist in passive) - min(dist.mean for dist in passive) <= 1e-12
        assert max(dist.sigma for dist in passive) - min(dist.sigma for dist in passive) <= 1e-12

        assert any(
            max(case.predict(h, intervention).mean for h in case.hypotheses)
            - min(case.predict(h, intervention).mean for h in case.hypotheses)
            > 0.25
            for intervention in case.interventions
        ), case.name


def test_continuous_observations_are_numeric_and_oracle_adaptation_is_solvable():
    for build in (build_dynamics_case, build_causal_case):
        case = build()
        for index, truth in enumerate(case.hypotheses):
            rng = np.random.default_rng(400 + index)
            observation = case.observe(truth, case.interventions[-1], rng)
            assert np.isfinite(float(observation))

            oracle = case.adapt(
                truth,
                truth,
                case.adapt_budget,
                np.random.default_rng(500 + index),
            )
            restart = case.restart(
                truth,
                case.adapt_budget,
                np.random.default_rng(600 + index),
            )
            assert oracle.success, (case.name, truth)
            assert 1 <= oracle.executions <= case.adapt_budget
            assert 1 <= restart.executions <= case.adapt_budget


def test_continuous_predictions_use_same_noise_model_as_observer_contract():
    for build in (build_dynamics_case, build_causal_case):
        case = build()
        for truth in case.hypotheses:
            for intervention in case.interventions:
                dist = case.predict(truth, intervention)
                assert isinstance(dist, GaussianDistribution)
                assert dist.sigma > 0.0
