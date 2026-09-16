import numpy as np

from evox.domains.program_transform import build_case as build_program_transform_case
from evox.domains.regression import build_case as build_regression_case
from evox.domains.state_machine import build_case as build_state_machine_case


def _categorical_signature(distribution):
    return tuple(distribution.quadrature())


def test_discrete_domains_are_passively_ambiguous_and_actively_informative():
    for build in (build_regression_case, build_state_machine_case, build_program_transform_case):
        case = build()
        assert len(case.hypotheses) == 4
        passive = {
            _categorical_signature(case.predict(hypothesis, case.passive_intervention))
            for hypothesis in case.hypotheses
        }
        assert len(passive) == 1, case.name

        informative = False
        for intervention in case.interventions:
            signatures = {
                _categorical_signature(case.predict(hypothesis, intervention))
                for hypothesis in case.hypotheses
            }
            informative |= len(signatures) > 1
        assert informative, case.name


def test_discrete_oracle_adaptation_is_solvable_inside_equal_budget():
    for build in (build_regression_case, build_state_machine_case, build_program_transform_case):
        case = build()
        for index, truth in enumerate(case.hypotheses):
            oracle = case.adapt(
                truth,
                truth,
                case.adapt_budget,
                np.random.default_rng(100 + index),
            )
            restart = case.restart(
                truth,
                case.adapt_budget,
                np.random.default_rng(200 + index),
            )
            assert oracle.success, (case.name, truth)
            assert 1 <= oracle.executions <= case.adapt_budget
            assert 1 <= restart.executions <= case.adapt_budget


def test_discrete_observation_matches_predictive_support():
    for build in (build_regression_case, build_state_machine_case, build_program_transform_case):
        case = build()
        for h_index, truth in enumerate(case.hypotheses):
            rng = np.random.default_rng(300 + h_index)
            for intervention in case.interventions:
                observation = case.observe(truth, intervention, rng)
                support = {outcome for outcome, weight in case.predict(truth, intervention).quadrature() if weight > 0}
                assert observation in support, (case.name, truth, intervention, observation)
