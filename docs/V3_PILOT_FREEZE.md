# EvoX V3 pilot freeze

This document freezes the V3 black-box worlds before canonical exposure.

## Pilot tape

Seeds `700–715` (16 trials per domain), with truth cycling evenly through all four hypotheses.

The generic active engine, success thresholds, confidence threshold (`0.95`), observation models, follow-up tasks, adaptation budgets, and canonical seed range were not tuned from the pilot.

## First pilot

Three domains passed immediately:

- `regression_fault`: PASS — active accuracy `1.000` vs random `0.875`; active censored probes `2.000` vs `2.3125`.
- `state_machine`: PASS — active accuracy `1.000` vs random `0.750`; active probes `1.000` vs `1.875`.
- `causal_circuit`: PASS — active accuracy `1.000` vs random `0.9375`; active probes `1.4375` vs `1.9375`.

Two domains exposed permitted world-design pathologies rather than engine failures:

1. `program_transform` was too easy: four of five diagnostic inputs uniquely separated all four hypotheses, so random also solved the task in nearly one probe (`1.125` mean censored probes).
2. `dynamics` had too little spread in intervention informativeness: active and random both frequently sampled highly informative controls, giving only `0.125` mean-probe advantage.

Both were classified `FAIL_IDENTIFICATION_EFFICIENCY` under the already frozen gate.

## Allowed pilot corrections

Only the diagnostic intervention libraries changed.

### Program transformation

The diagnostic tape is frozen as:

```python
(
    (0, 0, 0, 1),
    (0, 0, 0, 2),
    (0, 0, 1, 0),
    (0, 0, 2, 0),
    (0, 1, 2, 3),
)
```

The first four are balanced two-vs-two partitions in two redundant partition families. The last is a four-way separating input. Active information gain can identify the strongest experiment; random receives the same five interventions and two-probe budget.

No transformation hypothesis, passive example, follow-up payload, adaptation rule, budget, threshold, or engine code changed for this correction.

### Dynamics

The diagnostic control tape is frozen as:

```python
(
    (0.2,),
    (-0.2,),
    (0.2, 0.2),
    (1.0, -1.0),
    (1.0, 1.0),
    (0.0, 1.0),
)
```

This preserves the same four dynamical hypotheses and Gaussian observation noise `sigma=0.15` but creates a graded experiment set: small pulses, a small two-pulse experiment, and three stronger controls. Active and random again receive the identical set and two-probe budget.

No dynamics parameters, noise level, target, target tolerance, control library for adaptation, budget, threshold, or engine code changed for this correction.

## Final pilot

After those two corrections the same seeds `700–715` produced:

| domain | classification | active accuracy | random accuracy | active censored probes | random censored probes | active reuse cost | restart cost |
|---|---|---:|---:|---:|---:|---:|---:|
| regression fault | PASS | 1.000 | 0.875 | 2.0000 | 2.3125 | 1.000 | 2.500 |
| state machine | PASS | 1.000 | 0.750 | 1.0000 | 1.8750 | 1.000 | 3.250 |
| program transform | PASS | 1.000 | 0.875 | 1.0000 | 2.0000 | 1.000 | 2.500 |
| dynamics | PASS | 1.000 | 0.9375 | 1.6875 | 2.3125 | 1.000 | 9.750 censored |
| causal circuit | PASS | 1.000 | 0.9375 | 1.4375 | 1.9375 | 1.000 | 4.000 |

All oracle adaptation success rates were `1.000`.

## Canonical lock

Canonical seeds are `800–927` (128 trials per domain). After those seeds are observed, no scientific parameter, threshold, intervention set, noise level, hypothesis, follow-up task, or budget may change based on the canonical result.
