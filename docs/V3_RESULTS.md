# EvoX V3 — generic active mechanism selection and reuse

V3 extracted the useful half of EvoX out of the original direct-vs-relational toy world.

The question was:

> Given several executable explanations that fit what has been seen so far, what intervention should be performed next to identify the real mechanism, and can that identification be exploited immediately?

## Generic interface

The inference engine has no knowledge of programs, state machines, regressions, dynamics, or circuits. It receives:

```text
candidate hypotheses
allowed interventions
predictive observation distributions
experiment budget
observation callback
```

A domain adapter separately supplies its adaptation/restart operation. The same `run_identification(...)` implementation is used by every world.

The finite-hypothesis engine supports arbitrary hypothesis counts and both categorical and Gaussian predictive distributions. Active intervention choice maximizes expected information gain. Random probing receives the same intervention set and probe budget.

## Domains

The frozen canonical experiment used five adapters:

- `regression_fault` — four candidate fault mechanisms;
- `state_machine` — four hidden transition mechanisms;
- `program_transform` — four executable transformation rules;
- `dynamics` — four noisy continuous dynamical mechanisms;
- `causal_circuit` — four causal circuit mechanisms.

Each adapter has four hypotheses. Truth cycles evenly through them on the seed tape. No domain-specific inference logic is permitted in the generic engine.

## Frozen gate

Pilot seeds were `700–715`. Only two pilot pathologies were corrected before canonical exposure: the diagnostic intervention libraries for program transformation and dynamics. Those corrections and the final pilot were frozen in `docs/V3_PILOT_FREEZE.md`.

Canonical seeds were `800–927`, 128 trials per domain.

A domain passes only if all of its identification/reuse conditions pass. The identification-efficiency condition requires active probing to be no less accurate than random and to achieve at least one of:

```text
random censored probes - active censored probes >= 0.40
active accuracy - random accuracy >= 0.10
```

Reuse must also retain success relative to restart, cost at most 75% of restart's censored execution cost, cost no more than random reuse, and the oracle condition must succeed at least 95% of the time.

**Overall V3 passes only if every domain passes. There is no cross-domain averaging escape hatch.**

## Canonical result

| domain | classification | active accuracy | random accuracy | active censored probes | random censored probes | active reuse cost | restart cost |
|---|---|---:|---:|---:|---:|---:|---:|
| regression fault | `PASS` | **1.0000** | 0.7500 | **2.0000** | 2.3672 | **1.0000** | 2.5000 |
| state machine | `PASS` | **1.0000** | 0.8516 | **1.0000** | 1.7969 | **1.0000** | 3.2500 |
| program transform | `PASS` | **1.0000** | 0.8906 | **1.0000** | 2.0625 | **1.0000** | 2.5000 |
| dynamics | `PASS` | **0.9688** | 0.8516 | **1.3516** | 2.0234 | **1.0000** | 9.7500 censored |
| causal circuit | `FAIL_IDENTIFICATION_EFFICIENCY` | **0.9922** | 0.9609 | **1.3047** | 1.6484 | **1.0312** | 4.0000 |

The frozen overall result is therefore:

```text
overall_pass = false
4 / 5 domains PASS
```

### Why the causal-circuit adapter failed

The result is not an identification collapse. Active probing is better than random on every headline metric:

```text
accuracy advantage = 0.9921875 - 0.9609375 = 0.03125
probe advantage    = 1.6484375 - 1.3046875 = 0.34375
```

But the frozen gate requires `>= 0.10` accuracy advantage or `>= 0.40` probe advantage. It reaches neither. The paired result is 53 active wins, 17 active losses, and 58 ties.

Reuse still works in that world: active identification followed by reuse costs 1.03125 executions on average versus 4.0 for restart, with 100% success for both. The failure is specifically that the intervention library does not give active design a large enough identification-efficiency advantage over random under the predeclared threshold.

No threshold, circuit, intervention, noise model, budget, or classifier was changed after canonical exposure.

## What generalized

The important positive result is architectural rather than universal:

```text
candidate executable mechanisms
          ↓
shared Bayesian posterior
          ↓
expected-information intervention choice
          ↓
real observation
          ↓
updated mechanism belief
          ↓
reuse/adaptation from selected mechanism
```

The exact same engine produced canonical PASS results in discrete state transitions, executable program transformations, fault localization, and noisy continuous dynamics. In those four worlds, active identification was at least as accurate as random, materially more probe-efficient or accurate under the frozen gate, and the selected mechanism reduced adaptation cost versus restart.

The strongest adaptation contrast is the dynamics world: active reuse succeeds on 100% of trials at 1.0 execution, while restart succeeds on only 50% within budget and has a 9.75 censored mean execution cost.

## What this does not establish

V3 does not establish a universal active-diagnosis algorithm. In particular:

- one of five canonical worlds misses the frozen identification-efficiency criterion;
- all five worlds are small controlled black boxes with a finite candidate hypothesis set;
- the true mechanism is guaranteed to be represented among the hypotheses;
- interventions are supplied by the adapter rather than invented by the engine;
- adaptation operations are domain-specific and deliberately small;
- this is not yet a real software-regression, physical-system, or large program-synthesis benchmark.

## Interpretation

The preservation problem from V0–V2 and the active-identification problem should now be treated separately.

V0–V2 showed that maintaining alternative evolved computations at substantial population mass is hard. V3 shows that **once a finite set of plausible executable mechanisms is available, a common active-experiment engine can identify and exploit them across substantially different kinds of black boxes**—but not with a guaranteed efficiency margin in every intervention library.

The next useful step is not another synthetic novelty mechanism. It is to attach this engine to a real source of competing executable explanations, where hypotheses and interventions have operational meaning.

Frozen receipt: [`results/v3_active_mechanisms.json`](../results/v3_active_mechanisms.json)

Canonical Actions run: `35088799353`

Exact receipt SHA-256:

```text
129d57b74d7484f11e2c487f8376ec4f87c8c63a06531637b6347241b75545c2
```
