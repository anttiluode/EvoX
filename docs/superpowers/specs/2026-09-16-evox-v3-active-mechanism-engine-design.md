# EvoX V3 — Generic Active Mechanism Engine

## Status

Frozen design for implementation. This branch intentionally leaves the V0–V2 preservation problem behind and extracts the part that repeatedly worked: active discrimination between executable hypotheses followed by reuse of the identified mechanism.

## Question

Given several executable hypotheses that are all consistent with what has been observed so far:

1. can one domain-agnostic engine choose the next intervention by expected information gain;
2. can that intervention identify the hidden mechanism more efficiently than a matched random intervention budget; and
3. can the identified mechanism be reused immediately to solve a follow-up control/adaptation task more cheaply than restarting without the identification result?

The engine must work unchanged across qualitatively different black boxes.

## Scope

V3 is not another diversity-preservation experiment. Candidate hypotheses are supplied explicitly. It does not attempt to keep evolutionary modes alive, discover the hypothesis set, or solve ARC.

V3 proves the interface and cross-domain mechanism only. Later work may attach the same engine to public regression benchmarks or learned candidate models.

## Architecture

### Generic objects

The engine receives:

```text
candidate hypotheses
allowed interventions
predictive observation model
experiment budget
observation callback
adaptation operation
```

The core package must not import any domain adapter.

### Predictive distributions

Each `(hypothesis, intervention)` pair returns a predictive distribution rather than a point prediction. V3 supports two exact public implementations:

- `CategoricalDistribution` for finite observations;
- `GaussianDistribution` for noisy scalar observations.

Both expose a common interface sufficient for Bayesian posterior updates and deterministic expected-information calculations.

Categorical expected entropy is exact over the union of the predicted supports. Gaussian expected entropy uses fixed Gauss–Hermite quadrature over each hypothesis-conditioned predictive distribution. There is no Monte Carlo noise in intervention selection.

### Active engine

The generic engine provides:

```python
posterior_update(...)
expected_information_gain(...)
choose_intervention(...)
run_identification(...)
```

Requirements:

- any finite number of hypotheses, not only two;
- arbitrary hashable intervention objects;
- arbitrary categorical observations or scalar Gaussian observations;
- deterministic tie-breaking by stable intervention key;
- a hard experiment budget;
- intervention reuse disabled by default within one identification run;
- complete trace recording: prior, chosen intervention, information gain, observation, posterior, and confidence.

The identification result exposes the MAP hypothesis, posterior, probes used, confidence trace, correctness when a hidden truth is supplied for benchmarking, and whether the confidence threshold was reached.

### Adaptation/reuse boundary

Adaptation is deliberately domain-owned. The generic engine does not define what a successful follow-up task means. Each adapter exposes an operation conceptually equivalent to:

```python
adapt(selected_hypothesis, true_hidden_system, budget) -> AdaptationResult
```

The result records simulator/test executions, success, and domain-specific final error or status.

This prevents V3 from hiding domain assumptions inside the inference engine while still testing the claim that identification has downstream value.

## Benchmark adapters

All adapters use the same active engine and matched random policy. None may implement custom Bayesian updates or custom probe scoring.

### 1. Regression/configuration fault

A small executable software-service model contains four candidate regression mechanisms. They agree on the passive/default configuration but fail under different combinations of feature flags / request shapes.

- Intervention: run one diagnostic configuration.
- Observation: categorical `pass` / named failure signature.
- Reuse task: choose and validate a candidate patch on the hidden system.
- Cost: full validation-suite executions until a patch succeeds.

The active arm uses the MAP fault hypothesis to order patches. Restart ignores identification and uses a fixed neutral patch order. Random identification gets the same probe budget as active.

### 2. State machine

Four small deterministic Mealy-style machines share the passive trace but differ under short input sequences.

- Intervention: reset and execute a short input word.
- Observation: categorical output trace.
- Reuse task: reach a target terminal output/state using candidate control words.
- Cost: true-machine rollouts until success.

The selected hypothesis ranks control words by predicted success and length; restart uses a fixed neutral order.

### 3. Program transformation

Four executable sequence/string transformations agree on deliberately symmetric passive examples but differ on asymmetric probes.

- Intervention: transform one diagnostic input.
- Observation: categorical transformed tuple/string.
- Reuse task: apply the correct transformation family to a new payload and validate it.
- Cost: candidate transformation validations until the payload is accepted by the hidden oracle.

This adapter is intentionally discrete and noiseless, acting as a sanity check that the engine is not relying on Gaussian geometry.

### 4. Dynamical system

Four linear controlled scalar systems share the zero-input passive trajectory but differ in response to control pulses.

- Intervention: execute a short control sequence from reset.
- Observation: noisy final scalar state.
- Predictive model: Gaussian with the known observation noise.
- Reuse task: drive the hidden system toward a target state using a finite library of control sequences.
- Cost: true-system rollouts until target tolerance is reached.

This adapter forces the same engine to work with overlapping continuous likelihoods rather than exact categorical signatures.

### 5. Causal circuit

Four small causal circuits produce the same baseline output but differ in how upstream clamps/stimulations propagate through a hidden mediator / inhibitory path.

- Intervention: clamp or stimulate a named node.
- Observation: noisy scalar output.
- Reuse task: select a compensatory intervention after a fixed disturbance.
- Cost: true-circuit intervention trials until output returns within tolerance.

This is the mechanistic/neural-style adapter. It remains an abstract circuit and makes no biological claim.

## Experimental protocol

### Candidate prior

Uniform prior over hypotheses within each domain.

### Identification arms

For every hidden hypothesis and seed/noise realization:

1. `active`: choose each unused intervention by expected information gain;
2. `random`: choose unused interventions uniformly from the identical candidate set using a deterministic seeded RNG.

Both receive the same maximum probe budget and the same stopping confidence threshold.

### Follow-up arms

For each identification trace:

- `active_reuse`: adapt from the active MAP hypothesis;
- `random_reuse`: adapt from the random-policy MAP hypothesis;
- `restart`: perform the same adaptation task without an identified hypothesis, using the adapter's frozen neutral ordering;
- `oracle`: optional positive control using the true hypothesis directly.

All adaptation conditions get the same maximum true-system evaluation budget.

## Frozen metrics

Per domain record:

- identification accuracy;
- mean censored probes to confidence;
- posterior entropy after budget;
- active-vs-random paired identification wins/losses/ties;
- reuse success rate;
- mean censored adaptation executions;
- active-reuse vs restart cost ratio;
- active-reuse vs random-reuse cost ratio;
- oracle gap.

No cross-domain average can rescue a failed adapter.

## Frozen success gate

A domain passes only if all conditions hold on its canonical tape:

1. active identification accuracy is at least random accuracy;
2. active mean censored probes are at least `0.40` lower than random, **or** active accuracy exceeds random by at least `0.10` at the identical budget;
3. active-reuse success is not more than `0.05` below restart success;
4. active-reuse mean censored adaptation executions are at most `0.75 ×` restart;
5. active-reuse is no worse than random-reuse on mean censored adaptation executions;
6. oracle succeeds at least `0.95` of trials, establishing that the follow-up task is solvable when the mechanism is known.

V3 overall passes only if **every implemented domain passes individually**.

If an adapter fails because its intervention set contains insufficient information, that is reported as an adapter/gate failure rather than tuned away after canonical exposure.

## Pilot / canonical separation

Pilot seeds are used only to catch degenerate worlds, impossible follow-up tasks, or implementation mistakes. Pilot may tune observation noise, candidate intervention libraries, and follow-up target tolerances once.

After those values are frozen, canonical seeds are disjoint and no scientific parameter may change based on canonical results.

Suggested default:

```text
pilot:     seeds 700–715
canonical: seeds 800–927
```

Each canonical seed cycles through all hidden hypotheses so every mechanism is represented equally.

## Tests

Mechanism tests must establish:

- categorical posterior update against hand-computed values;
- Gaussian posterior update against direct likelihood calculation;
- expected information gain is zero for intervention-equivalent hypotheses;
- active chooser selects a uniquely discriminating intervention in a hand-built case;
- row/hypothesis permutation does not change the physical selected intervention;
- random and active policies cannot exceed the same budget;
- distributions normalize and reject invalid parameters;
- every adapter's passive observation is ambiguous by construction;
- every adapter has at least one intervention that changes the posterior;
- adaptation conditions count only true-system/test executions and obey equal budgets;
- core active engine has no imports from adapter modules.

## Claim boundary

A V3 pass would establish only:

> One finite-hypothesis active experimental-design engine can discriminate and immediately exploit hidden executable mechanisms across several synthetic black-box domains under matched budgets.

It would not establish optimal Bayesian experimental design in general, discovery of candidate mechanisms, robust diversity preservation, real-world regression debugging, general causal discovery, or biological realism.

A later gate must attach at least one adapter to an external/public benchmark before making a practical-tool claim.
