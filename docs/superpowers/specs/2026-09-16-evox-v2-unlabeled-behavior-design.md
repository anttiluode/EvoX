# EvoX v2 — Unlabeled Behavioral Novelty

## Purpose

V1 falsified three generic proxies for computational diversity: syntax novelty, root-lineage diversity, and a neutral solution archive. The best of them, structural novelty, improved survival but reached only 71.9% valid two-mode seeds and 7.3% minority-mode mass against frozen 85% / 15% gates.

V2 tests the stronger hypothesis suggested by GAx, novelty search, and behavioral/semantic genetic programming:

> Preserve procedures because they *behave differently on unlabeled situations*, without revealing which behavior is correct.

The later active probe remains the first place where an oracle answer is supplied.

## Information firewall

Behavioral preservation may use:

- visible-task MSE;
- candidate program outputs on a fixed unlabeled descriptor-input tape;
- deterministic RNG state.

It may not use:

- target values on descriptor inputs;
- the V0 `counterfactual_pool()` construction or its disagreement filter;
- direct/relational hidden labels;
- SVD mode labels used for post-hoc audit;
- active-probe observations or information gain;
- transfer results.

The descriptor tape itself must be generated independently of the hidden mechanisms.

## Fixed world and evolutionary budget

Keep V0/V1 unchanged:

- direct procedure: `x0`;
- relational procedure: `x1*x2`;
- visible labeled examples satisfy `x0 = x1*x2`;
- population: 128;
- generations: 16;
- elites: 24;
- max depth: 3;
- viable MSE: `1e-12`.

Every arm receives the same number of labeled fitness evaluations. Behavioral-novelty arms perform additional **unlabeled descriptor executions**; these are counted and reported separately rather than hidden.

## Descriptor tapes

### Generic unlabeled tape

Construct a deterministic list of 32 unique integer triples by sampling without replacement from the full cube:

```text
x0,x1,x2 in {-3,-2,-1,0,1,2,3}
```

Use a fixed tape-generation seed stored in the experiment config. No point is accepted/rejected based on `x0`, `x1*x2`, target values, or disagreement between hidden procedures.

### In-manifold unlabeled control

Construct 32 deterministic triples satisfying:

```text
x0 = x1*x2
```

again without target labels. On this tape the reference direct and relational procedures have identical behavior by construction. If behavioral novelty succeeds here equally well, the claimed mechanism is suspect.

## Behavioral descriptor

For program `p` and descriptor inputs `z_1 ... z_m`, define:

```text
b(p) = [p(z_1), ..., p(z_m)]
```

No target vector exists inside the preservation code.

Before distance calculations, standardize descriptor coordinates across the current candidate population. Constant coordinates contribute zero distance.

## Strategies

### 1. Baseline

Unchanged V0 selector.

### 2. Structural novelty control

Frozen V1 best setting: 25% ordinary elite pressure, 75% syntax-space novelty.

### 3. Generic behavioral novelty

Use the same 25% ordinary / 75% novelty elite allocation, but farthest-point novelty is computed in the standardized unlabeled behavioral descriptor space rather than syntax space. Candidates must still be tied at the best visible MSE before novelty slots are spent.

### 4. In-manifold behavioral novelty control

Identical selector and allocation, using only the in-manifold descriptor tape.

## Selection invariants

- The ordinary baseline slots are chosen first.
- Novelty slots use deterministic farthest-point sampling among unique best-visible-MSE candidates.
- Ties use the existing deterministic rank key.
- Descriptor values never affect visible MSE.
- No extra offspring or labeled evaluations are granted.

## Pilot and canonical ranges

Pilot seeds: **500–507**. Pilot may be used only to fix descriptor-tape seed/length and behavioral-novelty fraction if an implementation pathology is discovered. The default is 32 descriptor points and the V1-frozen 0.75 novelty fraction; prefer leaving them unchanged.

Canonical seeds: **600–631**, disjoint from V0/V1/pilot seeds.

All four strategies use the same seed tape.

## Primary metrics

Reuse V1 post-hoc audit after evolution:

- valid two-mode seed fraction;
- mean and median minority-mode fraction;
- post-hoc mode purity;
- viable count;
- visible-task best/median MSE;
- active-vs-random probe efficiency;
- correct-mode vs restart transfer performance.

Add:

- descriptor program executions per run;
- mean pairwise behavioral distance among final visible-viable programs;
- fraction of descriptor coordinates with nonzero population variance.

## Frozen classification

A behavioral strategy is `PRESERVES_FUNCTIONAL_MODES` only if canonical results satisfy all of:

1. valid two-mode seed fraction >= **0.85**;
2. mean minority-mode fraction >= **0.15**;
3. mean post-hoc mode purity >= **0.95**;
4. median best visible MSE no worse than baseline by more than `1e-12`;
5. active probing uses at least **0.5 fewer censored probes** than random;
6. correct-mode transfer mean censored evaluations <= **0.75×** restart and success is no more than 0.05 below restart;
7. generic behavioral novelty beats the in-manifold behavioral control by at least **0.10 absolute valid-seed fraction OR 0.05 absolute minority-mode mass**.

`FUNCTIONAL_BUT_THIN`: conditions 1,3,4,5,6,7 pass but minority mass <0.15.

`NO_FUNCTIONAL_PRESERVATION_GAIN`: the generic behavioral strategy fails conditions 1 or 7 or gives no meaningful improvement over V1 structural novelty.

`LEAK_OR_CONTROL_FAILURE`: the in-manifold control performs essentially as well as generic behavioral novelty on the preserved-mode metrics, undermining the intended counterfactual-behavior interpretation.

## Destructive controls

Tests must prove:

1. descriptor generation does not import/call `target_value`, `counterfactual_pool`, mode audit, probe, or transfer code;
2. direct and relational references are identical on the in-manifold tape;
3. generic tape generation is independent of whether direct/relational code exists;
4. common permutation of descriptor coordinates leaves pairwise distances/selection invariant;
5. adding target labels to a test helper cannot alter behavioral selection because the production API accepts inputs only;
6. all strategies retain identical labeled fitness-evaluation counts;
7. descriptor-execution cost is reported explicitly;
8. existing baseline and structural-novelty behavior remain unchanged.

## Claim boundary

A V2 pass would show that unlabeled counterfactual response geometry can preserve alternative executable procedures in this controlled ambiguous world. It would not yet show general program synthesis or transfer to ARC. It would, however, justify moving the same separation—labeled fitness versus unlabeled behavioral descriptor—to a public task family.

## Prior-art fence

V2 is deliberately adjacent to novelty search, quality-diversity/MAP-Elites, semantic GP, and behavioral program synthesis. The novelty is not the generic idea of behavioral diversity. EvoX's specific scientific question is whether **unlabeled behavior can preserve alternative algorithms that are observationally equivalent on the labeled task, so a later actively chosen observation can select between them**.
