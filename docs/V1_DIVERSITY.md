# EvoX V1 — probe-blind diversity preservation

V0's bottleneck was not mode extraction or active identification. It was **purification**: one executable procedure family often disappeared before the later machinery could use it.

V1 therefore attacked only preservation. Every intervention was forbidden from reading counterfactual target outputs, SVD coordinates, hidden mode labels, probe information gain, or transfer results.

## Strategies

All strategies used the same visible examples, mutations, 128-program population, 16 generations, 24 elite slots, and total evaluation budget.

- **baseline** — V0 fitness/parsimony ranking;
- **structural novelty** — 25% ordinary elite pressure + 75% deterministic farthest-point selection in syntax-feature space among best-visible-fitness ties;
- **root-lineage niching** — preserve distinct generation-0 ancestry roots;
- **neutral archive** — remember unique visible-task solutions and re-inject them in 25% of elite slots.

The 75% structural-novelty allocation was frozen on pilot seeds 300–307 before any canonical seed was examined. See `V1_PILOT_FREEZE.md`.

## Canonical result

Canonical seeds: **400–431**, 32 matched seeds per strategy.

| strategy | valid two-mode seeds | valid fraction | mean minority-mode mass | mode purity | classification |
|---|---:|---:|---:|---:|---|
| baseline | 17/32 | 0.531 | 0.0442 | 1.000 | BASELINE |
| structural novelty | **23/32** | **0.719** | **0.0727** | 1.000 | `NO_PRESERVATION_GAIN` |
| root-lineage niching | 17/32 | 0.531 | 0.0678 | 1.000 | `NO_PRESERVATION_GAIN` |
| neutral archive | 16/32 | 0.500 | 0.0413 | 1.000 | `NO_PRESERVATION_GAIN` |

The frozen gate required at least **0.85 valid-seed fraction** and **0.15 mean minority-mode mass**, while preserving visible fitness and the V0 downstream benefits. No intervention crossed it.

Structural novelty is not useless. Relative to baseline it raised valid-mode survival from 53.1% to 71.9% and minority mass from 4.4% to 7.3%. But the pilot effect did not generalize strongly enough to count as preservation.

## Downstream audit

When two modes did survive, the rest of the EvoX chain remained strong.

For structural novelty on valid canonical seeds:

- active identification accuracy: **1.000**;
- active probes to confidence: **1.00** censored mean;
- matched random probes: **2.26**;
- correct-mode transfer success: **1.000**;
- correct-mode transfer evaluations: **36.2** censored mean;
- fresh restart: **293.5** censored mean, 0.674 success.

So V1 did **not** invalidate V0's extraction / interrogation / reuse result. It isolated preservation as the unresolved layer.

## What failed

Three intuitive notions of "diversity" were not equivalent to computational diversity:

1. **Syntax diversity** helped somewhat, but many structurally different trees implement the same effective procedure.
2. **Genealogical diversity** did not preserve the hidden computational distinction; different roots can converge to the same computation.
3. **Historical archiving** preserved old visible-task solutions, but without a behavioral descriptor it mostly remembered redundant solutions.

This is an important negative result. The system cannot infer which neutral structural distinctions matter from visible fitness when the visible world deliberately makes the algorithms observationally equivalent.

## Next gate

V2 should preserve **functional diversity without target labels**.

For a fixed set of unlabeled probe inputs, evaluate each program's own output vector:

```text
program
   ↓
outputs on unlabeled inputs
   ↓
behavior descriptor
```

Use that descriptor for novelty / quality-diversity selection while withholding the oracle target on those inputs. The preservation mechanism therefore learns only that two programs *behave differently*, not which one is correct.

This would test the stronger idea suggested by GAx:

> preserve distinct computations by their counterfactual response geometry, then use a later real observation to select which computation is appropriate.

That is the next gate before ARC-style tasks.
