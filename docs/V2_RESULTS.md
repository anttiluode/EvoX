# EvoX V2 — unlabeled behavioral novelty

## Canonical result

V2 asked whether a population could preserve alternative executable procedures using **only unlabeled behavior** on a generic input tape. The descriptor never receives target outputs, hidden mode labels, counterfactual-probe labels, or the later transfer task.

The canonical sweep was frozen before opening seeds **600–631**. The structural-novelty control retained its V1 setting (`0.75` novelty allocation); the behavioral arm used the pilot-frozen `0.50` allocation. Descriptor length and seeds were unchanged after the pilot.

The exact canonical receipt is `results/v2_behavioral.json`, copied byte-for-byte from GitHub Actions run `35071529706` after the run completed successfully.

## What happened

| arm | seeds with two post-hoc modes | minority mass, conditional on mode present | minority mass across all 32 seeds | descriptor executions / seed |
|---|---:|---:|---:|---:|
| baseline | 20/32 = 62.5% | 4.62% | 2.89% | 0 |
| in-manifold behavioral control | 20/32 = 62.5% | 4.62% | 2.89% | 61,440 |
| generic behavioral novelty | **30/32 = 93.75%** | 4.99% | **4.68%** | 61,440 |
| structural novelty | 28/32 = 87.5% | **9.34%** | **8.17%** | 0 |

`minority mass across all 32 seeds` is a descriptive post-hoc summary computed by assigning zero mass to seeds where the second mode is absent. It is useful because the receipt's `mean_minority_mode_fraction` is conditional on a seed being scorable as two-mode.

The frozen V2 classifier returns:

```text
behavioral_generic = FUNCTIONAL_BUT_THIN
```

That label is accurate only in the narrow sense encoded by the gate: the behavioral arm clears the 85% mode-presence threshold and preserves downstream probe/transfer behavior, but it does **not** clear the required 15% minority-mass threshold.

## The important distinction: presence is not mass

Behavioral novelty did measurable work. Relative to baseline it changed ten seeds from `INSUFFICIENT_DIVERSITY` to `OK`; across all 32 seeds its zero-inclusive minority mass rose from 2.89% to 4.68%. This is not just a reporting artifact.

But the population is still extremely thin. Among seeds where a second mode exists, the minority contains only 4.99% of viable programs on average. The canonical target was 15%.

So V2 establishes:

> **Unlabeled behavioral novelty can rescue the presence of an alternative procedure without knowing which behavior is correct, but it does not make that procedure competitively abundant.**

That is weaker than the original preservation goal.

## The destructive control worked

The in-manifold behavioral tape samples only inputs on the ambiguity manifold where the two reference procedures agree. Its descriptor variance is exactly zero in the canonical summary:

```text
mean_pairwise_descriptor_distance = 0.0
mean_descriptor_nonzero_coordinate_fraction = 0.0
```

Its population-level summary is therefore identical to baseline on mode-presence rate, minority mass, active probing, and transfer.

This is the cleanest evidence that the generic behavioral arm is not receiving hidden labels through the descriptor path. The descriptor helps only when the unlabeled input geometry actually exposes behavioral disagreement.

## Structural novelty still wins the preservation comparison

The surprising part of V2 is that the cheaper V1 structural proxy remains stronger on the quantity we actually wanted to increase.

Structural novelty reaches 28/32 two-mode seeds and 9.34% conditional minority mass, versus behavioral novelty's 30/32 and 4.99%. Counting missing modes as zero, structural novelty reaches 8.17% mean minority mass versus behavioral novelty's 4.68%.

Behavioral novelty therefore buys slightly broader **presence**, but structural novelty buys much more **mass**, with no extra descriptor-execution cost.

Neither reaches the frozen 15% minority-mass target.

## The downstream half continues to work

For the behavioral arm, once two modes are present:

- active identification accuracy: **98.33%**;
- matched random identification accuracy: **98.33%**;
- probes to confidence: **1.00 active** vs **2.58 random**;
- correct-mode transfer success: **98.33%**;
- correct-mode transfer evaluations, censored mean: **56.65**;
- fresh-restart transfer success: **66.67%**;
- fresh-restart evaluations, censored mean: **296.92**.

The preservation problem and the identification/reuse problem have now separated empirically. The second is robust once an alternative survives; the first remains the bottleneck.

## V0 → V1 → V2 conclusion

The three gates now tell a coherent story:

1. **V0:** rare alternative procedures can survive, be exposed by counterfactual behavior, actively identified, and reused.
2. **V1:** syntax, genealogy, and neutral-history mechanisms do not reliably preserve alternatives at useful mass; structural novelty gives a partial improvement.
3. **V2:** unlabeled behavioral geometry substantially reduces complete mode extinction, but still leaves the alternative at only about 5% conditional mass and is less effective than structural novelty at increasing abundance.

The next experiment should therefore **not** be a fifth novelty proxy. A more informative direction is to separate the successful active-diagnosis/reuse machinery from the failed abundance objective, or to change the selection ecology itself so multiple procedures have an explicit reason to remain competitively viable.

The latter would be a different question from novelty preservation: instead of asking for a better proxy for diversity, ask what objective or environment makes multiple computational strategies genuinely useful before the hidden context is known.
