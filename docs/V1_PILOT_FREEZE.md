# EvoX V1 pilot freeze

The V1 design reserved seeds **300–307** for implementation debugging and hyperparameter selection. These seeds are exploratory and are excluded from the canonical classification.

## Pilot result

Under the fixed 128-program / 16-generation budget:

| strategy | valid two-mode seeds | mean minority-mode fraction |
|---|---:|---:|
| baseline | 6/8 | 0.0439 |
| structural novelty, 50% novelty slots | 6/8 | 0.0840 |
| structural novelty, 75% novelty slots | **7/8** | **0.1288** |
| structural novelty, 100% novelty slots | 7/8 | 0.0493 |
| root-lineage niching | 3/8 | 0.0654 |
| neutral archive | 5/8 | 0.0359 |

The novelty-allocation curve was non-monotonic. Nearby pilot checks also underperformed the 75% setting: 72.5% novelty averaged about 0.1108 minority mass, 77.5% about 0.0818, and 82.5% about 0.0521.

The full pilot audit for the frozen 75% structural-novelty setting remained inside the downstream claim fences:

- post-hoc mode purity: ~0.9913;
- active probes to confidence: 1.00 vs random ~2.64 censored mean;
- correct-mode transfer success: ~0.929;
- correct-mode transfer mean censored evaluations: 66 vs restart ~384.

## Frozen canonical configuration

Before examining any seed in the canonical range **400–431**, freeze structural novelty at:

```text
novelty_fraction = 0.75
```

This supersedes the original design's initial 50/50 elite allocation. No other classification threshold changes.

The canonical experiment still reports all four predeclared strategies. Root-lineage niching and the neutral archive remain as negative/control mechanisms even though they underperformed in the pilot.

No further strategy or hyperparameter tuning is allowed on seeds 400–431.
