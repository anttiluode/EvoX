# EvoX V2 pilot freeze

V2 asks whether **unlabeled program behavior** is a better preservation signal than syntax, ancestry, or historical solution identity.

The preservation tape is deliberately target-free. The generic tape is sampled from the full `[-3,3]^3` cube with fixed seed `271828`; it is not filtered using the direct or relational reference mechanisms. The in-manifold control is sampled independently with fixed seed `314159` from inputs satisfying `x0=x1*x2`, where the two reference procedures are observationally identical.

## Pilot-only seeds

All tuning below used seeds **500–507**. Canonical seeds **600–631** were not inspected before this freeze.

With the inherited V1 novelty allocation of 0.75, the first four-arm pilot produced:

| arm | valid seeds | mean minority-mode mass |
|---|---:|---:|
| baseline | 4/8 | 0.0637 |
| structural novelty (V1 control, 0.75) | 6/8 | 0.0880 |
| generic behavioral novelty (0.75) | 3/8 | 0.0492 |
| in-manifold behavioral control (0.75) | 4/8 | 0.0637 |

The in-manifold control had exactly zero descriptor variance and zero mean pairwise descriptor distance, while its downstream result matched baseline. That is the intended firewall control: merely paying for descriptor evaluations does not create diversity if the unlabeled situations do not expose functional differences.

The generic behavioral arm at 0.75 was worse than the structural control, so the V2 design's one allowed pilot-only hyperparameter—the novelty-slot fraction—was swept at `0.25, 0.50, 0.75, 1.00` on the same pilot seeds.

| behavioral novelty fraction | valid seeds | mean minority-mode mass |
|---:|---:|---:|
| 0.25 | 4/8 | 0.0671 |
| **0.50** | **4/8** | **0.0741** |
| 0.75 | 3/8 | 0.0492 |
| 1.00 | 4/8 | 0.0687 |

No setting is close to the canonical V2 success gates of 0.85 valid-seed fraction and 0.15 minority-mode mass. The sweep therefore does **not** establish a positive result. It only selects the least-bad predeclared pilot setting.

## Frozen canonical configuration

The canonical sweep uses:

- population `128`, generations `16`, elites `24`;
- canonical seeds `600–631`;
- structural novelty fraction **0.75**, unchanged from V1;
- behavioral novelty fraction **0.50**, frozen from the V2 pilot sweep;
- generic descriptor seed `271828`, length `32`;
- in-manifold descriptor seed `314159`, length `32`;
- all V2 thresholds unchanged.

The structural and behavioral fractions are separate configuration fields so tuning V2 cannot silently move the frozen V1 control.

Descriptor computation is not free: each behavioral run reports `population_size × descriptor_length × (generations-1)` extra program executions. Labeled fitness evaluations remain identical across all arms.

No further pilot tuning is permitted before the canonical sweep.
