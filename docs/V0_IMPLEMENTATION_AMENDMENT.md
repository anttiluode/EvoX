# EvoX V0 implementation amendment

This note records two implementation-time changes that supersede the earlier sketches in `docs/superpowers/specs/2026-09-16-evox-v0-design.md` and `docs/superpowers/plans/2026-09-16-evox-v0.md`.

## 1. Follow-up task

The original design sketched a follow-up target `base + 1`. The first transfer pilot exposed a confound: destructive subtree replacement could erase the relational core `x1*x2`, making the transfer test partly a test of mutation fragility rather than reuse of a discovered computational mode.

Before the canonical run, the follow-up was changed to:

```text
direct:      x0    + x2
relational:  x1*x2 + x2
```

This preserves the core mechanism while requiring a new context term.

## 2. Transfer mutation

Base-task evolution retains ordinary subtree replacement. Seeded transfer additionally uses a generic structure-preserving wrapper mutation: keep the whole parent program and compose one small `add`, `sub`, or `mul` operation with a terminal around it.

The same mutation policy is applied to correct-mode, wrong-mode, and scrambled-mode seeded controls. Fresh restart continues to use the normal random program generator. No hidden family label enters mutation.

## Frozen configurations

Canonical V0 used:

```text
seeds        100–115
population   128
generations  16
elites       24
probe sigma  2.0
```

It retained separable behavior on 10/16 seeds and therefore failed the unchanged 70% diversity threshold with classification `INSUFFICIENT_DIVERSITY`.

V0b attacked only the evolutionary budget on a new disjoint seed range:

```text
seeds        200–215
population   192
generations  24
elites       32
```

All downstream mode extraction, probing, transfer controls, and classification thresholds were unchanged. V0b retained separable behavior on 13/16 seeds and classified `PASS_ACTIVE_MODES`.

See `docs/V0_DIVERSITY_ATTACK.md` and the frozen receipts in `results/`.
