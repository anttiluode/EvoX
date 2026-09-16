# EvoX v2 Unlabeled Behavioral Novelty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether output vectors on unlabeled, mechanism-independent inputs preserve alternative executable procedures better than syntax/ancestry/history diversity.

**Architecture:** Add descriptor-tape generation and behavior vectors as a separate probe-blind module. Extend evolution with a behavioral-novelty selector that consumes only an input tape and program outputs. A V2 runner compares baseline, V1 structural novelty, generic behavioral novelty, and an in-manifold unlabeled behavioral control while reusing V0/V1 audit machinery unchanged.

**Tech Stack:** Python 3.11+, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-evox-v2-unlabeled-behavior-design.md`

## Global Constraints

- Labeled fitness budget remains 128 programs × 16 generations for every arm.
- Descriptor tape has no targets and is generated independently of hidden mechanism code.
- Default descriptor length 32; behavioral novelty fraction 0.75 inherited from V1.
- Pilot seeds 500–507; canonical seeds 600–631.
- V0/V1 baseline and structural-novelty behavior must remain unchanged.

---

### Task 1: Mechanism-independent descriptor tapes

**Files:**
- Create: `src/evox/descriptors.py`
- Create: `tests/test_behavioral_novelty.py`

**Interfaces:**
- `generic_unlabeled_tape(*, size: int = 32, seed: int = 20260916) -> tuple[tuple[int,int,int], ...]`
- `manifold_unlabeled_tape(*, size: int = 32, seed: int = 20260916) -> tuple[tuple[int,int,int], ...]`
- `behavior_vector(tree: Node, tape: Sequence[Input]) -> np.ndarray`

- [ ] Write failing tests for deterministic uniqueness, cube bounds, and manifold relation.
- [ ] Assert direct `x0` and relational `x1*x2` vectors are identical on the manifold tape and differ on the generic tape.
- [ ] Implement tape generation without importing `worlds.target_value` or `worlds.counterfactual_pool`.
- [ ] Run targeted then full tests.

### Task 2: Behavioral-novelty elite selection

**Files:**
- Modify: `src/evox/evolution.py`
- Modify: `tests/test_behavioral_novelty.py`

**Interfaces:**
- New strategy: `behavioral_novelty`.
- `evolve(..., descriptor_inputs: Sequence[Input] | None = None, novelty_fraction=0.75)`.
- `EvolutionResult` gains `descriptor_executions: int = 0`.

- [ ] Write a failing selector test with equally fit programs where syntax is misleading but unlabeled outputs separate direct/relational behavior.
- [ ] Implement standardized behavior matrix + deterministic farthest-point selection among best-visible-MSE candidates.
- [ ] Count each program×descriptor-input execution performed for selection.
- [ ] Assert baseline/structural-novelty labeled evaluation sequences remain unchanged and descriptor executions are zero there.
- [ ] Run full suite.

### Task 3: Invariance and leakage controls

**Files:**
- Modify: `tests/test_behavioral_novelty.py`

- [ ] Test common permutation of descriptor coordinates leaves selected ids unchanged.
- [ ] Test production descriptor APIs accept inputs only, no targets.
- [ ] Test in-manifold tape cannot distinguish the two reference mechanisms.
- [ ] Test generic tape is generated without disagreement filtering.
- [ ] Run full suite.

### Task 4: V2 matched runner

**Files:**
- Create: `experiments/run_v2_behavioral_novelty.py`
- Modify: `tests/test_behavioral_novelty.py`

**Interfaces:**
- `V2Config` with descriptor seed/length and canonical seed range.
- Strategies: `baseline`, `structural_novelty`, `behavioral_generic`, `behavioral_manifold`.
- `run_strategy_seed(...)` maps behavioral arms to the same `behavioral_novelty` selector with different tapes.
- Reuse V1 post-hoc metrics/classification helpers where possible.

- [ ] Write failing smoke test for all four arms, equal labeled evaluations, explicit descriptor costs.
- [ ] Implement runner and frozen V2 classification.
- [ ] Run targeted/full tests.

### Task 5: Pilot without target leakage

**Files:**
- Create: `results/v2_pilot.json`
- Create: `docs/V2_BEHAVIORAL_NOVELTY.md`

- [ ] Run seeds 500–507 with default descriptor size 32 and novelty fraction 0.75.
- [ ] If there is an implementation pathology, fix with tests; otherwise do not tune.
- [ ] Freeze descriptor seed/size/allocation before any canonical seed is examined.
- [ ] Record descriptor execution cost alongside preservation metrics.

### Task 6: Canonical sweep

**Files:**
- Create: `results/v2_behavioral_novelty.json`
- Modify: `docs/V2_BEHAVIORAL_NOVELTY.md`
- Modify: `README.md`

- [ ] Run matched seeds 600–631 for all four strategies.
- [ ] Apply frozen thresholds exactly.
- [ ] Preserve negative/control outcomes without retuning.
- [ ] Compare generic behavior novelty specifically against the in-manifold behavioral control and V1 structural novelty.

### Task 7: Verification and integration

- [ ] Fresh `pytest -q`.
- [ ] Run one-seed V2 receipt twice and compare byte-for-byte.
- [ ] Open PR and verify exact branch on GitHub Actions Python 3.11/3.12.
- [ ] Merge only after green CI.
