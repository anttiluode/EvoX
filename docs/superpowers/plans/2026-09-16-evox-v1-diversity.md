# EvoX v1 Diversity Preservation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three probe-blind preservation strategies and test whether any maintains counterfactually distinct executable procedures at meaningful mass under the failed V0 budget.

**Architecture:** Preserve the existing `evolve()` default path exactly. Extend program trees with deterministic syntax features and evolution individuals with root lineage ids. Strategy-specific elite selection and a bounded neutral archive live inside `evolution.py`; a new experiment module reuses V0's post-hoc mode/probe/transfer pipeline unchanged.

**Tech Stack:** Python 3.11+, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-evox-v1-diversity-design.md`

## Global Constraints

- Population 128, generations 16, elites 24, max depth 3 for the canonical V1 comparison.
- Preservation code may not read counterfactual outputs, mode labels, SVD coordinates, probe information gain, or transfer results.
- Every strategy uses exactly the same program-evaluation budget.
- Pilot seeds 300–307; canonical seeds 400–431.
- Existing `evolve()` calls without a strategy must remain behaviorally identical.

---

### Task 1: Syntax feature representation

**Files:**
- Modify: `src/evox/program.py`
- Test: `tests/test_diversity.py`

**Interfaces:**
- Produces: `tree_depth(node: Node) -> int`
- Produces: `syntax_features(node: Node) -> np.ndarray` with fixed feature order `[nodes, depth, x0, x1, x2, constants, neg, add, sub, mul]`.

- [ ] **Step 1: Write failing tests** asserting exact depth and feature counts for `x0` and `x1*x2`.
- [ ] **Step 2: Run** `pytest tests/test_diversity.py -q` and verify failure because the functions do not exist.
- [ ] **Step 3: Implement** recursive `tree_depth` and `syntax_features` using syntax only.
- [ ] **Step 4: Run** `pytest tests/test_diversity.py -q` and verify green.
- [ ] **Step 5: Run** `pytest -q` to protect V0 behavior.

### Task 2: Root-lineage identity and baseline compatibility

**Files:**
- Modify: `src/evox/evolution.py`
- Modify: `tests/test_diversity.py`

**Interfaces:**
- `Individual` gains `root_id: int`.
- Generation-0 individuals use `root_id == id`; all clones/mutations inherit `parent.root_id`.
- Existing `evolve(..., strategy="baseline")` and omitted strategy are equivalent.

- [ ] **Step 1: Write failing tests** for root inheritance through at least two generations and default/baseline receipt equality for a fixed seed.
- [ ] **Step 2: Run** targeted tests and verify red.
- [ ] **Step 3: Add root propagation** without changing ranking, RNG draws, mutation, or population sizes on the default path.
- [ ] **Step 4: Run** targeted tests and full suite.

### Task 3: Structural-novelty selector

**Files:**
- Modify: `src/evox/evolution.py`
- Modify: `tests/test_diversity.py`

**Interfaces:**
- `evolve(..., strategy="structural_novelty")`.
- Half of elite slots use baseline rank; remaining slots use deterministic farthest-point syntax-feature selection among best-MSE ties before fallback.

- [ ] **Step 1: Write a failing selector test** with equally fit direct/relational/redundant trees; assert the novelty selector retains structurally separated trees while baseline chooses by normal rank.
- [ ] **Step 2: Verify red.**
- [ ] **Step 3: Implement feature standardization and deterministic farthest-point filling.** No counterfactual evaluation imports are permitted in `evolution.py`.
- [ ] **Step 4: Verify targeted and full tests.**

### Task 4: Root-lineage niching selector

**Files:**
- Modify: `src/evox/evolution.py`
- Modify: `tests/test_diversity.py`

**Interfaces:**
- `evolve(..., strategy="lineage_niching")`.
- Elite filling takes the best program from distinct root ids before ordinary fallback.

- [ ] **Step 1: Write a failing test** where ordinary rank would select several descendants of one root but niching must include multiple roots.
- [ ] **Step 2: Verify red.**
- [ ] **Step 3: Implement deterministic root-diverse elite selection.**
- [ ] **Step 4: Verify targeted and full tests.**

### Task 5: Neutral archive

**Files:**
- Modify: `src/evox/evolution.py`
- Modify: `tests/test_diversity.py`

**Interfaces:**
- `evolve(..., strategy="neutral_archive", archive_capacity=64, archive_fraction=0.25)`.
- Archive contains unique source strings that achieved `mse <= viable_mse` in any completed generation.
- Archive reinjection consumes elite/population slots; it never increases evaluation count.

- [ ] **Step 1: Write failing tests** proving non-viable programs never enter the archive and evaluation counts equal baseline.
- [ ] **Step 2: Verify red.**
- [ ] **Step 3: Implement deterministic insertion-order bounded archive and round-robin reinjection.**
- [ ] **Step 4: Verify targeted and full tests.**

### Task 6: Matched V1 experiment runner

**Files:**
- Create: `experiments/run_v1_diversity.py`
- Modify: `tests/test_diversity.py`

**Interfaces:**
- `V1Config` carries fixed V0 budget, pilot/canonical ranges, archive parameters.
- `run_strategy_seed(strategy: str, seed: int, config: V1Config) -> dict[str, object]`.
- `run_canonical(config: V1Config) -> dict[str, object]` returns per-strategy summaries/classifications.

- [ ] **Step 1: Write failing smoke tests** asserting four strategies, equal evaluation counts, and minority-mode fraction calculation.
- [ ] **Step 2: Verify red.**
- [ ] **Step 3: Implement runner** by reusing V0 mode extraction and hidden-trial helpers or factoring shared post-hoc audit helpers without changing their behavior.
- [ ] **Step 4: Add the fixed classification thresholds from the spec.**
- [ ] **Step 5: Run smoke tests and full suite.**

### Task 7: Pilot and freeze parameters

**Files:**
- Create: `results/v1_pilot.json`
- Create: `docs/V1_DIVERSITY.md`

- [ ] **Step 1: Run** all four strategies on seeds 300–307.
- [ ] **Step 2: Record** valid-seed fraction, minority-mode mass, purity, visible fitness, active/random probes, and transfer metrics for every strategy.
- [ ] **Step 3: If implementation bugs are found, fix them with tests.** Strategy hyperparameters may be adjusted only here and must then be frozen/documented before canonical seeds are run.
- [ ] **Step 4: Write the pilot receipt and explicit frozen configuration.**

### Task 8: Disjoint canonical experiment

**Files:**
- Create: `results/v1_diversity.json`
- Modify: `docs/V1_DIVERSITY.md`
- Modify: `README.md`

- [ ] **Step 1: Run** seeds 400–431 for all four strategies using the frozen pilot configuration.
- [ ] **Step 2: Classify** every strategy independently as `PRESERVES_MODES`, `SURVIVES_BUT_THIN`, `DIVERSITY_HURTS_FITNESS`, or `NO_PRESERVATION_GAIN`.
- [ ] **Step 3: Record failures as failures** without relaxing thresholds.
- [ ] **Step 4: Update README** with the narrow result and the next falsifiable gate.

### Task 9: Verification and integration

**Files:**
- All changed files.

- [ ] **Step 1: Run** `pytest -q` fresh.
- [ ] **Step 2: Run** a deterministic one-seed V1 receipt twice and compare byte-for-byte.
- [ ] **Step 3: Push/open PR** and verify GitHub Actions on Python 3.11 and 3.12.
- [ ] **Step 4: Merge only after CI is green.**
