# EvoX V3 Active Mechanism Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract EvoX's active identification/reuse machinery into a domain-agnostic Bayesian experimental-design engine and prove it across five independent black-box adapters.

**Architecture:** Add a generic finite-hypothesis engine supporting categorical and Gaussian predictive distributions. Keep domain logic in isolated adapter modules that provide hypotheses, interventions, observation callbacks, and adaptation routines. A single V3 experiment runner evaluates active versus random identification and active/random/restart/oracle reuse under matched budgets and freezes per-domain results.

**Tech Stack:** Python 3.11/3.12, NumPy, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-16-evox-v3-active-mechanism-engine-design.md`

## Global Constraints

- Core inference must not import any domain adapter.
- Candidate hypotheses are supplied explicitly; V3 does not discover or preserve hypotheses.
- Active and random identification arms receive identical intervention sets and maximum budgets.
- Canonical domains pass individually; no cross-domain average may rescue a failure.
- Pilot seeds: 700–715. Canonical seeds: 800–927.
- Canonical scientific parameters are frozen before canonical exposure.
- Keep NumPy as the only runtime dependency.

---

### Task 1: Generic predictive distributions and Bayesian updates

**Files:**
- Create: `src/evox/active.py`
- Create: `tests/test_active.py`

**Interfaces:**
- Produces `CategoricalDistribution`, `GaussianDistribution`, `entropy_bits`, `posterior_update`, `expected_information_gain`, `choose_intervention`.
- Predictive distributions expose `log_prob(observation)` and deterministic `quadrature()` weighted outcomes.

- [ ] **Step 1: Write failing categorical/Gaussian posterior tests**

Test a hand-computed two-hypothesis categorical update and a Gaussian update against direct normal log-likelihood arithmetic. Also test invalid probabilities / non-positive sigma.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `pytest -q tests/test_active.py`

Expected: import failure because `evox.active` does not yet exist.

- [ ] **Step 3: Implement distribution classes and posterior update**

Use normalized categorical mappings and `numpy.polynomial.hermite.hermgauss` for deterministic Gaussian quadrature. Compute posterior in log space with max subtraction.

- [ ] **Step 4: Add information-gain and chooser tests**

Include: zero gain when all hypotheses predict the same distribution; uniquely discriminating intervention selected; hypothesis-order permutation leaves the selected physical intervention unchanged; tie-breaking uses stable intervention representation.

- [ ] **Step 5: Implement information gain and chooser**

Compute `H(prior) - E_y[H(posterior|y)]` by summing each hypothesis-conditioned quadrature weighted by the prior. Never assume two hypotheses.

- [ ] **Step 6: Run tests and commit**

Run: `pytest -q tests/test_active.py tests/test_probes.py`

Expected: all pass; legacy `probes.py` behavior remains intact.

Commit: `feat: add generic active inference engine`

---

### Task 2: Identification loop and trace model

**Files:**
- Modify: `src/evox/active.py`
- Modify: `tests/test_active.py`

**Interfaces:**
- Produces `IdentificationStep`, `IdentificationResult`, and `run_identification(...)`.
- `run_identification(hypotheses, interventions, predict, observe, *, budget, confidence, policy, rng)` supports `policy="active"` and `policy="random"`.

- [ ] **Step 1: Write failing budget/trace tests**

Assert active and random never exceed budget, do not reuse interventions by default, record every prior/posterior, and stop early only when confidence is reached.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest -q tests/test_active.py`

- [ ] **Step 3: Implement identification loop**

Active calls `choose_intervention`; random samples from the same unused set with the supplied deterministic RNG. MAP tie-breaking follows hypothesis order only after equal posterior probability.

- [ ] **Step 4: Run tests and commit**

Run: `pytest -q tests/test_active.py`

Commit: `feat: add generic identification loop`

---

### Task 3: Domain adapter contract plus discrete adapters

**Files:**
- Create: `src/evox/domains/__init__.py`
- Create: `src/evox/domains/common.py`
- Create: `src/evox/domains/regression.py`
- Create: `src/evox/domains/state_machine.py`
- Create: `src/evox/domains/program_transform.py`
- Create: `tests/test_domains_discrete.py`

**Interfaces:**
- `DomainCase` contains `name`, `hypotheses`, `interventions`, `predict`, `observe`, `adapt`, `restart`, `oracle_adapt`, `probe_budget`, `adapt_budget`.
- `AdaptationResult` records `success`, `executions`, and optional `error`.

- [ ] **Step 1: Write contract and passive-ambiguity tests**

For each discrete adapter assert at least four hypotheses, a passive/default observation shared across all hypotheses, at least one informative intervention, and identical adaptation budget accounting across conditions.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_domains_discrete.py`

- [ ] **Step 3: Implement regression adapter**

Create four executable regression mechanisms that agree on default requests and diverge under feature/config diagnostic probes. Adaptation tries candidate patches on the true hidden mechanism; cost is validation-suite executions.

- [ ] **Step 4: Implement state-machine adapter**

Create four resettable deterministic machines with shared passive trace. Diagnostic interventions are short input words; adaptation ranks control words to reach a target output/state and counts true-machine rollouts.

- [ ] **Step 5: Implement program-transform adapter**

Create four transformations that agree on symmetric passive examples but diverge on asymmetric diagnostic inputs. Adaptation validates candidate transformation families on a new payload.

- [ ] **Step 6: Run tests and commit**

Run: `pytest -q tests/test_domains_discrete.py tests/test_active.py`

Commit: `feat: add discrete active-mechanism domains`

---

### Task 4: Continuous dynamical and causal adapters

**Files:**
- Create: `src/evox/domains/dynamics.py`
- Create: `src/evox/domains/causal_circuit.py`
- Create: `tests/test_domains_continuous.py`

**Interfaces:**
- Both adapters use `GaussianDistribution` from `evox.active`; neither defines custom posterior logic.

- [ ] **Step 1: Write passive-ambiguity, noisy-observation, and solvability tests**

Assert zero-input/baseline observations overlap by construction, at least one allowed intervention produces distinct Gaussian means, and oracle adaptation succeeds across all four hidden mechanisms under the frozen budget.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_domains_continuous.py`

- [ ] **Step 3: Implement dynamics adapter**

Use four scalar controlled linear systems with identical reset/passive behavior. Diagnostic probes are short control sequences; adaptation ranks a frozen library of control sequences to hit a target state and counts true-system rollouts.

- [ ] **Step 4: Implement causal-circuit adapter**

Use four abstract mediator/inhibition circuit mechanisms with equal baseline output. Diagnostic probes clamp/stimulate named nodes; adaptation chooses compensatory interventions after a fixed disturbance and counts true-circuit trials.

- [ ] **Step 5: Run tests and commit**

Run: `pytest -q tests/test_domains_continuous.py tests/test_active.py`

Commit: `feat: add continuous active-mechanism domains`

---

### Task 5: Cross-domain experiment runner and classifier

**Files:**
- Create: `experiments/run_v3_active_mechanisms.py`
- Create: `tests/test_v3_receipt.py`

**Interfaces:**
- `V3Config` freezes pilot/canonical ranges, confidence, trial counts, and thresholds.
- `run_domain(case, seeds, config)` returns per-trial traces and aggregate metrics.
- `classify_domain(summary)` applies the six frozen domain gates.
- `run_sweep(...)` returns a receipt with one classification per domain and `overall_pass` only when every domain passes.

- [ ] **Step 1: Write failing receipt/classification tests**

Test equal active/random probe budgets, equal adaptation budgets, correct censored-cost accounting, each gate independently causing a failure, and overall failure when any one domain fails.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_v3_receipt.py`

- [ ] **Step 3: Implement runner and summaries**

Cycle hidden hypotheses evenly across each seed tape. For every hidden trial run active and random identification, then active-reuse, random-reuse, restart, and oracle adaptation with matched budgets. Record raw traces rather than only aggregates.

- [ ] **Step 4: Run smoke sweep**

Run: `python -m experiments.run_v3_active_mechanisms --seed-start 700 --seed-count 2 --out /tmp/v3-smoke.json`

Expected: receipt contains all five domain names and per-domain classifications without exceptions.

- [ ] **Step 5: Run full tests and commit**

Run: `pytest -q`

Commit: `feat: add cross-domain active mechanism benchmark`

---

### Task 6: Pilot freeze, canonical receipt, documentation, and integration

**Files:**
- Create: `docs/V3_PILOT_FREEZE.md`
- Create: `docs/V3_RESULTS.md`
- Create: `results/v3_active_mechanisms.json`
- Modify: `README.md`
- Optional temporary workflow: `.github/workflows/v3-experiment.yml` only if local execution is unavailable; remove it before merge unless it remains useful as a reproducibility workflow.

**Interfaces:**
- Frozen canonical receipt is generated by the committed runner without post-canonical scientific tuning.

- [ ] **Step 1: Run pilot seeds 700–715**

Run all five adapters. Permit changes only for degenerate/uninformative probe libraries, impossible oracle follow-up tasks, or observation noise/tolerance making the task trivial/impossible. Document every pilot change.

- [ ] **Step 2: Freeze pilot parameters**

Write `docs/V3_PILOT_FREEZE.md` with exact final adapter constants and thresholds. Add regression tests for every tuned constant that must not drift into canonical execution.

- [ ] **Step 3: Verify full CI before canonical exposure**

Run: `pytest -q`

Expected: zero failures on Python 3.11 and 3.12.

- [ ] **Step 4: Run canonical seeds 800–927 exactly once**

Command: `python -m experiments.run_v3_active_mechanisms --seed-start 800 --seed-count 128 --out results/v3_active_mechanisms.json`

Do not change scientific parameters after reading this receipt.

- [ ] **Step 5: Write results interpretation**

Report every domain independently, including failures. State whether the same engine passed all domains; do not average away a failure. Document active/random identification and active/random/restart/oracle adaptation metrics.

- [ ] **Step 6: Update README and verify**

Run: `pytest -q`

Check that README claim language matches the frozen receipt exactly.

- [ ] **Step 7: Open PR, wait for GitHub CI, merge only if checks pass**

PR title: `EvoX V3: generic active mechanism selection and reuse`

Merge only after both Python matrix jobs pass on the actual PR head.
