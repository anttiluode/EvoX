# EvoX

**Evolve procedures. Keep the alternatives. Ask the experiment that tells you which procedure you actually need.**

EvoX is a small synthesis of two earlier threads:

- **GAx:** evolutionary history can contain persistent computational alternatives rather than only a single winner;
- **AnotherOddThing:** when several hidden mechanisms remain plausible, choose the intervention that separates them most efficiently.

The first gate is intentionally synthetic. It asks whether those ideas can be joined without hiding the hard part behind a large benchmark.

## V0 question

Can a genetic-programming population contain two executable procedures that are indistinguishable on the data seen so far, yet become separable as vectors of counterfactual behavior?

And if so:

1. can SVD expose those behavioral directions without reading program syntax?
2. can an information-gain policy choose a probe that identifies the hidden procedure family?
3. can the identified family be reused on a related task more efficiently than restarting search?

## The deliberately ambiguous world

Programs are safe expression trees over `(x0, x1, x2)` using constants, `+`, `-`, `*`, and negation.

The two reference mechanisms are:

```text
direct:       x0
relational:   x1 * x2
```

Evolution initially sees only examples satisfying:

```text
x0 = x1 * x2
```

so both mechanisms are exactly correct. Training fitness cannot tell them apart.

A separate counterfactual pool breaks that correlation. For each viable program we evaluate its outputs on those unseen probes, producing a behavior matrix:

```text
programs × counterfactual probes
```

The matrix is centered, decomposed by SVD, and clustered in the retained singular-coordinate space. **Program syntax, ancestry labels, and the hidden family label are not inputs to mode extraction.** The direct/relational labels are revealed only afterward to audit purity.

## Active identification

Once two behavioral modes exist, EvoX treats them as competing hypotheses. For every possible probe it asks how much the expected posterior entropy would fall under a Gaussian observation model, then selects the highest-information probe.

The matched random control gets the same candidate pool and the same maximum of three probes.

## Transfer test

The follow-up target keeps the discovered core computation but adds a new context term:

```text
direct follow-up:       x0      + x2
relational follow-up:   x1*x2   + x2
```

The transfer mutation is generic rather than target-specific: it can replace a subtree or preserve the entire parent program and compose one small `+`, `-`, or `*` operation around it.

Four equal-budget conditions are compared:

```text
correct mode seed
wrong mode seed
scrambled-mode warm start
fresh random restart
```

The scrambled control is important. If any old program population were useful simply because it was already evolved, scrambled warm starts should match the correctly identified mode.

# Results

There are two frozen receipts because the first canonical run failed for an interesting reason.

## V0 — smaller evolutionary budget: failed diversity gate

Configuration: 128 programs, 16 generations, seeds 100–115.

The downstream mechanism was strong when both procedures survived, but only **10/16 seeds (62.5%)** retained counterfactually distinct behavior. The predeclared diversity threshold was 70%, so the correct classification is:

```text
INSUFFICIENT_DIVERSITY
```

Among the valid seeds:

| metric | active/correct | control |
|---|---:|---:|
| mode purity | **1.000** | — |
| probes to confidence, censored mean | **1.00** | random **1.85** |
| final posterior entropy | **4.17e-7 bits** | random **0.142 bits** |
| transfer evaluations to exact, censored mean | **53.7** | restart **290.2** |
| transfer success | **100%** | restart **75%** |

Six seeds purified to zero counterfactual variance before mode extraction. That failure is the point: the alternative algorithm disappeared.

Frozen receipt: [`results/v0.json`](results/v0.json)

## V0b — larger search/history budget: passed the same gate

V0b changed only the evolutionary budget and used a new disjoint seed range. Thresholds and all downstream machinery were unchanged.

Configuration: 192 programs, 24 generations, seeds 200–215.

**13/16 seeds (81.25%)** retained separable behavior, crossing the same frozen 70% threshold.

Classification:

```text
PASS_ACTIVE_MODES
```

Across the 13 valid seeds / 26 hidden-mode trials:

| metric | correct / active | wrong mode | scrambled | restart / random |
|---|---:|---:|---:|---:|
| post-hoc mode purity | **1.000** | — | — | — |
| probes to confidence, censored mean | **1.00** | — | — | **2.00** |
| final posterior entropy | **4.17e-7 bits** | — | — | **0.161 bits** |
| transfer success | **100%** | 50.0% | 76.9% | 53.8% |
| evaluations to exact, censored mean | **22.1** | 388.0 | 178.2 | 306.5 |
| evaluations to exact, censored median | **12.5** | 459.0 | 32.5 | 253.5 |

Frozen receipt: [`results/v0b_large_budget.json`](results/v0b_large_budget.json)

## The uncomfortable detail

The passing V0b populations are **not balanced mixtures of two rich species**. The minority behavioral mode averages only about **4.4%** of the viable population, with some valid seeds retaining a single minority program.

That narrows the result considerably:

> EvoX v0 shows that a rare surviving alternative procedure can be exposed by counterfactual behavior, actively identified, and reused. It does **not** yet show stable broad computational lineages.

The ancestry audit is consistent with long-lived families once present: among viable parent→child links, the mean mode-transition rate in V0b is about **0.49%**. But the main unsolved problem is keeping the alternative family alive reliably and at meaningful mass.

That is the next gate, not something to hide with a lower threshold.

# Run

```bash
python -m pip install -e '.[test]'
pytest -q
python -m experiments.run_v0 --out results/v0.json
```

The canonical runner is deterministic. The larger-budget V0b receipt uses the same runner with:

```python
Config(
    population_size=192,
    generations=24,
    elite_count=32,
    canonical_seed_start=200,
    canonical_seed_count=16,
)
```

## Repository map

```text
src/evox/program.py      safe expression trees and mutations
src/evox/worlds.py       ambiguous visible world + counterfactual probes
src/evox/evolution.py    deterministic GP + complete genealogy ledger
src/evox/modes.py        behavior matrix, SVD, deterministic two-mode clustering
src/evox/probes.py       Bayesian posterior + information-gain probe selection
src/evox/transfer.py     correct/wrong/scrambled/restart transfer conditions
experiments/run_v0.py    frozen sweep, aggregation and classification
results/                 immutable experiment receipts
tests/                   mechanism tests and destructive controls
```

# What this does and does not establish

The positive result is narrow:

```text
ambiguous observations
        ↓
evolution retains >1 executable mechanism
        ↓
counterfactual behavior vectors
        ↓
SVD / mode separation
        ↓
active probe identifies useful mode
        ↓
local evolution reuses it on a related task
```

It does **not** establish:

- general program synthesis;
- ARC solving;
- universal spectral "algorithms";
- that SVD will find useful modes in arbitrary evolutionary runs;
- that genealogy by itself reveals computation;
- that the minority mode would survive under a much longer optimizing run.

## Next gate

Do not jump to ARC yet.

First attack the actual V0 weakness: preserve alternatives **without using counterfactual probe outputs**. Candidate controls include structural novelty, genealogy-aware niching, and neutral archives. The experiment should ask whether those mechanisms raise minority-mode mass and valid-seed rate while leaving mode extraction and active probing blind to family labels.

If that survives, then the same interfaces can be attached to generated ARC-style program tasks or symbolic-regression problems.
