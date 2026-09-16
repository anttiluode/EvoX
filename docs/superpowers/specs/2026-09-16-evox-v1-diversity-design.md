# EvoX v1 — Probe-Blind Diversity Preservation

## Purpose

V0 showed that behaviorally distinct executable procedures can be recovered, actively identified, and reused **if both procedures survive evolution**. Its main weakness was purification: at the smaller 128×16 budget only 10/16 seeds retained separable behavior, and even the larger-budget passing run retained a minority mode averaging only about 4.4% of viable programs.

V1 asks one narrower question:

> Can a preservation rule that never sees counterfactual probe outputs keep alternative procedures alive at meaningful population mass without degrading the visible task?

This is a prerequisite gate before ARC-style or symbolic-regression tasks.

## Information firewall

Preservation strategies may use only:

- visible-task MSE;
- expression-tree syntax and structural features;
- ancestry/root-lineage identity;
- program age/history;
- deterministic RNG state.

They may **not** use:

- counterfactual-pool outputs;
- SVD embeddings or cluster labels;
- post-hoc direct/relational labels;
- active-probe information gain;
- follow-up transfer performance.

Counterfactual behavior is evaluation-only and is computed after evolution has finished.

## Fixed world and budget

Reuse V0's ambiguous visible task unchanged:

- direct mechanism: `x0`;
- relational mechanism: `x1*x2`;
- visible examples satisfy `x0 = x1*x2`.

Use the deliberately difficult V0 budget:

- population: 128;
- generations: 16;
- elites: 24;
- max depth: 3;
- viable MSE: `1e-12`.

No method receives extra program evaluations.

## Strategies

### 1. Baseline

The existing V0 selector: rank by visible MSE, then node count/source/id, deduplicate syntax, keep the first `elite_count`.

### 2. Structural novelty

Keep half the elite slots by the ordinary baseline ranking. Fill the remaining slots with programs selected by deterministic farthest-point sampling in a **syntax feature space** among candidates tied at the best visible MSE before falling back to worse candidates.

Syntax features are fixed and probe-blind:

- node count;
- tree depth;
- counts of terminals `x0`, `x1`, `x2`;
- count of constants;
- counts of `neg`, `add`, `sub`, `mul`.

Features are standardized within the candidate set. Farthest-point selection maximizes distance to already selected elites; ties use the normal deterministic rank key.

### 3. Root-lineage niching

Every generation-0 individual defines a root lineage. Descendants inherit the same root id. Elite selection first takes the best currently available program from each distinct root lineage in deterministic rank order, cycling across roots until no unseen roots remain, then fills remaining slots by the ordinary baseline rank.

The mechanism preserves ancestry diversity, not behavioral diversity.

### 4. Neutral archive

Maintain a bounded archive of unique programs that have achieved viable visible MSE at any previous generation. Archive membership is keyed only by source syntax. The archive is a deterministic reservoir capped at 64 entries.

At each generation, reserve 25% of elite slots for archive reinjection. Archive entries are chosen by deterministic round-robin over insertion order; the remaining elite slots come from the ordinary baseline selector. Reintroduced archive entries are re-evaluated and count inside the same fixed population/evaluation budget.

The archive therefore remembers neutral successful procedures without knowing whether they are direct-like or relational-like.

## Evaluation after evolution

After a run completes, use the unchanged V0 counterfactual pool and mode extractor. For every strategy/seed report:

- viable program count;
- whether counterfactual behavior has nonzero variance;
- post-hoc mode purity;
- cluster sizes;
- **minority-mode fraction** = `min(cluster_sizes) / sum(cluster_sizes)`;
- retained rank and explained variance;
- ancestry transition rate;
- best and median visible-task MSE.

For valid two-mode seeds, run the unchanged active-vs-random probe and correct/wrong/scrambled/restart transfer tests from V0.

## Experimental phases

### Pilot

Use seeds 300–307 only to debug implementation and choose fixed strategy hyperparameters. Pilot seeds never contribute to the canonical classification.

### Canonical

Use disjoint seeds 400–431 (32 seeds) for all four strategies with a common seed tape.

Each strategy receives identical population size, generations, mutation code, visible examples, and evaluation budget.

## Predeclared classification

Classify each preservation strategy independently against baseline.

A strategy is `PRESERVES_MODES` only if on the canonical sweep:

1. valid two-mode seed fraction is at least **0.85**;
2. mean minority-mode fraction across valid seeds is at least **0.15**;
3. mean post-hoc mode purity is at least **0.95**;
4. median best visible MSE is no worse than baseline by more than `1e-12`;
5. active probing still uses at least **0.5 fewer censored probes** than random on average;
6. correct-mode transfer mean censored evaluations remain at most **0.75×** fresh restart, with success rate no more than 0.05 below restart.

`SURVIVES_BUT_THIN` means conditions 1, 3, 4, 5, and 6 pass but minority-mode mass is below 0.15.

`DIVERSITY_HURTS_FITNESS` means diversity improves but condition 4 fails.

`NO_PRESERVATION_GAIN` means the strategy does not materially clear the V0 bottleneck.

V1 as a whole is positive only if at least one predeclared probe-blind strategy is `PRESERVES_MODES`. All strategies are reported regardless of outcome.

## Architecture

- `src/evox/program.py` — add deterministic syntax feature extraction and tree depth.
- `src/evox/evolution.py` — add root ids, preservation strategy configuration, archive state, and strategy-specific elite selection while preserving the existing `evolve` default behavior.
- `experiments/run_v1_diversity.py` — matched four-strategy sweep, post-hoc mode audit, downstream V0 probe/transfer audit, aggregate classification.
- `tests/test_diversity.py` — information-firewall and selector tests.
- `results/v1_diversity.json` — frozen canonical receipt after pilot parameters are fixed.
- `docs/V1_DIVERSITY.md` — interpretation and claim fence.

## Destructive controls

Tests must establish that:

1. baseline behavior is unchanged when no strategy is requested;
2. structural novelty depends only on syntax features and visible fitness;
3. row/order permutations do not change deterministic selection except tie-equivalent ids;
4. root ids are inherited correctly through clone and mutation;
5. archive entries come only from visible-viable programs and never inspect counterfactual data;
6. all strategies consume exactly the same number of evaluations;
7. a deliberately syntax-diverse but behaviorally collapsed population is **not** counted as a V1 success;
8. downstream mode extraction/probe/transfer code is unchanged from V0.

## Claim boundary

A V1 pass would show only that a probe-blind preservation mechanism can maintain multiple counterfactually distinct procedures in this controlled program family at useful population mass. It would not establish general algorithm discovery, universal niching, or ARC transfer.
