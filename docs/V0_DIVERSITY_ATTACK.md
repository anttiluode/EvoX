# V0 diversity attack

## What failed

The first disjoint canonical run used seeds 100–115 with population 128, 16 generations and 24 elites. Only 10/16 seeds retained nonzero counterfactual behavioral variance. Six seeds converged to programs that were all behaviorally equivalent under the probe pool.

The frozen classification threshold required at least 70% valid seeds. V0 therefore remained `INSUFFICIENT_DIVERSITY` even though active probing and transfer were strong on the valid subset.

## Predeclared V0b change

V0b attacked only that bottleneck. Before evaluating the new seed range, the change was fixed as:

- seeds 200–215, disjoint from pilot seeds 0–7 and V0 seeds 100–115;
- population 192;
- 24 generations;
- 32 elites;
- identical grammar, viable threshold, SVD, clustering, probe policy, transfer task, controls, and classification thresholds.

No threshold was lowered.

## Result

V0b retained separable behavior on 13/16 seeds (81.25%), crossing the unchanged 70% gate. On valid seeds, post-hoc mode purity was 1.0. Active probing reached the confidence threshold in one probe on average versus two censored probes for random selection.

Correct-mode transfer reached an exact follow-up solution with 100% success and 22.1 mean censored evaluations, compared with:

- wrong mode: 50.0% success, 388.0 evaluations;
- scrambled warm start: 76.9% success, 178.2 evaluations;
- fresh restart: 53.8% success, 306.5 evaluations.

Classification: `PASS_ACTIVE_MODES`.

## Important limitation exposed by the same receipt

The minority cluster is small. Across valid V0b seeds, its mean fraction of viable programs is approximately 4.4%, with a range of roughly 1.6% to 8.2%.

So V0b demonstrates preservation of a *recoverable alternative*, not a balanced ecological coexistence of algorithms. The next experiment should increase robustness and minority mass without giving evolution access to the counterfactual probe outputs used later for identification.
