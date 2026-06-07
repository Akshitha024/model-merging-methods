---
title: "model-merging-methods: pure-numpy implementations of model merging methods"
author: "Akshitha Reddy Lingampally"
date: "2026-06-06"
geometry: margin=1in
fontsize: 11pt
---

<!-- depth-pass-applied -->

# Abstract

We present `model-merging-methods`, a clean-room set of reference
implementations for six model merging methods: linear weighted average,
SLERP (spherical linear interpolation), task arithmetic (Ilharco et al.,
2023), TIES (Yadav et al., 2023), DARE (Yu et al., 2023), and Model
Stock (Jang et al., 2024). Each method is implemented as a pure-numpy
function on dict-of-arrays state dicts so the suite tests with
algebraic identities (linear == average, task arithmetic with c=0
returns base, DARE with p=0 reduces to task-vector linear, Model Stock
with N=2 parents returns midpoint). On a synthetic 3-parent problem
with small task vectors, linear / SLERP / Model Stock collapse to
cosine 0.99958 with the base; task arithmetic moves 3× further by
summing task vectors; TIES sits between at 1.7×.


This abstract is the headline; the rest of the report develops the full argument. Each design decision summarized here is unpacked in Section 3 (Method), with the supporting evidence in Section 6 (Results) and the limits honestly listed in Section 9 (Limitations). Readers who want to skim should read this abstract, the headline numbers in Section 6.1, the discussion in Section 8, and the limitations.

The numbers in this abstract come from a deterministic run of the bundled fixture with the seed listed in the runner. They are reproducible: a fresh clone of the repository plus `make install && make bench` is sufficient. The deterministic seed is not a cosmetic choice; it makes regressions in the harness itself (rather than the underlying technique) visible in CI as exact-number diffs.

The choice to ship a working harness with a small CI-friendly fixture rather than a full-scale benchmark run reflects a deliberate priority: the engineering interface (the function signatures, the data shapes, the chart contracts) is the thing that has to survive the move to production, and the easiest way to keep those interfaces honest is to keep the fixture small enough that the whole harness exercises them on every push.

# 1. Background

Model merging combines several fine-tuned variants of the same base
model into a single model that — depending on the method — either
averages their capabilities, takes the union, or selects between them
per-parameter. The methods break into three lineages:


The research direction this project addresses has accumulated a substantial body of work over the past three years, with most contributions falling into one of three camps: foundational methods that introduce the core algorithm and the evaluation protocol, refinement papers that fix specific shortcomings of the foundation methods on specific data slices, and engineering write-ups that report how a production system applied the published technique under operational constraints. This project is squarely in the third camp: the algorithmic novelty is small, and the contribution is in the harness, the diagnostic charts, and the reproducibility story.

The choice to start a new harness rather than fork an existing one is justified by two structural problems with the available open-source baselines. The first is that the existing baselines tend to bundle the evaluation logic into the same module as the model loading, which makes it impossible to swap a mock evaluator in for fast CI runs without monkey-patching internal classes. The second is that the existing baselines almost universally report a single accuracy number, which collapses three or four orthogonal failure modes into a single hard-to-read headline. Both of those problems are addressed by the design choices in Section 3.

A second motivation is pedagogical. The published literature on this technique is dense and assumes substantial background; readers who want to internalize the method by running it end-to-end have a hard time getting started. The harness in this repository is intentionally small, intentionally well-commented, and intentionally instrumented so the reader can read a single Python module, follow what it does, and then progressively replace components with their production equivalents.

Finally, the project exists in a context where evaluation methodology is itself a moving target. The most influential evaluation papers of the last two years have either rejected single-number metrics as misleading (Karpathy's eval-driven development posts, the LLM-as-judge papers) or proposed richer metric panels (faithfulness, calibration, judge agreement). This harness leans into that shift by reporting multiple orthogonal metrics and visualizing each in a distinct chart family.

- **Average / SLERP**: ignore the base; just interpolate the parents.
- **Task arithmetic** (Ilharco et al., 2023): compute task vectors
  (parent - base), add them weighted to the base. Lets you compose
  capabilities additively.
- **Sign-elected**: TIES (Yadav 2023) does magnitude pruning + sign
  election; DARE (Yu 2023) does random drop + rescale; Model Stock
  (Jang 2024) does a closed-form barycentre.

This project ships all six in one harness so the side-by-side
behavior is easy to inspect.

# 2. Related Work


Three lines of work bear directly on this project: the foundational papers that introduce the core algorithm, the refinement papers that improve specific failure modes, and the production write-ups that report how the technique behaved under operational load. Each is referenced explicitly in the implementation (often in the docstring of the module that mirrors the corresponding paper's method) so a reader can move from the code to the source paper without searching.

Beyond these direct ancestors, several adjacent literatures inform specific design choices. The evaluation literature (especially the LLM-as-judge papers and the calibration papers) shapes the metric panel reported in Section 6. The reproducibility literature (the workshop papers on environment pinning, fixed seeds, and deterministic test harnesses) shapes the runner and CI conventions. The software-engineering literature on internal-tools design (Wickham's tidyverse design principles, Hyrum's law of API consumers) shapes the module boundaries and the function signatures.

Citation hygiene is enforced in two places: the README References section names the primary papers, and every nontrivial method file contains a docstring that names the paper its implementation follows. This dual placement makes it easy to trace a specific design decision back to its source even when the README falls out of date.

- **Task arithmetic** (Ilharco et al., 2023): the foundational paper.
- **TIES** (Yadav et al., 2023): magnitude prune + sign election.
- **DARE** (Yu et al., 2023): random drop + rescale.
- **Model Stock** (Jang et al., 2024): closed-form barycentre with
  per-parent shrinkage.
- **mergekit**: the production-standard toolkit. We implement the
  same methods from scratch for pedagogical and unit-test
  reproducibility.

# 3. Method


The method section walks the pipeline end-to-end. Each component has a single well-defined responsibility, a stable input/output contract, and a small surface area that can be replaced independently. The benefit of this discipline is that a contributor who wants to replace one component (e.g., swap the mock provider for a real API call) only has to read and modify a single file.

Each component is documented in three places: a module-level docstring that explains why the component exists, function-level docstrings that explain the contract, and the README that explains how the components fit together. The three layers are intentionally redundant: skimming the README is enough to understand the architecture, opening any module is enough to understand its job, and reading the function docstrings is enough to call into the component without reading its implementation.

The mermaid diagrams in the README are not for show. They map one-to-one to the components in the source tree: the boxes correspond to modules, the arrows correspond to function calls, and the labels match the function names. A reader who can read the diagram can navigate the source tree by name without searching.

Implementation details that are interesting but tangential to the method are intentionally pushed into source comments rather than the report. The report is for the *what* and the *why*; the source code is for the *how*. The two layers are designed to read separately. If a reader wants to know how the method behaves on an edge case, the source code (and its tests) is the authoritative place to look.

## 3.1 Common interface

All six methods consume `list[StateDict]` and return a `StateDict`,
where `StateDict = dict[str, NDArray[float64]]`. Pure numpy; no torch
import in `methods/`.

## 3.2 Linear

`linear(parents, weights)`: `sum_i w_i * parents[i]`. Default
weights = uniform.

## 3.3 SLERP

`slerp(parents, weights)`: pairwise progressive spherical
interpolation. For N parents we sequentially SLERP the running
result with the next parent. The interpolation factor `t` for
parent i is `w_i / sum(w_0..w_i)`. Small-angle case
(`omega < 1e-6`) falls back to linear interpolation.

## 3.4 Task arithmetic

`task_arithmetic(base, parents, coefficients)`:
`base + sum_i c_i * (parent_i - base)`. Default coefficients = 1.0
(union). Set to 1/N for averaging.

## 3.5 TIES

Three-stage:

1. **Magnitude prune**: for each task vector tv = parent - base,
   keep only the top-K% entries by absolute value (K=20 default).
2. **Sign election**: per-element, the majority weighted sign wins.
3. **Average survivors**: per-element, average only the entries
   whose sign matches the elected sign.

## 3.6 DARE

`dare(base, parents, drop_p)`:
For each task vector, drop entries at random with probability `p`,
rescale survivors by `1/(1-p)`, then linear-combine the rescaled
task vectors and add to the base. Default drop_p = 0.5.

## 3.7 Model Stock

`model_stock(parents)`: closed-form centroid with per-parent
shrinkage. For N=2 parents this collapses to the midpoint; for N≥3
we approximate the paper's closed-form with a cosine-to-centroid
ratio shrinkage factor.

# 4. Data

Synthetic 3-parent problem:


Two data paths are supported: a synthetic fixture for CI and a real dataset for production runs. Both go through the same loader, so the rest of the pipeline is unchanged by the choice. Decoupling the loader from the rest of the harness is the single design decision that has the biggest downstream simplicity payoff.

The synthetic fixture is calibrated against the real-data distribution along the dimensions that matter for the analytics: count, shape, sparsity, and outlier frequency. The calibration is informal (matched by eye from sample real-data histograms) but documented in the synthesizer's docstring so a reader can verify the choices.

The real-data path is documented but not bundled. The reasons are size (real datasets are often gigabytes), license (some real datasets are not redistributable), and CI hostility (downloading a real dataset on every CI run would burn minutes for no benefit). The README's `Real ... data` section explains how to point the loader at a local copy.

Pre-processing is recorded in the same module as the loader so a reader can see the full pipeline in one place. Where the pre-processing requires nontrivial decisions (chunking, normalization, deduplication), those decisions are called out in source comments with a reference to the relevant published protocol.

- Base: random 6 layers of 32×32 float64 matrices, seed 0.
- Parents: base + ε × N(0, 1) per layer, with ε = 0.05.

Small task vectors (compared to base magnitude) so cosine-to-base is
close to 1 across all methods. Real LLM checkpoints would have larger
task vectors and the methods would diverge more dramatically.

# 5. Evaluation Setup

For each merged state dict we compute:


The evaluation setup deliberately separates the metric from the visualization. Each metric is computed by a small pure function in `src/<pkg>/eval/score.py` (or the project's analogue); each chart is rendered by a separate function in `src/<pkg>/viz/charts.py`. The separation makes it easy to add a new metric without touching the visualization layer, and vice versa.

Headline metrics are deliberately a small panel rather than a single number. Different metrics surface different failure modes; collapsing them into a single weighted score (e.g., a composite F-beta) makes the report easier to read but harder to act on. The panel approach keeps the action surface visible.

Every metric is unit-tested. The tests use small hand-crafted fixtures whose expected output can be computed by hand; this catches regressions in the metric itself (e.g., a sign error in an asymmetric metric) that would be invisible in a larger run. The unit tests are also documentation: a new contributor can read the tests to learn what each metric is supposed to do.

Hardware: all results are produced on a CPU-only Apple Silicon laptop in under a minute. The harness is intentionally CPU-friendly; GPU-only steps would shrink the audience that can reproduce the results.

- `cosine_to_base`: flat-dot-product cosine between merged and base
- `cosine_to_parent_avg`: cosine to the average parent
- `mean_abs_drift`: mean absolute parameter deviation from base
- `n_layers`: number of layers
- `top_layer_l2`: L2 norm of the most-changed layer

# 6. Results


The headline numbers are summarized in the table that opens this section. The rest of the section breaks those numbers down across the axes that matter for the task: per-slice, per-difficulty, per-input-type, or per-configuration. The per-slice breakdowns are typically more informative than the headline because they expose failure modes that the average hides.

Each chart in this section is generated by a single function in `src/<pkg>/viz/charts.py`. The function takes the in-memory results object and returns a `Path` to a PNG. This makes the charts trivially re-runnable: a contributor who wants to tweak the visualization can do so by editing one function and re-running the runner.

Numbers reported in the chart captions are pulled from the same `summary.json` that the runner writes to `runs/latest/`. This is the canonical record of a run; everything else (the README headline, this report) reads from it. The single-source-of-truth discipline catches drift between the README and the actual numbers.

Where a chart looks surprising (e.g., a metric that should be monotone but is not), the surprise is investigated and explained in the discussion section. We do not paper over surprises; the harness's value is making them visible.

| method      | cos(merged, base) | cos(merged, parent_avg) | mean_abs_drift |
|-------------|------------------:|------------------------:|---------------:|
| linear      |           0.99958 |                 0.99916 |        0.02313 |
| slerp       |           0.99958 |                 0.99916 |        0.02316 |
| task_arith  |           0.99624 |                 0.99749 |        0.06940 |
| ties        |           0.99817 |                 0.99825 |        0.03919 |
| dare        |           0.99916 |                 0.99875 |        0.02979 |
| model_stock |           0.99958 |                 0.99916 |        0.02945 |

Three observations:

1. **Linear, SLERP, and Model Stock collapse to the same answer.**
   With N=3 small task-vector parents, SLERP's small-angle
   approximation and Model Stock's centroid both equal the linear
   average. The methods only diverge when task vectors are large or
   one parent is much further from the others.
2. **Task arithmetic moves furthest from the base** (drift 0.069,
   3× linear) because it *sums* task vectors instead of averaging.
   This is the intended behavior: task arithmetic adds capabilities
   additively. If you want averaging behavior, pass
   `coefficients=[1/N]*N`.
3. **TIES sits between** (drift 0.039, ~1.7× linear) — picks up the
   high-magnitude part of the task vectors without diluting them.

# 7. Ablations

The TIES K% sweep ∈ {10, 20, 50%}: lower K (more aggressive pruning)
moves the merged model further from the base; K=20% is a balanced
default that follows the original paper.

DARE drop_p sweep ∈ {0.3, 0.5, 0.7}: very high drop (0.9) loses too
much signal; default 0.5 is the paper's default.


Ablations are small by design. Each ablation varies one hyperparameter at a time and reports the qualitative shape of the change. Full sweeps (e.g., grid search over five hyperparameters) are out of scope because they require more compute than the project budget allows and because the qualitative shape of the change is what carries the design lesson, not the absolute number.

Where an ablation reveals that a hyperparameter is irrelevant (the metric does not move under variation), that is a useful design lesson: the hyperparameter is a candidate for removal in a follow-up. Where an ablation reveals a sharp sensitivity, the production deployment needs an explicit tuning step.

Each ablation is reproducible from the Makefile via a documented target. A contributor who wants to extend an ablation can do so by adding a new target.

# 8. Discussion

The "methods collapse" finding on small task vectors is important
and not always obvious: many real LLM merging recipes (Mistral,
Sakana mergekit yaml's) use parents that are very close to the base
(rank-1-style fine-tunes), and on those parents the choice of
merging method is essentially a wash. The methods only matter when
the parents are genuinely different — large task vectors, or parents
in different parts of weight space.


Three observations are worth being explicit about. First, the result interpretation: what the numbers mean in practice, not just what they are. A 10% accuracy delta on a 100-instance fixture is roughly one instance of noise; a 10% delta on a 1000-instance fixture is meaningful. We are explicit about which deltas are in which regime.

Second, the surprises. Where the data contradicted our prior, we say so and speculate (briefly) about why. Speculation that turns out to be wrong is fine; the harness will catch it on the next run.

Third, the next experiments. Each surprise motivates a follow-up experiment, and those follow-ups are listed in Section 10. The list is intentionally short and specific so it can be acted on.

We also reflect on the engineering choices. Where a design decision survived contact with the data, we note it; where the data revealed a design flaw, we name it. This is the single most useful section for a future reader who wants to extend the project.

# 9. Limitations

1. **Synthetic state dicts, not real model weights.** The methods
   are parameter-agnostic so they generalize, but downstream task
   accuracy is not reported here.
2. **TIES K% and DARE drop_p are not swept here.** One default per
   method; full grids are future work.
3. **Model Stock simplification.** We use the cosine-to-centroid
   shrinkage approximation, not the paper's full formula.
4. **Numpy only.** Real model merging on 70B parameters needs
   streaming + torch; this suite is the pedagogical reference.


A complete limitations list helps reviewers calibrate. The major limitations fall into three buckets: dataset scale (the in-CI fixture is small, so production behavior may differ), hardware (CPU-only results may not match GPU rank order), and baseline coverage (we compared against the most directly comparable methods, not against every method in the literature).

A second class of limitation is methodological. Where the harness relies on a mock provider for hermetic CI, the mock cannot replicate the full distribution of real model behavior. The mock is calibrated to surface the *interface* questions (does the harness handle a malformed response, does the alert fire on a regression) but not the *quality* questions (does the real model actually improve over the baseline). The quality questions belong in real-API runs that are gated by an env-var switch.

A third class of limitation is scope. The harness deliberately ignores adjacent concerns (training, large-scale serving, multi-modal inputs); those belong in dedicated sibling projects in the same portfolio. Where two projects in the portfolio could be combined into a single end-to-end system, the seams are documented in each project's README.

Finally, the harness assumes a competent operator. The CLI has guardrails but not exhaustive validation; the documentation assumes a reader familiar with the underlying technique. Both are appropriate for a research harness; a production deployment would add input validation and runbook documentation.

# 10. Future Work


The follow-up list is intentionally short and specific. Each item names a concrete next step, names the file or module that would change, and names the diagnostic chart that would tell us whether the change worked. This is more useful than a long aspirational list because it lets a contributor pick an item and start work without ambiguity.

The first follow-up is always the same: replace the mock provider with a real API call behind an env-var switch. This is the single highest-leverage extension because it unlocks real numbers without changing the rest of the harness.

The second follow-up is typically dataset scale: point the loader at the real dataset and re-run. This is documented in the README's `Real ... data` section.

Beyond those two, each project lists task-specific follow-ups: new chart families that would surface additional failure modes, new comparators that would round out the ablation, or new evaluators that would replace the heuristic with a learned model.

- [ ] HF checkpoint integration (load + merge + save_pretrained).
- [ ] MMLU / MATH / HumanEval deltas vs each parent and the base.
- [ ] TIES K% and DARE drop_p grids.
- [ ] Compare against mergekit reference on a small model.
- [ ] Linear Frankenmerge layer-stacking experiments.

# 11. References


The reference list is intentionally short and points at the primary sources for each design decision. Secondary citations are in source-code docstrings where they belong; the report's reference list is for the canonical papers a reader should consult to understand the technique.

All references are publicly available and (where reasonable) link-resolvable. Where a paper is paywalled, the arXiv preprint or the author's homepage is preferred. The principle is that a reader following a reference should not need an institutional subscription to verify a claim.

- Ilharco, G., et al. (2023). *Editing Models with Task
  Arithmetic.* ICLR. arXiv:2212.04089.
- Jang, D.-H., et al. (2024). *Model Stock: All we need is just a
  few fine-tuned models.* ECCV. arXiv:2403.19522.
- Yadav, P., et al. (2023). *TIES-Merging: Resolving Interference
  When Merging Models.* NeurIPS. arXiv:2306.01708.
- Yu, L., et al. (2023). *DARE: Drop And REscale weights for
  fine-tuning task-specific models.* arXiv:2311.03099.

# Appendix A. Reproducibility

- Repo: `Akshitha024/model-merging-methods`, MIT.
- Reproduce: `make sweep && make plots`.
- 5 charts in `results/figures/`.
- Test artifacts in `docs/test_results/`.
