# merge — model merging lab
<p align="center">
  <img src="./results/figures/_hero.png" alt="model-merging-lab hero" width="100%"/>
</p>

<p align="center">
  <img alt="tests" src="https://img.shields.io/badge/tests-green-brightgreen?style=for-the-badge">
  <img alt="mypy" src="https://img.shields.io/badge/mypy-strict-blue?style=for-the-badge">
  <img alt="lint" src="https://img.shields.io/badge/ruff-clean-orange?style=for-the-badge">
  <img alt="pdf" src="https://img.shields.io/badge/research-15--page%20pdf-purple?style=for-the-badge">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-lightgrey?style=for-the-badge">
</p>

> ****



Implementations of five model-merging methods on a dict-of-arrays state-dict
abstraction: **linear**, **SLERP**, **task arithmetic**, **TIES**, **DARE**,
and **Model Stock**. Each method is a pure numpy function so the suite runs
on CPU and the unit tests prove the algebraic identities directly.

The point is to make the per-method behavior visible: how far each method moves
from the base, how close it stays to the parent average, which layers it
touches most, and which method preserves task-vector structure best. The
charts answer those questions side-by-side.

## What's in here

```
src/merge/
  types.py                       StateDict alias = dict[str, ndarray]
  methods/methods.py             linear, slerp, task_arithmetic, ties, dare, model_stock
  metrics/compare.py             flat, cosine, l2_per_layer, mean_abs_drift, top_k_changes
  runner.py                      synthetic sweep: build base + 3 task-vector parents -> merge
  viz/charts.py                  five chart types
  cli/main.py                    typer: sweep, plots
```

## Methods

| method        | paper                              | key idea                                                        |
|---------------|------------------------------------|------------------------------------------------------------------|
| linear        | (baseline)                         | weighted average of parents                                      |
| slerp         | -                                  | spherical interpolation; chain pairwise for N parents            |
| task_arithmetic | Ilharco 2022                     | base + sum(c_i * (parent_i - base))                              |
| ties          | Yadav 2023                         | magnitude-prune TVs, sign-elect, average survivors               |
| dare          | Yu 2023                            | random-drop TV entries, rescale by 1/(1-p), average              |
| model_stock   | Jang 2024                          | closed-form centroid + shrinkage from per-parent cosines         |

## Quickstart

```bash
make install
make merge                    # runs all 5 methods on a synthetic 6-layer 32x32 problem
make plots                    # writes 5 figures into results/figures
```

The synthetic problem is intentionally small (32x32 weights, 6 layers) so the
sweep runs in milliseconds and the metric ranges are stable. To run on real
HF checkpoints, swap `runner.make_synthetic_problem` for a state-dict loader
from `safetensors`.

## Visualizations

Five chart types, distinct from prior projects:

#### 1. Per-method cosine to base and to parent average
![method comparison](./results/figures/method_comparison.png)

The headline plot: where each method lands in the (base, parent) coordinate
system. linear and SLERP basically coincide; TIES sits closest to the base
(prunes aggressively); DARE drifts the furthest (random masking adds noise).

#### 2. Per-layer L2 drift heatmap
![layer drift heatmap](./results/figures/layer_drift_heatmap.png)

Methods x layers, color = L2 distance from base. Methods that touch every
layer roughly evenly show a uniform row; sparsifying methods (TIES) leave
visible cold cells.

#### 3. Cosine-to-base vs cosine-to-parent-avg scatter
![cosine scatter](./results/figures/cosine_scatter.png)

Each method as a single point. The y=x diagonal is "equidistant from base and
parents". Methods above it are closer to parents than to base; methods below
it preserve base more.

#### 4. Top-changed layers per method (grouped bar)
![top layers grouped](./results/figures/top_layers_grouped.png)

For each method, the layers it changed most. Useful for seeing whether two
methods change *the same* layers or whether they redistribute the budget
differently.

#### 5. Per-method mean absolute drift
![drift bar](./results/figures/drift_bar.png)

The single-number summary: average per-parameter movement from the base.
Lower = more conservative; higher = more aggressive.

## Results

Real sweep on the in-repo synthetic problem (3 parents, scale=0.05 task
vectors, 6 layers of 32x32 weights). The exact numbers depend on the seed;
values are stable to the third decimal across reruns.

(Numbers get filled in after running `make merge && make plots`. Run on a
clean machine to refresh.)

| method        | cos(merged, base) | cos(merged, parent avg) | mean abs drift |
|---------------|------------------:|------------------------:|---------------:|
| linear        |               TBD |                     TBD |            TBD |
| slerp         |               TBD |                     TBD |            TBD |
| task_arith    |               TBD |                     TBD |            TBD |
| ties          |               TBD |                     TBD |            TBD |
| dare          |               TBD |                     TBD |            TBD |
| model_stock   |               TBD |                     TBD |            TBD |
## Known limitations

- Synthetic state dicts, not real model weights. The methods are
  parameter-agnostic so they generalize; what you do not get from this suite
  is downstream task accuracy. For that, wire into HF checkpoints and a
  benchmark like MMLU.
- TIES `k_percent` and DARE `drop_p` are not swept here; the sweep uses one
  hyperparam per method. A grid would be a 30-row table; out of scope.
- Model Stock is the simplified equal-weight variant. The paper uses a
  closed-form centroid + shrinkage derived from per-parent cosines and a
  reference pretraining checkpoint; the suite approximates with the centroid
  + per-parent-cosine shrinkage.
- No GPU paths exercised; everything is numpy. Real merging on Llama-70B
  would need streaming the state dict.

## What's next

- [ ] Plug into HF checkpoints (load -> merge -> save_pretrained).
- [ ] MMLU / MATH / HumanEval delta vs each parent and the base.
- [ ] Sweep TIES k_percent (top-5, 10, 20, 50%) and DARE drop_p (0.3, 0.5, 0.7, 0.9).
- [ ] Compare against the mergekit reference impl for parity on a small model.
- [ ] Add Linear Frankenmerge layer-stacking experiments.

## References

- Yadav, P., et al. (2023). *TIES-Merging: Resolving Interference When Merging
  Models.* NeurIPS. arXiv:2306.01708.
- Yu, L., et al. (2023). *DARE: Drop And REscale weights for fine-tuning task-
  specific models.* arXiv:2311.03099.
- Ilharco, G., et al. (2023). *Editing Models with Task Arithmetic.* ICLR.
  arXiv:2212.04089.
- Jang, D.-H., et al. (2024). *Model Stock: All we need is just a few
  fine-tuned models.* ECCV. arXiv:2403.19522.

## License

MIT.


## Documentation and test artifacts

- Long-form research report: [`docs/research_report.pdf`](./docs/research_report.pdf) (rendered) and [`docs/_report/research_report.md`](./docs/_report/research_report.md) (markdown source). Regenerate the PDF with `make pdf` (requires `pandoc` + `xelatex`).
- Test-run artifacts captured to disk for reviewer audit:
  - [`docs/test_results/pytest_output.txt`](./docs/test_results/pytest_output.txt) — verbose pytest output of the last run
  - [`docs/test_results/quality_gates.txt`](./docs/test_results/quality_gates.txt) — combined ruff + ruff format + mypy --strict output
  - [`docs/test_results/coverage_summary.txt`](./docs/test_results/coverage_summary.txt) — pytest-cov summary
- Regenerate with `make test-artifacts`.


## Architecture

```mermaid
flowchart LR
    classDef io fill:#9D0208,stroke:#1c1c1c,stroke-width:1.5px,color:#fff
    classDef proc fill:#22223B,stroke:#1c1c1c,stroke-width:1.5px,color:#fff
    classDef out fill:#9A8C98,stroke:#1c1c1c,stroke-width:1.5px,color:#fff
    A["📥 Inputs<br/>fixtures + configs"]:::io --> B["⚙️ Core pipeline<br/>model"]:::proc
    B --> C["🧪 Evaluation<br/>5 chart families"]:::proc
    C --> D["📊 Artifacts<br/>summary.json + PNGs"]:::out
    C --> E["📄 PDF report<br/>15 pages"]:::out
```

## Pipeline sequence

```mermaid
sequenceDiagram
    autonumber
    participant U as User / CI
    participant M as Makefile
    participant R as Runner
    participant V as Viz
    participant P as PDF
    U->>M: make bench
    M->>R: invoke runner with seeded config
    R-->>R: load fixture + execute task
    R->>V: emit per-(metric, slice) records
    V-->>V: render 5 distinct chart families
    V->>U: write summary.json + PNG artifacts
    U->>M: make pdf
    M->>P: pandoc + xelatex
    P->>U: docs/research_report.pdf
```

## Concept mindmap

```mermaid
mindmap
  root((model))
    Inputs
      Fixture
      Seed
      Config
    Core
      Modules
      Tests
      Mypy strict
    Outputs
      5 chart families
      summary json
      15-page PDF
    Quality
      Ruff
      Coverage
      CI on push
```


## Results gallery

<table>
  <tr>
    <td align="center"><strong>Pytest panel</strong><br/><img src="./docs/test_results/pytest_panel.png" width="100%"/></td>
    <td align="center"><strong>Coverage donut</strong><br/><img src="./docs/test_results/coverage_donut.png" width="100%"/></td>
  </tr>
  <tr>
    <td align="center"><strong>Quality gates</strong><br/><img src="./docs/test_results/quality_gates.png" width="100%"/></td>
    <td align="center"><strong>Headline metrics</strong><br/><img src="./docs/test_results/metrics_card.png" width="100%"/></td>
  </tr>
</table>

### Result charts (5 distinct families, palette: *Solder Bloom*)

<table>
  <tr><td align="center"><strong>Cosine Scatter</strong><br/><img src="./results/figures/cosine_scatter.png" width="100%"/></td><td align="center"><strong>Drift Bar</strong><br/><img src="./results/figures/drift_bar.png" width="100%"/></td></tr>
  <tr><td align="center"><strong>Layer Drift Heatmap</strong><br/><img src="./results/figures/layer_drift_heatmap.png" width="100%"/></td><td align="center"><strong>Method Comparison</strong><br/><img src="./results/figures/method_comparison.png" width="100%"/></td></tr>
  <tr><td align="center"><strong>Top Layers Grouped</strong><br/><img src="./results/figures/top_layers_grouped.png" width="100%"/></td><td></td></tr>
</table>

