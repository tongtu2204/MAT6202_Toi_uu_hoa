# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A workspace for a **MAT6202 (Advanced Optimization)** course presentation. It is *not* a churn-prediction project — churn is only a **test bed**. The object under study is the **optimization algorithm**, not the customer. Every artifact must answer an optimization question ("which step rule converges faster and why"), never an ML question ("is the churn model accurate"). If AUC/accuracy/confusion-matrix ever takes center stage, the work has drifted off-axis; those belong only in an appendix.

The intended deliverables are (1) a Python feature-engineering pipeline producing the design matrix `X` (`churn_opt/`) and (2) a from-scratch benchmark of four optimizers on regularized logistic regression (`optim/`) — **both implemented**; the deck is `presentation_v2/main.tex`. `outline.txt` and `feature engineer.txt` (both Vietnamese) are the authoritative spec; read them before changing the pipeline.

## Running the pipeline

```bash
python run_pipeline.py                      # defaults: T=2019-09-30, K=6mo, H=3mo
python run_pipeline.py --lam 1e-3 --no-poly # tweak conditioning report / skip poly
```

Takes ~90s (dominated by reading the 16M-row activity CSV). Writes to `artifacts/`:
`<variant>.npz` (`X_train`, `X_test`, `y_train`, `y_test`), `<variant>.features.txt`, and `diagnostics.txt` (the L/μ/κ/rank slide numbers). Three variants are produced — see below.

Current run: **K=3 months, three stacked snapshots** (T ∈ 2019-03-31 / 06-30 / 09-30), no hold-out → **n = 75,026 rows** (one per customer×snapshot), churn 10.8%. ridge **d=415, full rank** (L=14.07, κ_bound=1.41e4, κ*=3646, μ*=1.00e-3 at λ=1e-3), lasso d=519 (rank 506, redundant on purpose), poly d=428 full rank, ridge_raw d=415 but numerically **rank 376** — κ≈2.7e19 exceeds float64, so λI is invisible and its Hessian is singular. Measured cost at this size: Newton ~0.8 s/iteration (15 iterations to machine precision), GD/AGD ~26 ms/iteration.

## Pipeline architecture (`churn_opt/`)

Modules map 1:1 onto the spec's execution order; read them in this sequence:
- `config.py` — `WindowConfig` (snapshot/observation/outcome windows) and `FeatureConfig` (pruning thresholds, poly knob, λ). All tuning lives here.
- `loaders.py` — per-table load + **id normalization** (`_norm_id` zero-pads to 8 chars: xlsx stores ids as ints with leading zeros stripped; CSVs store padded strings) + per-file date parsing.
- `labeling.py` (Step 0) — eligible population + leakage-free `y`.
- `aggregate.py` (Step 1) — folds each table to one row/customer on the observation window. `_slope_momentum` computes monthly trend features (missing month = 0). `TRANS_LV1` pivot (`TXG_*` columns) is the main dimensionality driver.
- `build.py` (Steps 2–7) — assemble → `_clean` (impute + log1p) → `_encode` (one-hot drop_first) → (optional split, off by default) → `StandardScaler` → `_corr_prune` + `_vif_prune` → `_add_polynomial`. Produces the three variants. With `use_holdout=False` the `X_test`/`y_test` slices in the `.npz` are empty but kept, so `optim.data.Dataset` and every downstream consumer are unchanged.
- `diagnostics.py` — `conditioning(X, λ)` returns L, μ, κ_upper, rank.

## Documentation (`docs/`)

A full replication report (Vietnamese) lives in `docs/`:
- `feature_engineering_report.md` — the source report, with embedded PNG diagrams **and** Mermaid source blocks (renders in VS Code/GitHub/Obsidian).
- `report.tex` → `report.pdf` — the typeset PDF. **Rebuild requires XeLaTeX** (Vietnamese via `fontspec`/Times New Roman; `underscore` package lets column names keep raw `_`). Note: `titlesec`/`tcolorbox`/`sectsty`/`mdframed` are NOT installed in this TeX Live — the preamble deliberately avoids them (custom note/warn boxes are built from `xcolor`+`\colorbox`). Build: `cd docs && xelatex report.tex` (twice, for the TOC).
- `make_diagrams.py` — regenerates the four flow diagrams into `docs/diagrams/*.png` using matplotlib (font Arial for Vietnamese; no node/mermaid needed). Both the markdown and the PDF reference these PNGs, so rerun this after changing a diagram, then rebuild the PDF.

To refresh everything after a pipeline change: `python docs/make_diagrams.py` → `cd docs && xelatex report.tex && xelatex report.tex`.

### The three design-matrix variants
- **`ridge`** — correlation+VIF pruned **then rank-repaired by pivoted QR**, full rank → for Ridge + Newton (needs invertible Hessian). ~415 features.
- **`lasso`** — fuller matrix, redundancy kept on purpose so L1/ISTA/FISTA has something to zero out. ~519 features.
- **`poly`** — ridge core + degree-2 interactions on a small continuous core. **No longer the κ knob**: at d=415 the 36 interaction columns leave κ unchanged to four digits (κ_poly = κ_ridge = 1.407e4), so the AGD-vs-GD story rests on **λ** instead — `optim.experiments.kappa_sweep` spans κ over two decades by sweeping λ, which is cheaper and cleaner than widening the matrix. `poly` is kept only as an illustration of what a wider d costs Newton. The poly core is restricted to **continuous** columns only: squaring/interacting a 0/1 dummy reproduces an affine function of it → exact collinearity → singular Hessian (this bug was hit and fixed; don't reintroduce it).

## Core technical decision (do not change without reason)

Model = **L2-regularized (Ridge) Logistic Regression**, chosen because the hardest constraint is *implementing Newton's method by hand*. The objective (per `feature engineer.txt` / `outline.txt`):

```
f(w) = (1/n) Σ [ -y_i log p_i - (1-y_i) log(1-p_i) ] + (λ/2)||w||²,   p_i = σ(w·x_i)
∇f(w)  = (1/n) Xᵀ(p - y) + λw
∇²f(w) = (1/n) Xᵀ S X + λI,   S = diag(p_i(1-p_i))
```

This is the one place logistic regression appears; after defining `f(w)` the presentation is entirely convergence curves. Key theory quantities to compute *from the actual data* and put on a slide: `L = (1/4n)λ_max(XᵀX) + λ`, `μ`, `κ = L/μ`. Newton on this objective = IRLS.

**The intercept is unpenalized**, so the ridge term is really `λ·diag(reg)` with a zero on the intercept and `∇²f = (1/n)XᵀSX + λ·diag(reg)`. `μ ≥ λ` therefore holds only on the penalized coordinates — see "Measurement hygiene" below before putting `μ ≥ λ` on a slide.

Four algorithms to implement and compare: **GD, Accelerated GD (Nesterov), Newton (with damping/line search), SGD**. Optional bonus: **subgradient / ISTA / FISTA / coordinate descent** for the L1 Lasso extension (subgradient is the control that makes the proximal case: `sign(w)` is ±1 at every nonzero coordinate, so coordinates overshoot zero instead of landing on it and exact zeros never appear — soft-thresholding is not a faster route to the same answer, it is the part that produces sparsity at all; its η₀ is swept, not set by hand), and **AdaGrad/RMSprop/Adam** (+AdamW/AMSGrad flags) in `optim/optimizers/adaptive.py` — one `adaptive()` driver, full-batch by default so its curves drop straight into the same money plot; run via `run_adaptive.py`. (Numbers for this family at d=415 are produced by `run_adaptive.py`; the d=39 result was Adam 413 iterations vs AGD 800 at equal O(nd) cost, with RMSprop never converging because its effective step does not vanish.) The unifying thesis is the **per-iteration-cost vs convergence-rate trade-off**, governed by `κ` and the step-size strategy.

## Non-obvious correctness constraints

These are failure modes the docs call out explicitly:

- **Standardize X (z-score) — mandatory.** Raw features span ~6 orders of magnitude (AGE vs TRANS_AMOUNT in VND); without scaling `κ` explodes and the GD-vs-AGD-vs-Newton comparison becomes meaningless.
- **One-hot with `drop_first=True` — mandatory.** Keeping all `k` dummies + intercept gives perfect collinearity → `XᵀX` singular → **Newton literally crashes** (inverting a singular matrix). GD/AGD won't error but the solution is undefined.
- **`log1p` heavy-tailed money/count columns** before scaling — outliers inflate `λ_max(XᵀX)` → large `L` → artificially slow GD.
- **No NaNs may survive.** A single NaN turns gradient/objective into NaN and breaks every iteration. Fill 0 for absent counts/amounts and add a `HAS_X` flag.
- **No train/test split by default.** This is an optimization study: `f(w)` is defined by whatever `(X, y)` it is handed, and nothing we measure (`L`, `μ`, `κ`, iterations, time, `f - f*`) needs a hold-out — splitting only costs 25% of `n`. `FeatureConfig.use_holdout=False` is the default; `run_pipeline.py --holdout` restores the old 75/25 behaviour, and is only needed for out-of-sample AUC/F1. When the split IS on, it must happen *before* the scaler/encoder/imputer are fit (leakage). Appendix classification metrics are computed in-sample by `run_classification_metrics.py` and labelled as such.
- **VIF/correlation pruning** to keep Hessian well-conditioned — but keep *two* versions of `X`: a clean pruned one for Ridge+Newton, and a fuller (intentionally redundant) one for Lasso so ISTA/FISTA has something to zero out.
- **Comparing to sklearn:** sklearn minimizes a **sum** (no `1/n`) with penalty `(1/2)wᵀw` and `C = 1/(λn)` (our objective averages the loss, so matching the penalty/loss ratio gives `1/(nλ)`, **not** `1/(2λn)`; see `optim.objective.LogisticObjective.sklearn_C`). Match its objective form exactly and compare **objective value, not accuracy**. Default solver baseline is `lbfgs`.
- **Plot `log(f - f*)` on the y-axis** (linear scale hides linear/quadratic rates). `f*` comes from `optim/reference.py`, never from an inline `min(f_history)` — see below.
- Deliberately run a **divergent config** (`η > 2/L`) to produce the "objective flies to infinity" teaching plot.

## Headline measurements at n=75,026 / d=415 (λ=1e-3)

Several conclusions FLIPPED when the matrix grew from d=39. Do not copy older numbers forward.

- `f* = 0.212258252065`, certified: ‖∇f(w*)‖ = 7.9e-15, μ* = 1.00e-3 ⇒ f* − f_true ≤ 3.1e-26.
- **The money plot is `run_tuning.py money7`, not `run_stage.py money`.** It draws **all seven hand-tuned configurations** of the deck's §4.16 summary table on one pair of axes, at one common stopping rule (‖∇f‖ < 1e-10, cap 3000), every constant read from `tuning.json`. `run_stage.py money`'s 4-curve figure (`money_iter/time_ridge.png`) is no longer referenced by the deck — its Newton curve was the f* reference run, i.e. library defaults, which broke the "everything here was hand-tuned" claim. Measured, seven configs: Newton bt (ρ=0.5, c=0.2) **8 iter / 6.0 s** · Newton t=1 10 / 7.2 s · SGD 50 ep / 11.4 s · AGD β-const 1202 / 37.5 s · GD t=0.4 >3000 / 77.7 s · AGD (k−2)/(k+1) >3000 / 99.1 s · GD backtracking >3000 / **583.2 s**. Only 3 of 7 meet the tolerance.
- **The hand-tuned Newton backtracking (ρ=0.5, c=0.2) BEATS pure Newton at a practical tolerance (8 vs 10 iterations) yet cannot produce f*.** Both facts are measured and both are on slides: at tol=1e-10 it wins; at the tol=1e-14 the reference needs, c=0.2's strict Armijo test fails on floating-point noise near w*, the step is backtracked toward 0, and it hits the 100-iteration cap. §4.2 is a dedicated slide for f* precisely so the reference run has a home that is not "an untuned competitor".
- **Money plot** (superseded by money7; kept for the 4-curve view):  Newton 15 iter / 10.4 s · SGD 51 epochs / 10.5 s · AGD 1203 / 31.5 s · GD >3000 (cap) / 77.8 s. Newton costs 26.4× an AGD iteration (0.692 s vs 0.0262 s) but needs 80× fewer. Newton and SGD are now a dead heat on wall-clock.
- **AGD has two momentum schemes** (`optim/optimizers/agd.py`, `scheme=`): `"const"` (default) is the strongly-convex β=(√κ−1)/(√κ+1) at step 1/L; `"k"` is Nesterov's merely-convex β_k=(k−2)/(k+1), whose momentum carries **no problem constant**, so its step `t` is a genuinely free parameter and can be swept by hand (`run_tuning.py agd-k`, chosen t=0.3). Measured head-to-head at 3000 iterations: `"k"` is **710× ahead at 150 iterations** (3.1e-5 vs 2.2e-2) because momentum starts at 0 and ramps, but it **never converges** (6.3e-10 at the cap) since O(1/k²) is sublinear; the const-β curves cross it around iteration 600–900 and drop vertically.
- **"Changing t breaks the (t, β) pair" was too strong — measured.** The swept pair (t=0.3 with β recomputed from a fictitious L_eff=1/t=3.33 < L=14.07) converges in **1202 iterations / 32.2 s**, vs 2400 / 66.8 s at 1/L — 2.0× faster. What actually breaks AGD is changing t *per iteration while holding β fixed*, which is exactly what backtracking does. `optim.benchmark.agd_config()` feeds the swept step in as `L_eff` precisely so β is recomputed with it. The money plot now runs this pair; it rests on an L below the true one, i.e. outside the guarantee, so §5.2 quotes the 1/L run as the conservative number.
- **Hand-tuned steps depend on the budget — the fine sweeps run 1000 iterations, not 150.** AGD's best step moved 0.2 → **0.3** when the fine grid went from 150 to 1000 iterations (2.3e-15 vs 4.2e-13, 180× better); at 150 iterations 0.2 genuinely wins. Both momentum schemes now pick t=0.3, so §4.11 compares them at the *same* step and the gap is momentum alone (β constant crosses (k−2)/(k+1) at iteration **445**, converges at 1202). For both schemes 0.3 is the largest value *before the cliff* at 0.5 — the true optimum is inside (0.3, 0.5) and this grid cannot resolve it.
- **The coarse grid is `[0.01, 0.1, 0.5, 1, 2, 5]` at 20 iterations, shared by GD / AGD / AGD-k / Newton** (`run_tuning.COARSE_T`), so those four slides compare directly. It **misleads twice**, on purpose kept in the deck: for GD it ranks t=2 first (t=2 is 14× the 2/L threshold and merely caught in a low oscillation phase at iteration 20) and for AGD-k it ranks t=0.5 first — both refuted by the fine sweep. Newton's row now shows all three regimes including genuine death at t=2 and t=5 (Cholesky fails at iterations 15 and 5).
- **GD in the money plot runs at the hand-tuned t=0.4, not 1/L** — `optim.benchmark.gd_config()` reads `artifacts/tuning.json["gd_fixed"]["chosen"]` (fallback 1/L, and it says which it used). 1/L is the *safe* step, not the best one: at 1/L GD ends 3000 iterations at 8.1e-4, at t=0.4 at **1.2e-5** (66× better) — still not converged, which is the point: the GD–AGD gap is κ vs √κ, not a bad step. **AGD deliberately keeps 1/L**, because β=(√κ−1)/(√κ+1) is derived assuming that step; substituting the swept t=0.2 breaks the (t, β) pair. `experiments.adaptive_family` uses the same `gd_config` so its GD baseline matches — every method on that slide is then quoted at its own tuned step.
- **κ scaling** (`kappa_sweep`, 15k-row subsample): slopes GD **0.819**, AGD **0.413** — the *ratio* is **1.98 ≈ 2**, i.e. the κ-vs-√κ law holds almost exactly. At d=39 the ratio was only 1.31: the wider matrix made the central claim *sharper*, not just slower.
- **lbfgs now beats Newton** (4.49 s vs 7.52 s). At d=39 Newton dominated; at d=415 the O(nd²+d³) Hessian cost exceeds 225 cheap quasi-Newton iterations. This is the per-iteration-cost-vs-rate trade-off measured in one table, and it matches the predicted breakeven d* ≈ 139.
- **Backtracking helps GD but breaks AGD**: GD 3.3e-4 → 7.8e-8 for 2.9× the f-evals; AGD goes from converging in 2131 iterations to *not* converging in 5000 (gap 1.6e-2). Nesterov's β = (√κ−1)/(√κ+1) is derived *assuming* step 1/L — varying t per iteration breaks the (t, β) pair the proof relies on. Armijo ρ=0.5 is now clearly cheaper than ρ=0.8 (7,239 vs 18,206 f-evals), the reverse of the d=39 run.
- **Raw (unstandardized) design no longer merely slow — it is unusable**: κ ≈ 2.7e19 exceeds float64, λI becomes invisible, numerical rank 376/415, Newton cannot run. GD/AGD sit 0.434 from f* after 2000 iterations.
- **L1** (every method at its own hand-tuned step, `optim.benchmark.l1_config`): f* = 0.221818416; at an equal 8000-iteration budget CD attains it exactly, FISTA 1.5e-11, ISTA 1.2e-6, subgradient 1.4e-4 — the O(1/√k) < O(1/k) < O(1/k²) order shows up cleanly *even after* each method gets its best step. ISTA/FISTA/CD/saga now all agree on **125–126 of 519** nonzeros (tuning is what fixed ISTA: 148 nonzeros at 1/L → 126 at 7/L).
- **FISTA's usable step ceiling is LOWER than ISTA's — 4/L vs 7/L, measured.** `run_tuning.py l1` sweeps them separately (`l1_ista`, `l1_fista`) and `l1_config` returns `t_ista` and `t_fista` as distinct fields. Do not reuse one for the other: feeding ISTA's 7/L to FISTA gives 1.17e-2 with **346** nonzeros — worse than the textbook 1/L — because momentum accumulates the overshoot that plain ISTA absorbs. The cliff sits between 5/L and 6/L. Same shape as AGD being more fragile than GD in §4.12, and the deck now has a slide (7.5) for it.
- **`run_l1_benchmark` used to pick the subgradient η₀ itself** with an internal probe (it chose 1; the hand sweep chooses 2) and left ISTA/FISTA at 1/L, so slides 7.6–7.7 reported three runs at parameters no slide had chosen. `l1_config()` closes that, the same way `gd_config`/`agd_config` did for §4.
- **Pure Newton from a far start dies** at iteration 8 (‖w‖ = 5.1e6, Cholesky fails); damped needs 29.

## Running long jobs here

`run_revision.py` at this size is a multi-hour job and harness-tracked background jobs in this environment get killed after tens of minutes. Two things make that survivable:
- **`run_stage.py <stage>...`** runs one stage and merges into `artifacts/rev1_numbers.json` after each, so an interrupt costs at most one stage. Prefer it over `run_revision.py`.
- Stages under ~10 min run fine in the foreground. For longer ones, detach with Python's `subprocess.Popen(..., start_new_session=True)` — note **macOS has no `setsid`**, so a `setsid nohup ... &` shell incantation fails silently.
Measured stage costs are listed in `run_stage.py`'s docstring; `kappa-sweep` and `kappa-honesty` subsample to 15,000 rows (documented there) because they are statements about κ, which is a property of the feature geometry, not the sample size.

## Measurement hygiene (added after review; do not regress these)

- **`f*` is the value at the FINAL iterate, not `min(f_history)`.** `min` over a run's own history reports a different point from the one the run returned and shaves every other method's gap in our favour — it reads as cherry-picking. `optim.reference.newton_reference` returns f* plus a **certificate** `‖∇f(w*)‖²/(2μ)` bounding how far the reference sits above the true optimum (~2e-24 on `ridge`). The composite/L1 case has no strong convexity hence no certificate: there `best_known()` takes the smallest value ANY run reached, which is conservative (a lower f* only makes the reported gaps larger).
- **μ is NOT λ.** The intercept is deliberately unpenalized, so the ridge term is `λ·diag(reg)` with a zero on the intercept: `λI ⪯ ∇²f` holds only on the penalized coordinates, and along the intercept the curvature `(1/n)Σpᵢ(1−pᵢ)` decays to 0 as ‖w‖ grows. f is strongly convex on bounded sublevel sets, not globally. `newton_reference` therefore uses the **measured** `λ_min(∇²f(w*))` — on the current d=415 `ridge` matrix that is **1.001e-3 ≈ 1.00λ** (re-measured; the 2.85e-3 / ~2.8λ quoted here before was from the old d=39 matrix). The near-equality with λ is a numerical coincidence at this w*, not a structural identity: measure, never substitute λ. This is not academic: pure Newton from a far start walks to ‖w‖~1e16, S underflows, and the Hessian goes **exactly singular** in the intercept direction (`λ_min`: 9.7e-4 → 2.8e-17 → 0 over 7 steps while f climbs to 4e15). That failure is the content of the pure-vs-damped Newton slide.
- **The money plot's Newton keeps `newton_reference`'s DEFAULT line search (t0=1, ρ=0.5, c=1e-4), not the hand-tuned (t0=2, ρ=0.5, c=0.2) of §4.3 — and that is deliberate, measured.** Feeding the tuned pair in makes the reference run **hit the 100-iteration cap in 95 s without reaching `tol=1e-14`**, versus 15 iterations / 10.4 s at the default: c=0.2 is a strict Armijo test, and near the optimum floating-point noise makes it fail, so the step is backtracked toward 0 while ‖∇f‖ stalls near 1e-13. Since that same run produces `f*` and its certificate, the reference must be the configuration that actually certifies. §4.15 states this on the slide rather than hiding it. Don't "fix" it by wiring `newton_backtracking` into `run_benchmark`.
- **`newton(strict=True)` (the default) RAISES on a non-PD Hessian.** It used to `break` silently, which turned a rank-deficient design matrix into an innocuous-looking `converged=False`. Only `newton_init_study` passes `strict=False`, because there the singular Hessian *is* the result.
- **AGD logs and stops at the same point.** Row k holds `f(w_k)` with `‖∇f(y_{k-1})‖`, and the descent lemma at step 1/L plus PL gives `f(w_k) − f* ≤ ‖∇f(y_{k-1})‖²/(2μ)` — the stopping test certifies exactly the quantity plotted, on the same row. Evaluating `‖∇f(w_k)‖` directly would cost a second gradient per iteration and make the wall-clock comparison with GD dishonest.
- **SGD's hyperparameters are measured, not guessed.** GD/AGD get 1/L from theory and Newton gets Armijo; Robbins–Monro only constrains the *shape* of the SGD schedule, never the constant. `optim.experiments.sgd_hyper_sweep` (6×4 grid over (η₀, batch) + a 5-point γ sweep, equal 50-epoch budget) picks **η₀=0.1, batch=64, γ=2e-4** → `f−f*` = 1.30e-5; the old guess (0.5, 256) was 13× worse, and constant step (γ=0) is 51× worse. Constants live in `optim/benchmark.py` as `SGD_*`. Note `sgd_multiseed` deliberately keeps η₀=0.5 — the floor is ∝η, so at the tuned step the per-batch floors would overlap and that figure's whole point would vanish.
- **`blas_threads` verifies instead of assuming.** Setting the env vars after numpy has loaded is a silent no-op that quietly invalidates every timing. The module warns if imported late and `blas_threads.verify()` interrogates the live backend via `threadpoolctl`; timing drivers call it and print the result.
- **`Recorder.tick(grad=...)` takes a float** — SGD counts a minibatch as `|B|/n` of a full gradient, so the cost axis stays comparable with the full-batch methods.

## Data (`data/`)

Multi-table panel/transactional data from the VIB Hackathon (`0.Data VIB Hackathon Guidline.xlsx`), keyed by `CUSTOMER_NUMBER`. Must be folded into one cross-sectional row per customer. Schema is in `data/data description.txt`. Sizes: Customer ~290K rows, MyVIB Transaction ~1.4M, MyVIB Activity ~16M, Deposit ~1.26M.

- `1.Data_Customer.csv` — static demographics (one row/customer)
- `2.Data_MyVIB_Transaction.csv` — richest table; RFM + behavior + trends
- `3.Data_MyVIB_Activity.csv` — non-financial engagement (large)
- `4.Data_Deposit.csv`, `5.Data_Lending.xlsx`, `6.Data_Card.xlsx` — monthly panel balances/holdings

**Watch column-name drift:** `data description.txt` says `TRXN_LV1/TRXN_LV2`, but the CSV headers are `TRANS_LV1/TRANS_LV2`. Activity dates use `M/D/YYYY`; customer/transaction dates use `YYYY-MM-DD`. Verify actual headers/formats before coding against the description.

### Labeling (avoid leakage)

Define three disjoint time windows around a snapshot `T`: observation `[T-K, T]` (features only) and outcome `(T, T+H]` (label only). Suggested `K=6` months, `H=2–3`. Among customers active in the observation window, `y_i = 1` if they have *no* transaction *and* no activity in `(T, T+H]`. Drop already-churned and newly-onboarded customers. Aim for a churn rate of ~10–30% (extreme imbalance drives `p_i(1-p_i)→0`, shrinking the Hessian and inflating `κ`).

## Dimension target

`d ≈ 300–400` on the `ridge` variant, so every algorithm takes a measurable amount of wall-clock (the earlier d=39 matrix finished every run in under 5 s, which made the cost axis meaningless). Newton is `O(nd² + d³)`, so **d is the dominant runtime lever** — n only enters linearly.

d comes from pivoting the categorical axes the raw tables actually have, not from synthetic expansion: `TRANS_LV1` (3), `TRANS_LV2` (14), `ACTIVITY_NAME` (45), hour-of-day (24, for transactions AND activity), day-of-week (7), each as count / amount / share / recency where it makes sense; plus per-month levels `M0..M2`, amount-distribution shape (quantiles, IQR, CV, max-share), inter-arrival gap statistics, and Shannon entropy over each categorical axis. ~500 raw columns → 415 after pruning.

Degree-2 polynomial terms stay on a small core (~6–8 features) — a deliberate `κ` knob, not a blanket expansion.

### Rank is not automatic at this width
Correlation and VIF filters are statistical: they catch strong *pairwise* and *near*-collinearity. Neither removes an **exact** multi-column identity, and VIF cannot even see one — a singular correlation matrix sends `np.linalg.inv` to the `pinv` fallback whose finite diagonal looks like an ordinary VIF. Two rules:
1. **Never construct a feature that is an exact function of features already present.** `SPAN = FIRST_AGE − RECENCY` and `DAYS_PER_MONTH = ACTIVE_DAYS / K` were both written and both had to be deleted: zero information, guaranteed rank deficiency.
2. `churn_opt.build._rank_prune` (pivoted QR, keep pivots with `|diag(R)|` above the numerical tolerance) runs last on `ridge` and `poly` as the guarantee. Without it the first d=416 run came out rank 415 and Newton would have crashed.

### Multi-snapshot stacking
One row per (customer, snapshot). Shortening K **reduces** n on its own — eligibility means "active in (T−K, T]", so K=3 is a subset of K=6 (46,821 → 44,026); the stacking is what buys observations back (→ 75,026). Both windows are half-open on the left and land on month ends, so the three observation windows tile 2019 exactly and none reaches into 2018 (which the data does not cover). A customer may appear at several snapshots, so rows are **not iid** — irrelevant for an optimization study, but say so on the slide.
