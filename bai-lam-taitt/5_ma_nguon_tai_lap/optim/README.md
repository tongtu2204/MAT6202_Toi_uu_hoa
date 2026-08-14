# optim — optimizer benchmark

Deliverable #2: implement **GD, Accelerated GD (Nesterov), Newton (damped/IRLS),
SGD** by hand on the L2-regularized logistic objective from `churn_opt`, and
compare per-iteration cost vs convergence rate. ISTA/FISTA/CD on the `lasso`
variant is the L1 bonus; AdaGrad/RMSprop/Adam is the adaptive bonus.

## Layout
```
optim/
  data.py            load artifacts/<variant>.npz  -> Dataset
  objective.py       f(w), grad, hessian  (sklearn-matched)
  linesearch.py      Armijo backtracking
  reference.py       f* + its strong-convexity certificate   <- read this first
  optimizers/
    base.py          OptResult history + Recorder (cost counters)
    gd.py            gradient descent (fixed / backtracking)
    agd.py           Nesterov accelerated GD
    newton.py        damped Newton = IRLS, produces f*
    sgd.py           minibatch SGD
    ista.py          ISTA / FISTA / coordinate descent (L1 bonus)
    adaptive.py      AdaGrad / RMSprop / Adam (+AdamW, AMSGrad)
  benchmark.py       run all on a variant; SGD_* constants live here
  experiments.py     parameter sweeps + honesty studies
  plots.py           log(f - f*) curves + sweep/study figures
../run_benchmark.py    headline benchmark CLI
../run_experiments.py  sweeps + studies CLI
../run_adaptive.py     adaptive-family study
```

## Run
```bash
python run_pipeline.py                 # first: writes artifacts/*.npz (+ ridge_raw)
python run_benchmark.py --variant ridge
python run_benchmark.py --variant poly # wider d: what it costs Newton
#   (kappa is swept via lambda in kappa_sweep, NOT by widening the matrix)
python run_benchmark.py --variant lasso --l1
python run_experiments.py all          # gd-sweep, quad-div, newton-init, sgd, sgd-hyper, std, dim
```

Figures land in `artifacts/figures/`.

## Four rules that are easy to get subtly wrong

1. **Never write `min(f_history)` for `f*`.** Use `reference.newton_reference`
   (smooth case, returns a certificate) or `reference.best_known` (composite/L1).
   A `min` over a run's own history reports a point the run did not return and
   biases every other method's gap downward.
2. **μ is not λ.** The intercept is unpenalized, so `λI ⪯ ∇²f` holds only on the
   penalized coordinates and f is strongly convex on bounded sublevel sets, not
   globally. Certificates use the measured `λ_min(∇²f(w*))`.
3. **`newton(strict=True)` raises on a non-PD Hessian.** That is intentional — a
   silent `break` disguises a rank-deficient design matrix as slow convergence.
4. **A logged pair must describe one point.** AGD logs `f(w_k)` with
   `‖∇f(y_{k-1})‖`, which bounds it via the descent lemma + PL; do not mix in a
   gradient from the next momentum point.

Step sizes come from theory where theory has one (`1/L` for GD/AGD, Armijo for
Newton) and from a measured sweep where it does not (SGD: `sgd_hyper_sweep`;
adaptive family: `adaptive_family`). Nothing is hard-coded on taste.
