"""Driver: run every optimizer on one variant, estimate f*, collect histories.

Flow:
  1. load_variant -> build LogisticObjective (lambda from CLI/config).
  2. conditioning() -> L, mu, kappa (reuse churn_opt.diagnostics; feeds AGD/GD steps).
  3. Newton to deep convergence -> f_star (the reference for log(f - f*)).
  4. Run GD and AGD at their hand-tuned steps (gd_config / agd_config), plus
     SGD and Newton, all from the same w0.
Returns a BenchmarkResult consumed by plots.py / the report.

Step-size stability is studied separately and HONESTLY in optim/experiments.py:
`quadratic_divergence` (clean eta>2/L blow-up on the local quadratic model) and
`gd_step_sweep` (why that 2/L bound is pessimistic on the regularized logistic).
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field

import numpy as np

from churn_opt.diagnostics import conditioning
from .data import load_variant
from .objective import LogisticObjective
from .optimizers import gradient_descent, accelerated_gd, newton, sgd
from .optimizers.ista import (ista, fista, coordinate_descent, subgradient,
                              _composite, _nnz)
from .reference import newton_reference, best_known

# SGD's configuration is MEASURED, not guessed: GD/AGD get their step from 1/L and
# Newton from Armijo, but Robbins-Monro only constrains the SHAPE of the SGD
# schedule (sum eta_k = inf, sum eta_k^2 < inf), never the constant. So it is swept
# by `optim.experiments.sgd_hyper_sweep` (grid over (eta0, batch) plus a gamma
# sweep, equal 50-epoch budget), run via `python run_experiments.py sgd-hyper`.
#
# The configuration is READ BACK from artifacts/tuning.json (written by
# run_tuning.py, where it is chosen by hand off the figures with a stated reason)
# rather than copied here: a hand-copied constant is a second source of truth that
# silently drifts from the file it came from the first time the study is re-run.
# The values below are only the fallback for when nothing has been run at all.
SGD_EPOCHS = 50
SGD_FALLBACK = dict(eta0=0.1, batch=64, gamma=2e-4)


def sgd_config(art_dir: "Path | None" = None) -> dict:
    """(eta0, batch, gamma) for the headline SGD run — from the measured sweep.

    Returns the fallback (and says so) when artifacts/sgd_hyper.json is missing.
    """
    from pathlib import Path
    path = Path(art_dir or Path(__file__).resolve().parent.parent / "artifacts")

    f = path / "tuning.json"
    if f.exists():
        d = json.loads(f.read_text()).get("sgd_chosen")
        if d:
            # The trial-and-error study works in schedule names, not in the
            # eta0/(1+gamma*k) form the epoch-based `sgd` takes. Only "hằng"
            # (constant) maps onto it exactly, via gamma=0; anything else would be
            # a silent re-interpretation, so refuse rather than guess.
            if d["schedule"] != "hằng":
                raise ValueError(
                    f"sgd_chosen schedule={d['schedule']!r} has no exact equivalent "
                    "in the epoch-based sgd(eta0, gamma) parameterisation; run the "
                    "money plot with optim.tuning.sgd_steps instead.")
            return dict(eta0=float(d["eta0"]), batch=int(d["batch"]), gamma=0.0,
                        source=f"{f} [{d['schedule']}]")

    return dict(SGD_FALLBACK, source="fallback (tuning.json/sgd_chosen not found)")


def gd_config(L: float, art_dir: "Path | None" = None) -> dict:
    """Step size for the headline GD run — the hand-tuned one when it exists.

    GD's 1/L is a SAFE step, not the best one: the descent lemma only needs
    t < 2/L for monotone decrease, and run_tuning.py's trial-and-error found
    t = 0.4 = 2.8 * (1/L) reaches f - f* = 4.5e-3 in 150 iterations where 1/L is
    an order of magnitude behind. Running the money plot at 1/L therefore
    handicaps GD against a bound the deck itself shows to be pessimistic.

    AGD deliberately does NOT get the same treatment: Nesterov's constant
    momentum beta = (sqrt(kappa)-1)/(sqrt(kappa)+1) is derived ASSUMING step 1/L,
    so substituting the swept t breaks the (t, beta) pair the proof relies on —
    see optim/optimizers/agd.py. Its step stays 1/L.
    """
    from pathlib import Path
    path = Path(art_dir or Path(__file__).resolve().parent.parent / "artifacts")

    f = path / "tuning.json"
    if f.exists():
        d = json.loads(f.read_text()).get("gd_fixed")
        if d and d.get("chosen"):
            eta = float(d["chosen"])
            return dict(eta=eta, source=f"{f} [gd_fixed]", tuned=True,
                        eta_theory=1.0 / L)
    return dict(eta=1.0 / L, source="1/L (tuning.json/gd_fixed not found)",
                tuned=False, eta_theory=1.0 / L)


def agd_config(L: float, art_dir: "Path | None" = None) -> dict:
    """Step (and the implied momentum) for the headline AGD run.

    Same idea as gd_config, but AGD needs one extra care. Its momentum
    beta = (sqrt(kappa)-1)/(sqrt(kappa)+1) is DERIVED assuming step 1/L, so the
    swept step is not passed as a bare `t`: it is fed in as L_eff = 1/t and beta
    is recomputed from L_eff. That keeps (t, beta) a matched pair. Changing t
    alone while holding beta fixed is the thing that actually breaks AGD — it is
    what backtracking does, and it costs convergence entirely.

    L_eff = 1/0.2 = 5 is BELOW the true L = 14.07, i.e. an optimistic smoothness
    estimate, so this pair sits outside the theorem's guarantee. Measured it is
    1.7x faster than 1/L (1437 vs 2400 iterations); the guarantee-respecting
    number is quoted in the deck's section 5.2 as the conservative one.
    """
    from pathlib import Path
    path = Path(art_dir or Path(__file__).resolve().parent.parent / "artifacts")

    f = path / "tuning.json"
    if f.exists():
        d = json.loads(f.read_text()).get("gd_accel")
        if d and d.get("chosen"):
            t = float(d["chosen"])
            return dict(eta=t, L_eff=1.0 / t, source=f"{f} [gd_accel]", tuned=True,
                        eta_theory=1.0 / L)
    return dict(eta=1.0 / L, L_eff=L, source="1/L (tuning.json/gd_accel not found)",
                tuned=False, eta_theory=1.0 / L)


@dataclass
class BenchmarkResult:
    variant: str
    conditioning: dict
    f_star: float
    results: dict          # name -> OptResult
    f_sklearn: float = float("nan")   # lbfgs objective at its solution
    w_sklearn: "np.ndarray | None" = None
    f_star_bound: float = float("nan")     # certified over-estimate of f*: |g|^2/(2mu)
    f_star_grad_norm: float = float("nan")  # ||grad f|| at the reference point
    sgd_config: dict = field(default_factory=dict)   # what SGD actually ran with
    gd_config: dict = field(default_factory=dict)    # what GD actually ran with
    agd_config: dict = field(default_factory=dict)   # what AGD actually ran with


def l1_config(art_dir: "Path | None" = None) -> dict:
    """Hand-tuned L1 steps: subgradient eta0 (4.7.3) and the ISTA/FISTA multiple
    of 1/L (4.7.4).

    Without this, run_l1_benchmark ran its OWN probe sweep for the subgradient
    (it picked eta0=1 where the deck's hand sweep picks 2) and left ISTA/FISTA at
    the textbook t=1/L (the deck's hand sweep picks 7/L). Slides 7.5-7.6 then
    reported three runs at parameters no slide had chosen. Same rule as
    gd_config/agd_config: read tuning.json, fall back and say so.
    """
    from pathlib import Path
    path = Path(art_dir or Path(__file__).resolve().parent.parent / "artifacts")
    out = dict(eta0=None, t_ista=1.0, t_fista=1.0,
               source="fallback (tuning.json not found)")
    f = path / "tuning.json"
    if f.exists():
        d = json.loads(f.read_text())
        sub, ist = d.get("l1_subgradient"), d.get("l1_ista")
        fis = d.get("l1_fista")
        out = dict(eta0=float(sub["chosen"]) if sub and sub.get("chosen") else None,
                   t_ista=float(ist["chosen"]) if ist and ist.get("chosen") else 1.0,
                   # FISTA gets its OWN multiple: its usable ceiling is lower than
                   # ISTA's (5/L vs 7/L measured). Momentum amplifies the overshoot
                   # that plain ISTA absorbs — same shape as AGD vs GD in section 4.
                   t_fista=float(fis["chosen"]) if fis and fis.get("chosen") else 1.0,
                   source=f"{f} [l1_subgradient, l1_ista, l1_fista]")
    return out


@dataclass
class L1BenchmarkResult:
    variant: str
    alpha: float
    L: float
    d_reg: int             # number of regularizable (non-intercept) coords
    f_star: float          # deep-FISTA composite optimum
    results: dict          # name -> OptResult (ISTA, FISTA)
    f_sklearn: float = float("nan")   # saga composite at its solution
    nnz_sklearn: int = -1
    subgrad_eta0: float = float("nan")   # eta0 the probe sweep picked


def sklearn_baseline(obj: LogisticObjective, X_raw: np.ndarray, y: np.ndarray):
    """Fit sklearn's lbfgs on the SAME objective and return (w, f).

    sklearn minimizes  (1/2)||w||^2 + C * sum_i logloss_i  with an UNpenalized
    intercept, so C = 1/(lambda*n) (= obj.sklearn_C()) makes its minimizer equal
    ours. `X_raw` is the design matrix WITHOUT our appended intercept column;
    sklearn adds/handles the intercept itself. We reassemble w = [coef, intercept]
    in obj.X's column order (intercept last) and evaluate our f on it — the gap
    to Newton's f* is the cross-check that both the objective and f* are correct.
    """
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        C=obj.sklearn_C(), penalty="l2", solver="lbfgs",
        fit_intercept=obj.fit_intercept, tol=1e-10, max_iter=2000,
    )
    clf.fit(X_raw, y)
    if obj.fit_intercept:
        w = np.concatenate([clf.coef_[0], clf.intercept_])
    else:
        w = clf.coef_[0].copy()
    return w, obj.value(w)


def run_benchmark(
    variant: str = "ridge",
    lam: float = 1e-3,
    seed: int = 0,
    max_iter: int = 3000,
) -> BenchmarkResult:
    ds = load_variant(variant)
    obj = LogisticObjective(ds.X_train, ds.y_train, lam=lam)
    w0 = np.zeros(obj.d)

    # conditioning on the intercept-augmented X (matches what the optimizers see)
    cond = conditioning(obj.X, lam)
    L, mu = cond["L"], max(cond["mu"], lam)

    # reference optimum: Newton to deep convergence. f* is the value AT the final
    # iterate, certified by ||grad f(w*)||^2 / (2 mu) — see optim/reference.py.
    # Newton here keeps newton_reference's DEFAULT line search (t0=1, rho=0.5,
    # c=1e-4), not the (t0=2, rho=0.5, c=0.2) pair the deck picks in its tuning
    # section. Measured reason: c=0.2 is a strict Armijo test, and near the optimum
    # floating-point noise makes it fail, so the step is backtracked toward 0 and
    # ||grad f|| stalls around 1e-13 — 100 iterations / 95 s WITHOUT reaching the
    # 1e-14 tolerance, against 15 iterations / 10.4 s for the default. Since this
    # same run also produces f* and its certificate, the reference must be the
    # configuration that actually certifies.
    reference = newton_reference(obj, w0, mu=mu)
    ref, f_star = reference.result, reference.f_star

    # independent cross-check: sklearn lbfgs on the matched objective
    w_sk, f_sk = sklearn_baseline(obj, ds.X_train, ds.y_train)

    cfg = sgd_config()
    print(f"  SGD config: eta0={cfg['eta0']:g} batch={cfg['batch']} "
          f"gamma={cfg['gamma']:g}  [{cfg['source']}]")
    gcfg = gd_config(L)
    print(f"  GD  config: eta={gcfg['eta']:g} "
          f"({gcfg['eta'] * L:.2f}/L, 2/L threshold = {2.0 / L:.4g})  "
          f"[{gcfg['source']}]")
    acfg = agd_config(L)
    _kap = acfg["L_eff"] / mu
    print(f"  AGD config: eta={acfg['eta']:g} (L_eff={acfg['L_eff']:g} vs L={L:.4g}, "
          f"beta={(np.sqrt(_kap) - 1) / (np.sqrt(_kap) + 1):.5f})  [{acfg['source']}]")

    results = {
        "Newton": ref,
        "GD": gradient_descent(obj, w0, eta=gcfg["eta"], max_iter=max_iter, tol=1e-10),
        "AGD": accelerated_gd(obj, w0, L=acfg["L_eff"], mu=mu,
                              max_iter=max_iter, tol=1e-10),
        "SGD": sgd(obj, w0, eta0=cfg["eta0"], batch_size=cfg["batch"],
                   epochs=SGD_EPOCHS, gamma=cfg["gamma"], seed=seed),
    }
    return BenchmarkResult(variant, cond, f_star, results, sgd_config=cfg,
                           gd_config=gcfg, agd_config=acfg,
                           f_sklearn=f_sk, w_sklearn=w_sk,
                           f_star_bound=reference.bound,
                           f_star_grad_norm=reference.grad_norm)


def _smooth_lipschitz(obj: LogisticObjective) -> float:
    """Lipschitz constant of the SMOOTH part's gradient: L = (1/4)lambda_max(gram)
    (+ lambda if obj carries an L2 term). Used as the 1/L step for ISTA/FISTA."""
    gram = (obj.X.T @ obj.X) / obj.n
    lambda_max = float(np.linalg.eigvalsh(gram)[-1])
    return 0.25 * lambda_max + obj.lam


def sklearn_l1_baseline(obj: LogisticObjective, X_raw: np.ndarray,
                        y: np.ndarray, alpha: float):
    """saga L1 baseline on the matched composite. C = 1/(alpha*n) makes sklearn's
    (sum-of-logloss + ||w||_1) minimizer equal ours ((1/n)logloss + alpha||w||_1)."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        C=1.0 / (alpha * obj.n), penalty="l1", solver="saga",
        fit_intercept=obj.fit_intercept, tol=1e-7, max_iter=5000,
    )
    clf.fit(X_raw, y)
    if obj.fit_intercept:
        w = np.concatenate([clf.coef_[0], clf.intercept_])
    else:
        w = clf.coef_[0].copy()
    return _composite(obj, w, alpha), _nnz(obj, w)


def run_l1_benchmark(
    variant: str = "lasso",
    alpha: float = 1e-3,
    max_iter: int = 6000,
    validate: bool = True,
    subgrad_etas: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0),
    probe_iter: int = 400,
) -> L1BenchmarkResult:
    """ISTA vs FISTA on F(w) = (1/n)logloss + alpha||w||_1 (pure Lasso: no L2).
    The lasso variant keeps redundant columns on purpose, so the L1 term has
    something to zero out. f* comes from a deep FISTA run; saga cross-checks it.
    """
    ds = load_variant(variant)
    obj = LogisticObjective(ds.X_train, ds.y_train, lam=0.0)  # smooth part only
    w0 = np.zeros(obj.d)
    L = _smooth_lipschitz(obj)
    d_reg = int((obj._reg > 0).sum())

    # reference optimum: deep FISTA (20k iters), cross-checked by saga.
    # lambda = 0 here, so the composite is convex but NOT strongly convex: no PL
    # certificate is available. The rule instead is `best_known` — the smallest
    # composite value ANY run reached. That direction is safe: a lower f* can only
    # make the reported gaps larger. FISTA is genuinely non-monotone, so its own
    # history is scanned rather than only its last iterate. (A second deep CD run
    # used to sit here; it was pure duplicated work once best_known started folding
    # in the compared runs, and CD costs ~170 ms/epoch at d=520.)
    lcfg = l1_config()
    # ISTA/FISTA take L, and use step 1/L internally — so the hand-tuned multiple
    # t = t_mult/L is passed in as an EFFECTIVE L of L/t_mult. FISTA's momentum is
    # the theta-sequence, which carries no problem constant, so unlike the strongly
    # convex AGD there is no (t, beta) pair to break by doing this.
    L_ista, L_fista = L / lcfg["t_ista"], L / lcfg["t_fista"]
    print(f"  L1 config: subgradient eta0={lcfg['eta0']}, ISTA t={lcfg['t_ista']:g}/L, "
          f"FISTA t={lcfg['t_fista']:g}/L  [{lcfg['source']}]", flush=True)

    # the reference stays at the SAFE step 1/L: it produces f*, so it must be the
    # run inside the theory's guarantee, not the fastest one.
    ref = fista(obj, w0, alpha=alpha, L=L, max_iter=20000, tol=1e-12)

    results = {
        "ISTA": ista(obj, w0, alpha=alpha, L=L_ista, max_iter=max_iter, tol=1e-10),
        "FISTA": fista(obj, w0, alpha=alpha, L=L_fista, max_iter=max_iter, tol=1e-10),
        "CD": coordinate_descent(obj, w0, alpha=alpha, max_epochs=max_iter, tol=1e-11),
    }

    # Subgradient: the control that makes the proximal case. Its eta0 is SWEPT, not
    # guessed — the same rule the rest of the project follows, and it matters more
    # here than anywhere else: a subgradient method that looks bad because it was
    # given a bad step proves nothing. Short probe runs pick eta0, then the winner
    # gets the same budget as everyone else.
    if lcfg["eta0"] is not None:
        best_eta = lcfg["eta0"]                  # dò tay ở mục 7.3
    else:
        probe = {}
        for e in subgrad_etas:
            r = subgradient(obj, w0, alpha=alpha, eta0=e, max_iter=probe_iter)
            probe[e] = min(r.f_history)
        best_eta = min(probe, key=probe.get)
        if best_eta in (min(subgrad_etas), max(subgrad_etas)):
            warnings.warn(
                f"subgradient eta0={best_eta:g} is on the edge of the probe grid "
                f"{list(subgrad_etas)} — extend it before quoting the result.",
                RuntimeWarning, stacklevel=2)
    results["Subgradient"] = subgradient(obj, w0, alpha=alpha, eta0=best_eta,
                                         max_iter=max_iter)

    f_sk, nnz_sk = (float("nan"), -1)
    if validate:
        f_sk, nnz_sk = sklearn_l1_baseline(obj, ds.X_train, ds.y_train, alpha)

    # Take the best value observed ANYWHERE, the compared runs included: a
    # 6000-epoch CD legitimately lands below a 2000-epoch reference, and if f*
    # stayed above the best point actually found, that run's gap would go
    # negative. Folding them in can only LOWER f*, i.e. widen every reported gap.
    f_star = best_known(min(ref.f_history), f_sk,
                        *(min(r.f_history) for r in results.values()))

    return L1BenchmarkResult(variant, alpha, L, d_reg, f_star, results,
                             f_sklearn=f_sk, nnz_sklearn=nnz_sk,
                             subgrad_eta0=float(best_eta))
