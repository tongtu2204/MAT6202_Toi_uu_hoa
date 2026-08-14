"""Slide experiments beyond the headline benchmark — the parameter sweeps and
two honesty studies (A, B) the presentation content calls for.

Each driver returns a small result object that `plots.py` turns into one figure.
Everything reuses the same LogisticObjective / optimizers as `benchmark.py`, so
the objective, f*, and cost accounting are identical across the whole talk.

  gd_step_sweep          (Sweep 1 + study B) — GD across eta multiples of 1/L,
                          showing the textbook 2/L boundary is PESSIMISTIC on the
                          regularized logistic (it self-conditions off w0).
  quadratic_divergence   (study B) — GD on the local quadratic model, where the
                          curvature is constant so eta>2/L blows up cleanly. This
                          is the honest "objective flies to infinity" plot.
  newton_init_study      (Sweep 2) — pure vs damped Newton from a FAR w0.
  sgd_config_sweep       (Sweep 3) — batch size + constant-vs-decaying step,
                          exposing the SGD variance floor.
  dim_scaling            (study A) — Newton vs AGD wall-clock as d grows, so the
                          O(nd^2 + d^3) per-iteration cost of Newton overtakes the
                          O(nd) first-order methods (the money-plot reversal).
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from churn_opt.diagnostics import conditioning
from .data import load_variant, ARTIFACTS
from .diagnostics import solve_deep
from .objective import LogisticObjective, empirical_lipschitz
from .optimizers import gradient_descent, accelerated_gd, newton, sgd
from .optimizers.base import OptResult
from .reference import newton_reference


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _fstar(obj: LogisticObjective, w0: np.ndarray) -> tuple[float, np.ndarray]:
    """Deep-Newton optimum: the reference f* and its minimizer w*.

    f* is the objective AT the final iterate (not min over the history) and comes
    with a strong-convexity certificate — see optim/reference.py for why."""
    ref = newton_reference(obj, w0)
    return ref.f_star, ref.w_star


def _time_to_tol(res: OptResult, f_star: float, tol: float) -> float:
    """Wallclock at the first iterate with f - f* < tol (np.nan if never)."""
    sub = np.asarray(res.f_history) - f_star
    hit = np.flatnonzero(sub < tol)
    return float(res.time_s[hit[0]]) if hit.size else float("nan")


def _warn_if_on_grid_edge(chosen: float, grid, what: str) -> None:
    """Warn when a swept hyperparameter wins at the smallest/largest value tried.

    An optimum sitting on the boundary is not an optimum — it only means the grid
    stopped there. This has already bitten twice (RMSprop's eta, then SGD's eta0),
    so the check lives in the code instead of relying on someone reading the table.
    """
    lo, hi = min(grid), max(grid)
    if chosen in (lo, hi):
        side = "smallest" if chosen == lo else "largest"
        warnings.warn(
            f"{what}: best value {chosen:g} is the {side} point of the grid "
            f"[{lo:g}, {hi:g}] — this is a grid boundary, not a verified optimum. "
            "Extend the grid in that direction before quoting it.",
            RuntimeWarning, stacklevel=3)


def _iters_to_tol(res: OptResult, f_star: float, tol: float) -> int:
    sub = np.asarray(res.f_history) - f_star
    hit = np.flatnonzero(sub < tol)
    return int(hit[0]) if hit.size else -1


# --------------------------------------------------------------------------- #
# Study B + Sweep 1 — GD step-size sweep on the real logistic
# --------------------------------------------------------------------------- #
@dataclass
class StepSweep:
    variant: str
    lam: float
    L: float                 # global bound = lambda_max(H(0))
    L_star: float            # local curvature at the optimum
    f_star: float
    mults: list[float]       # eta expressed as mult / L
    runs: dict               # mult -> OptResult


def gd_step_sweep(
    variant: str = "ridge",
    lam: float = 1e-3,
    mults: tuple[float, ...] = (0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0),
    max_iter: int = 4000,
    tol: float = 1e-10,
) -> StepSweep:
    """Run full-batch GD at eta = mult / L for each mult and collect histories.

    L is the GLOBAL smoothness bound (attained at w0=0). The teaching point: on
    this regularized problem GD stays finite well past mult=2 because curvature
    collapses from L to L_star once the iterate leaves w0 — the classic 2/L
    divergence threshold is conservative here. (For the clean textbook blow-up see
    `quadratic_divergence`.)
    """
    ds = load_variant(variant)
    obj = LogisticObjective(ds.X_train, ds.y_train, lam=lam)
    w0 = np.zeros(obj.d)
    L = conditioning(obj.X, lam)["L"]
    f_star, w_star = _fstar(obj, w0)
    L_star = empirical_lipschitz(obj, w_star)
    runs = {
        m: gradient_descent(obj, w0, eta=m / L, max_iter=max_iter, tol=tol)
        for m in mults
    }
    return StepSweep(variant, lam, L, L_star, f_star, list(mults), runs)


# --------------------------------------------------------------------------- #
# Study B — clean divergence on the local QUADRATIC model
# --------------------------------------------------------------------------- #
@dataclass
class QuadDivergence:
    variant: str
    L: float                 # lambda_max(H0) — the quadratic's curvature bound
    mults: list[float]
    histories: dict          # mult -> np.ndarray of q(w_k) (may be +inf)


def quadratic_divergence(
    variant: str = "ridge",
    lam: float = 1e-3,
    mults: tuple[float, ...] = (0.5, 1.0, 1.9, 2.01, 2.1, 2.5),
    n_iter: int = 60,
    seed: int = 0,
) -> QuadDivergence:
    """GD on q(w) = 1/2 (w-w*)ᵀ H0 (w-w*), the second-order model at w0=0 whose
    Hessian H0 has lambda_max = L. Curvature is CONSTANT here, so the textbook
    result holds exactly: eta < 2/L converges, eta > 2/L diverges to infinity.
    This is the honest version of the "objective flies to infinity" slide — it is
    a property of the quadratic model, not of the full logistic (see the sweep).
    """
    ds = load_variant(variant)
    obj = LogisticObjective(ds.X_train, ds.y_train, lam=lam)
    w0 = np.zeros(obj.d)
    H0 = obj.hessian(w0)                       # curvature at w0; lambda_max == L
    L = float(np.linalg.eigvalsh(H0)[-1])
    rng = np.random.default_rng(seed)
    start = rng.standard_normal(obj.d)         # offset from the quadratic's min (0)

    histories = {}
    for m in mults:
        eta = m / L
        w = start.copy()
        hist = []
        for _ in range(n_iter):
            q = 0.5 * float(w @ (H0 @ w))
            hist.append(q)
            if not np.isfinite(q) or q > 1e12:
                break
            w = w - eta * (H0 @ w)             # grad of q is H0 w
        histories[m] = np.asarray(hist)
    return QuadDivergence(variant, L, list(mults), histories)


# --------------------------------------------------------------------------- #
# Sweep 2 — Newton pure vs damped from a far initialization
# --------------------------------------------------------------------------- #
@dataclass
class NewtonInit:
    variant: str
    lam: float
    w0_scale: float
    f_star: float
    pure: OptResult
    damped: OptResult


def newton_init_study(
    variant: str = "ridge",
    lam: float = 1e-3,
    w0_scale: float = 4.0,
    max_iter: int = 50,
    seed: int = 0,
) -> NewtonInit:
    """Pure (t=1) vs damped (Armijo) Newton from a random w0 far from the optimum.

    Far out the quadratic model is a poor fit, so the full step overshoots and the
    pure objective spikes; damping's line search keeps every step a descent step.
    Near the optimum both are identical (t->1), giving quadratic convergence — so
    this study is what MOTIVATES damping.

    Pure Newton here does not merely overshoot, it DIES, and the mechanism is
    worth stating exactly: the run walks out to ||w|| ~ 1e16, every sigmoid
    saturates, S = diag(p(1-p)) underflows to 0, and since the intercept carries
    no ridge term the Hessian becomes exactly singular in that one direction
    (measured: lambda_min goes 9.7e-4 -> 2.8e-17 -> 0 over 7 steps while f climbs
    to 4e15). So `strict=False` here is not papering over a bug — the singular
    Hessian IS the result. The warning it emits is the honest record of it.
    """
    ds = load_variant(variant)
    obj = LogisticObjective(ds.X_train, ds.y_train, lam=lam)
    rng = np.random.default_rng(seed)
    w0 = rng.standard_normal(obj.d) * w0_scale
    f_star, _ = _fstar(obj, np.zeros(obj.d))
    pure = newton(obj, w0, max_iter=max_iter, tol=1e-12, line_search=False,
                  strict=False)                      # singular H is the point here
    damped = newton(obj, w0, max_iter=max_iter, tol=1e-12, line_search=True)
    return NewtonInit(variant, lam, w0_scale, f_star, pure, damped)


# --------------------------------------------------------------------------- #
# Sweep 3 — SGD batch size + step schedule (variance floor)
# --------------------------------------------------------------------------- #
@dataclass
class SGDSweep:
    variant: str
    lam: float
    f_star: float
    floor_runs: dict         # label -> OptResult (constant etas + one diminishing)
    batch_runs: dict         # label -> OptResult (batch sizes, diminishing step)


def sgd_config_sweep(
    variant: str = "ridge",
    lam: float = 1e-3,
    const_etas: tuple[float, ...] = (0.3, 0.8, 2.0),
    batch_sizes: tuple[int, ...] = (32, 256, 2048),
    batch_size: int = 256,
    epochs: int = 80,
    gamma: float = 2e-4,
    seed: int = 0,
) -> SGDSweep:
    """Two studies in one result:

    floor_runs — CONSTANT step at several eta (gamma=0): each plateaus at a
      variance floor proportional to eta (the noise ball around w*), while one
      diminishing-step run (eta_t = eta0/(1+gamma*step)) anneals below them all.
      This is the textbook Robbins-Monro picture: constant step never converges.
    batch_runs — the diminishing schedule at several batch sizes: bigger batches
      cut the gradient-estimate variance, giving a smoother tail per epoch.

    gamma is small on purpose: the decay counts MINIBATCH steps (~n/batch per
    epoch), so a large gamma would kill the step in the first epoch. 2e-4 halves
    the step near mid-run instead.
    """
    ds = load_variant(variant)
    obj = LogisticObjective(ds.X_train, ds.y_train, lam=lam)
    w0 = np.zeros(obj.d)
    f_star, _ = _fstar(obj, w0)

    floor_runs = {
        f"const $\\eta$={e:g}": sgd(
            obj, w0, eta0=e, batch_size=batch_size, epochs=epochs, gamma=0.0, seed=seed)
        for e in const_etas
    }
    floor_runs[f"diminishing $\\eta_0$={max(const_etas):g}"] = sgd(
        obj, w0, eta0=max(const_etas), batch_size=batch_size, epochs=epochs,
        gamma=gamma, seed=seed)

    batch_runs = {
        f"batch={bs}": sgd(
            obj, w0, eta0=0.8, batch_size=bs, epochs=epochs, gamma=gamma, seed=seed)
        for bs in batch_sizes
    }
    return SGDSweep(variant, lam, f_star, floor_runs, batch_runs)


# --------------------------------------------------------------------------- #
# Sweep 4 — standardization contrast (raw vs z-scored X)
# --------------------------------------------------------------------------- #
@dataclass
class StdContrast:
    lam: float
    kappa_std: float
    kappa_raw: float
    f_star_std: float
    f_star_raw: float
    std_runs: dict           # method -> OptResult on the z-scored ridge matrix
    raw_runs: dict           # method -> OptResult on the same columns unscaled


def standardization_contrast(
    lam: float = 1e-3,
    max_iter: int = 2000,
    tol: float = 1e-10,
) -> StdContrast:
    """GD / AGD / Newton on the z-scored 'ridge' matrix vs the identical columns
    unscaled ('ridge_raw'). Raw column scales differ by orders of magnitude, so
    kappa explodes: GD (step 1/L with a huge L) and AGD crawl, while affine-
    invariant Newton converges at essentially the same rate on both — the slide
    that makes standardization mandatory. Needs `ridge_raw.npz` from the pipeline.
    """
    res = {}
    for tag, variant in (("std", "ridge"), ("raw", "ridge_raw")):
        ds = load_variant(variant)
        obj = LogisticObjective(ds.X_train, ds.y_train, lam=lam)
        w0 = np.zeros(obj.d)
        cond = conditioning(obj.X, lam)
        L, mu = cond["L"], max(cond["mu"], lam)
        # strict=False on the RAW design on purpose: at d=415 the unscaled Gram has
        # kappa ~ 1e19, past what float64 can represent, so lambda*I is numerically
        # invisible and the Hessian is singular. Newton not being able to run at all
        # is the finding of this slide, not a bug to hide — so it is recorded
        # (res.note) and reported rather than raised.
        ref = newton(obj, w0, max_iter=100, tol=1e-13, strict=(tag == "std"))
        runs = {
            "Newton": ref,
            "GD": gradient_descent(obj, w0, eta=1.0 / L, max_iter=max_iter, tol=tol),
            "AGD": accelerated_gd(obj, w0, L=L, mu=mu, max_iter=max_iter, tol=tol),
        }
        res[tag] = (runs, float(obj.value(ref.w)), cond["kappa_upper"])
    return StdContrast(
        lam, res["std"][2], res["raw"][2], res["std"][1], res["raw"][1],
        res["std"][0], res["raw"][0])


# --------------------------------------------------------------------------- #
# Study A — wall-clock crossover as dimension grows
# --------------------------------------------------------------------------- #
def expand_design(X: np.ndarray, target_d: int, seed: int = 0) -> np.ndarray:
    """Grow a standardized design matrix to `target_d` columns by appending
    z-scored products of random column pairs (degree-2 interactions), or truncate to
    the leading `target_d` columns when the base matrix is already wider. Keeps n and
    the leading columns fixed, so only d changes across the scaling study.

    The default d_list must STRADDLE the real d of the variant (415 for `ridge`):
    every point below it is a truncation of real columns and every point above is a
    synthetic expansion, and the crossover this study is about — Newton's O(nd^2+d^3)
    overtaking the O(nd) first-order methods — only shows if both sides are sampled.
    """
    _, d0 = X.shape
    if target_d <= d0:
        return X[:, :target_d].copy()
    rng = np.random.default_rng(seed)
    extra = []
    for _ in range(target_d - d0):
        i, j = rng.integers(0, d0, size=2)
        extra.append(X[:, i] * X[:, j])
    E = np.column_stack(extra)
    E = (E - E.mean(axis=0)) / (E.std(axis=0) + 1e-12)
    return np.hstack([X, E])


@dataclass
class DimScaling:
    variant: str
    lam: float
    tol: float
    d_list: list[int]
    rows: list[dict] = field(default_factory=list)   # per-d timings & iters


def _per_iter(res: OptResult) -> float:
    """Mean wallclock per outer iteration (total elapsed / #iterations)."""
    k = len(res.time_s)
    return float(res.time_s[-1] / k) if k else float("nan")


def dim_scaling(
    variant: str = "ridge",
    lam: float = 1e-3,
    d_list: tuple[int, ...] = (100, 200, 415, 800, 1600),
    tol: float = 1e-6,
    agd_max_iter: int = 3000,
    per_iter_reps: int = 12,
    seed: int = 0,
) -> DimScaling:
    """For each target d, measure Newton / AGD / GD on an expanded design matrix
    and record BOTH cost axes:

      per-iteration wallclock — Newton's O(nd^2 + d^3) rises far steeper than the
        first-order O(nd), and this crossover is robust: it is exactly the thesis
        "Newton is expensive per step". THIS is the honest money-plot signal.
      time-to-tolerance (total) — on this problem Newton still wins overall at
        every d here, because it needs ~8 iterations regardless of kappa while AGD
        needs O(sqrt(kappa)) and kappa grows with d. The classic total-time
        reversal only appears once d^3 dominates (d -> n); we report the numbers
        honestly rather than force it.
    """
    ds = load_variant(variant)
    Xbase, y = ds.X_train, ds.y_train
    out = DimScaling(variant, lam, tol, list(d_list))
    for d in d_list:
        X = expand_design(Xbase, d, seed=seed)
        obj = LogisticObjective(X, y, lam=lam)
        w0 = np.zeros(obj.d)
        cond = conditioning(obj.X, lam)
        L, mu = cond["L"], max(cond["mu"], lam)

        # convergence runs (for total time-to-tol): Newton is cheap; cap AGD so a
        # high-kappa expanded matrix cannot run away (nan t_agd if it misses tol).
        nt = newton(obj, w0, max_iter=100, tol=1e-13)
        f_star = float(obj.value(nt.w))
        ag = accelerated_gd(obj, w0, L=L, mu=mu, max_iter=agd_max_iter, tol=tol / 10)

        # per-iteration cost: short fixed-length runs, timed in isolation (no need
        # to converge — we only want seconds/iteration at this d).
        pit = per_iter_reps
        nt_c = newton(obj, w0, max_iter=pit, tol=0.0, line_search=False)
        ag_c = accelerated_gd(obj, w0, L=L, mu=mu, max_iter=pit, tol=0.0)
        gd_c = gradient_descent(obj, w0, eta=1.0 / L, max_iter=pit, tol=0.0)

        out.rows.append({
            "d": d,
            "kappa": cond["kappa_upper"],
            "t_newton": _time_to_tol(nt, f_star, tol),
            "t_agd": _time_to_tol(ag, f_star, tol),
            "it_newton": _iters_to_tol(nt, f_star, tol),
            "it_agd": _iters_to_tol(ag, f_star, tol),
            "pi_newton": _per_iter(nt_c),
            "pi_agd": _per_iter(ag_c),
            "pi_gd": _per_iter(gd_c),
        })
    return out


# --------------------------------------------------------------------------- #
# T1.1 — honest kappa: upper-bound vs local kappa*, predicted vs measured iters
# --------------------------------------------------------------------------- #
@dataclass
class KappaHonesty:
    variant: str
    lam: float
    L_bound: float           # deck bound = (1/4)lambda_max(gram)+lambda
    mu_bound: float          # = lambda
    kappa_bound: float
    L_star: float            # local lambda_max(H(w*))
    mu_star: float           # local lambda_min(H(w*))
    kappa_star: float
    f_star: float
    eps: float               # tolerance the predicted-iters use
    pred_gd_bound: float     # kappa_bound * ln(1/eps)
    pred_gd_star: float      # kappa_star  * ln(1/eps)
    meas_gd: int             # measured GD iters to f-f* < eps
    gd: OptResult            # the GD history (for the overlay figure)
    n_used: int = 0          # rows actually used (see the n_sub argument)


def kappa_honesty(
    variant: str = "ridge",
    lam: float = 1e-3,
    eps: float = 1e-8,
    gd_max_iter: int = 200_000,
    n_sub: int | None = 15_000,
    seed: int = 0,
) -> KappaHonesty:
    """The ROI-highest slide: show the deck's L/lambda bound over-predicts GD's
    iteration count by ~an order of magnitude, while the LOCAL kappa* at w* nails
    it. GD is run at the honest step 1/L* (survives thanks to self-conditioning);
    the plot overlays both theory rates (1-1/kappa)^t on the empirical curve.

    `n_sub` subsamples rows, for the same reason and with the same justification as
    in `kappa_sweep`: this is a statement about kappa, and kappa is a property of the
    feature geometry rather than of the sample size (15,000 of 75,026 rows moves
    kappa* by 2.4%). GD here needs O(kappa* log 1/eps) ~ 1.3e5 iterations at O(nd)
    each; on the full matrix that is 55 minutes for one number. Both kappa studies
    use the same subsample so their kappa* values are directly comparable, and
    `KappaHonesty.n_used` records it for the slide. n_sub=None uses every row.
    """
    ds = load_variant(variant)
    X, y = ds.X_train, ds.y_train
    if n_sub is not None and n_sub < len(y):
        idx = np.random.default_rng(seed).choice(len(y), n_sub, replace=False)
        X, y = X[idx], y[idx]
    obj = LogisticObjective(X, y, lam=lam)
    w0 = np.zeros(obj.d)

    cond = conditioning(obj.X, lam)
    L_bound, mu_bound = cond["L"], lam
    kappa_bound = L_bound / mu_bound

    w_star = solve_deep(obj)
    ev = np.linalg.eigvalsh(obj.hessian(w_star))
    mu_star, L_star = float(ev[0]), float(ev[-1])
    kappa_star = L_star / mu_star
    f_star = obj.value(w_star)

    gd = gradient_descent(obj, w0, eta=1.0 / L_star, max_iter=gd_max_iter, tol=1e-12)
    meas_gd = _iters_to_tol(gd, f_star, eps)

    ln = float(np.log(1.0 / eps))
    return KappaHonesty(
        variant, lam, L_bound, mu_bound, kappa_bound,
        L_star, mu_star, kappa_star, f_star, eps,
        kappa_bound * ln, kappa_star * ln, meas_gd, gd, n_used=int(obj.n))


# --------------------------------------------------------------------------- #
# T2.1 — kappa sweep -> log-log iters(kappa), fitted slope 1 (GD) vs 0.5 (AGD)
# --------------------------------------------------------------------------- #
@dataclass
class KappaSweep:
    variant: str
    lam_grid: list[float]
    kappa: list[float]       # local kappa*(lambda) per grid point
    it_gd: list[int]
    it_agd: list[int]
    slope_gd: float          # log-log fit; theory = 1
    slope_agd: float         # log-log fit; theory = 0.5
    rows: list[dict] = field(default_factory=list)
    n_used: int = 0          # rows actually used (see the n_sub argument)


def kappa_sweep(
    variant: str = "ridge",
    lam_grid: tuple[float, ...] = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3),
    eps: float = 1e-8,
    gd_max_iter: int = 80_000,
    agd_max_iter: int = 40_000,
    n_sub: int | None = 15_000,
    seed: int = 0,
) -> KappaSweep:
    """For each lambda: re-solve w*(lambda), read the LOCAL spectrum to get
    kappa*(lambda), then count GD and AGD iterations to f-f* < eps. Both run at the
    local step 1/L*, momentum from kappa*. The log-log slope of iters vs kappa* is
    the direct empirical check of the theory: 1 for GD, 1/2 for Nesterov AGD.

    `n_sub` runs the sweep on a random SUBSAMPLE of rows (default 15,000 of 75,026).
    This is a cost lever, and a legitimate one for this particular study: kappa is a
    property of the feature geometry, not of the sample size. Measured on `ridge`
    at lambda=1e-3, subsampling to 15,000 rows moves kappa_upper 1.407e4 -> 1.403e4
    and kappa* 3646 -> 3560, i.e. 2.4% — while cutting the sweep from ~3 hours to
    ~35 minutes, because GD needs O(kappa log 1/eps) iterations and each one costs
    O(nd). Set n_sub=None to run on every row. Whatever is used must be stated on
    the slide: `KappaSweep.n_used` carries it.

    The grid stops at lambda=1e-3 rather than 1e-4. GD needs O(kappa* log 1/eps)
    iterations, so each decade of lambda costs a decade of runtime, and the two
    smallest lambdas alone were most of a 3-hour run. What survives spans kappa*
    from ~40 to ~3600 — just under two decades, which is what a log-log slope fit
    needs. Widening it buys precision in the fit, not a different conclusion.
    """
    ds = load_variant(variant)
    X, y = ds.X_train, ds.y_train
    if n_sub is not None and n_sub < len(y):
        idx = np.random.default_rng(seed).choice(len(y), n_sub, replace=False)
        X, y = X[idx], y[idx]
    rows = []
    for lam in lam_grid:
        obj = LogisticObjective(X, y, lam=lam)
        w0 = np.zeros(obj.d)
        w_star = solve_deep(obj)
        ev = np.linalg.eigvalsh(obj.hessian(w_star))
        mu_s, L_s = float(ev[0]), float(ev[-1])
        kap = L_s / mu_s
        f_star = obj.value(w_star)
        gd = gradient_descent(obj, w0, eta=1.0 / L_s, max_iter=gd_max_iter, tol=1e-12)
        ag = accelerated_gd(obj, w0, L=L_s, mu=mu_s, max_iter=agd_max_iter, tol=1e-12)
        rows.append(dict(
            lam=lam, kappa_star=kap, mu_star=mu_s, L_star=L_s,
            it_gd=_iters_to_tol(gd, f_star, eps),
            it_agd=_iters_to_tol(ag, f_star, eps)))

    kap = np.array([r["kappa_star"] for r in rows])
    ig = np.array([r["it_gd"] for r in rows], float)
    ia = np.array([r["it_agd"] for r in rows], float)
    mg, ma = ig > 0, ia > 0
    slope_gd = float(np.polyfit(np.log10(kap[mg]), np.log10(ig[mg]), 1)[0])
    slope_agd = float(np.polyfit(np.log10(kap[ma]), np.log10(ia[ma]), 1)[0])
    return KappaSweep(
        variant, list(lam_grid), kap.tolist(),
        ig.astype(int).tolist(), ia.astype(int).tolist(),
        slope_gd, slope_agd, rows, n_used=int(len(y)))


# --------------------------------------------------------------------------- #
# T1.3 — Armijo backtracking: c1/rho sweep + fixed-vs-backtracking table
# --------------------------------------------------------------------------- #
def _armijo_count(fval, w, fw, g, direction, t0, beta, c, max_bt=60):
    """Armijo backtracking that also RETURNS the number of objective evaluations
    the line search spent (the hidden cost of backtracking)."""
    slope = float(g @ direction)
    t = t0
    for k in range(1, max_bt + 1):
        if fval(w + t * direction) <= fw + c * t * slope:
            return t, k
        t *= beta
    return t, max_bt


@dataclass
class ArmijoSweep:
    variant: str
    lam: float
    t0: float
    rows: list[dict]         # c1, rho, iters, fevals, bt_per_iter, mean_step, f_gap


def armijo_sweep(
    variant: str = "ridge",
    lam: float = 1e-3,
    c1_grid: tuple[float, ...] = (1e-4, 1e-2, 1e-1),
    rho_grid: tuple[float, ...] = (0.5, 0.8),
    budget: int = 1500,
    t0: float = 5.0,
) -> ArmijoSweep:
    """GD with Armijo backtracking across (c1, rho), started each step from a step
    t0 ABOVE the stable 2/L~1 so the search must backtrack — this is what exposes
    the hidden cost. Larger rho (0.8 vs 0.5) needs more trials per step but lands
    on a bigger accepted step; stricter c1 tightens acceptance. Reported at a fixed
    iteration budget so the fevals / accepted-step / final-gap tradeoff is visible."""
    ds = load_variant(variant)
    obj = LogisticObjective(ds.X_train, ds.y_train, lam=lam)
    w0 = np.zeros(obj.d)
    f_star, _ = _fstar(obj, w0)
    rows = []
    for c1 in c1_grid:
        for rho in rho_grid:
            w = w0.copy()
            fev = 0
            step_sum = 0.0
            for _ in range(budget):
                g = obj.grad(w)
                f = obj.value(w); fev += 1
                t, k = _armijo_count(obj.value, w, f, g, -g, t0=t0, beta=rho, c=c1)
                fev += k
                step_sum += t
                w = w - t * g
            rows.append(dict(
                c1=c1, rho=rho, iters=budget, fevals=fev,
                bt_per_iter=(fev - budget) / budget,   # avg line-search trials/iter
                mean_step=step_sum / budget,
                f_gap=float(obj.value(w) - f_star)))
    return ArmijoSweep(variant, lam, t0, rows)


@dataclass
class FixedVsBacktrack:
    variant: str
    lam: float
    rows: list[dict]         # method, mode, iters, fevals, time, f_gap


def fixed_vs_backtracking(
    variant: str = "ridge",
    lam: float = 1e-3,
    max_iter: int = 5000,
    tol: float = 1e-9,
    c1: float = 1e-4,
    rho: float = 0.5,
) -> FixedVsBacktrack:
    """One table: GD / AGD / Newton, each in FIXED and BACKTRACKING mode, with the
    iteration count, objective evaluations, wallclock and final gap. This is the
    half of the assignment the old deck never had a line for. Every runner is
    instrumented explicitly so the fevals column is exact."""
    import time
    ds = load_variant(variant)
    obj = LogisticObjective(ds.X_train, ds.y_train, lam=lam)
    w0 = np.zeros(obj.d)
    f_star, _ = _fstar(obj, w0)
    L = conditioning(obj.X, lam)["L"]
    mu = max(lam, 1e-12)
    rows = []

    def record(method, mode, w, fev, it, t0, conv):
        rows.append(dict(method=method, mode=mode, iters=it, fevals=fev,
                         time=time.perf_counter() - t0, converged=conv,
                         f_gap=float(obj.value(w) - f_star)))

    def run_gd(mode):
        w = w0.copy(); fev = 0; conv = False; t0 = time.perf_counter()
        it = 0
        for it in range(1, max_iter + 1):
            g = obj.grad(w); f = obj.value(w); fev += 1
            if float(np.linalg.norm(g)) < tol:
                conv = True; break
            if mode == "fixed":
                t = 1.0 / L
            else:
                t, k = _armijo_count(obj.value, w, f, g, -g, 1.0, rho, c1); fev += k
            w = w - t * g
        record("GD", mode, w, fev, it, t0, conv)

    def run_agd(mode):
        # AGD with optional Armijo on the gradient step at the momentum point.
        kappa = L / mu
        beta_m = (np.sqrt(kappa) - 1.0) / (np.sqrt(kappa) + 1.0)
        w = w0.copy(); w_prev = w0.copy(); fev = 0; conv = False
        t0 = time.perf_counter(); it = 0
        for it in range(1, max_iter + 1):
            yv = w + beta_m * (w - w_prev)
            g = obj.grad(yv); fy = obj.value(yv); fev += 1
            if float(np.linalg.norm(g)) < tol:
                conv = True; break
            if mode == "fixed":
                step = 1.0 / L
            else:
                step, k = _armijo_count(obj.value, yv, fy, g, -g, 1.0, rho, c1); fev += k
            w_prev = w
            w = yv - step * g
        record("AGD", mode, w, fev, it, t0, conv)

    def run_newton(mode):
        w = w0.copy(); fev = 0; conv = False; t0 = time.perf_counter(); it = 0
        for it in range(1, 100 + 1):
            g = obj.grad(w); f = obj.value(w); fev += 1
            if float(np.linalg.norm(g)) < tol:
                conv = True; break
            H = obj.hessian(w)
            delta = np.linalg.solve(H, g)
            if mode == "fixed":
                t = 1.0
            else:
                t, k = _armijo_count(obj.value, w, f, g, -delta, 1.0, rho, c1); fev += k
            w = w - t * delta
        record("Newton", mode, w, fev, it, t0, conv)

    for mode in ("fixed", "backtracking"):
        run_gd(mode); run_agd(mode); run_newton(mode)
    return FixedVsBacktrack(variant, lam, rows)


# --------------------------------------------------------------------------- #
# T2.2 — three regimes of the step size (safe / gray / divergent)
# --------------------------------------------------------------------------- #
@dataclass
class ThreeRegimes:
    variant: str
    lam: float
    L: float                 # 2/L is the safe/gray boundary
    two_over_L: float
    two_over_lam: float      # gray/divergent boundary
    etas: list[float]
    histories: dict          # eta -> suboptimality array (f-f*), +inf if diverged
    f_star: float


def three_regimes(
    variant: str = "ridge",
    lam: float = 1e-3,
    etas: tuple[float, ...] = (0.05, 0.1,                     # < 2/L: an toàn
                               0.5, 1.0, 5.0, 100.0, 1500.0,  # 2/L < eta < 2/lam
                               2500.0, 5000.0),               # > 2/lam: phân kỳ
    max_iter: int = 4000,
    tol: float = 1e-10,
) -> ThreeRegimes:
    """Sweep the GD step across all three regimes on the REAL logistic:
      eta < 2/L (~1.0)          -> monotone descent (Descent Lemma, sufficient)
      2/L < eta < 2/lambda      -> non-monotone but still converges (curvature
                                    collapses off w0; the theory is simply SILENT)
      eta > 2/lambda (~2000)    -> genuine blow-up: at ||w||->inf, S->0, H->lambda I,
                                    the map becomes w<-(1-eta*lambda)w.
    The old deck stopped at 10/L~5 and never reached the real 2/lambda threshold."""
    ds = load_variant(variant)
    obj = LogisticObjective(ds.X_train, ds.y_train, lam=lam)
    w0 = np.zeros(obj.d)
    f_star, _ = _fstar(obj, w0)
    L = conditioning(obj.X, lam)["L"]
    histories = {}
    for eta in etas:
        res = gradient_descent(obj, w0, eta=eta, max_iter=max_iter, tol=tol)
        histories[eta] = np.maximum(np.asarray(res.f_history) - f_star, 1e-16)
    return ThreeRegimes(variant, lam, L, 2.0 / L, 2.0 / lam,
                        list(etas), histories, f_star)


# --------------------------------------------------------------------------- #
# T2.3 — affine invariance done cleanly at lambda = 0
# --------------------------------------------------------------------------- #
@dataclass
class AffineInvariance:
    lam: float
    kappa_std: float         # local kappa at w* (z-scored design)
    kappa_raw: float         # local kappa at w* (raw, unscaled design)
    newton_std: OptResult
    newton_raw: OptResult
    gd_raw: OptResult        # first-order crawls on the raw (ill-conditioned) design
    f_star_std: float
    f_star_raw: float
    iters_newton_std: int
    iters_newton_raw: int
    n_nonpos_std: int = 0    # eigenvalues of H(w*) that are <= 0 (numerical rank loss)
    n_nonpos_raw: int = 0


def affine_invariance(lam: float = 0.0, tol: float = 1e-10) -> AffineInvariance:
    """The clean version of the standardization slide. At lambda=0 the objective is
    pure log-loss, which IS affine-invariant, so Newton takes the SAME number of
    steps on the raw and z-scored designs (its trajectory is w~=D w). GD, which is
    NOT affine-invariant, crawls on the raw design because kappa there is ~1e6+.
    (At lambda>0 the Ridge term breaks the invariance, which is why the old slide's
    11-vs-10 was two different problems, not numerical noise.)"""
    res = {}
    for tag, variant in (("std", "ridge"), ("raw", "ridge_raw")):
        ds = load_variant(variant)
        obj = LogisticObjective(ds.X_train, ds.y_train, lam=lam)
        w0 = np.zeros(obj.d)
        # see standardization_contrast: the raw design can be numerically singular
        nt = newton(obj, w0, max_iter=50, tol=1e-12, line_search=True,
                    strict=(tag == "std"))
        f_star = float(obj.value(nt.w))
        ev = np.linalg.eigvalsh(obj.hessian(nt.w))
        # At lambda=0 and d=416 the Hessian at w* is numerically SINGULAR: the
        # smallest eigenvalues come back as tiny negatives, and ev[-1]/ev[0] then
        # returns a negative "condition number" (measured -2.3e16), which is not a
        # number anyone should put on a slide. Report the ratio over the strictly
        # positive part and carry the count of non-positive eigenvalues so the
        # degeneracy is stated rather than hidden inside a bogus kappa.
        n_nonpos = int((ev <= 0).sum())
        pos = ev[ev > 0]
        kappa = float(ev[-1] / pos[0]) if pos.size else float("inf")
        res[tag] = (obj, w0, nt, f_star, kappa, n_nonpos)

    obj_raw, w0_raw, nt_raw, fs_raw, k_raw, nnp_raw = res["raw"]
    L_raw = conditioning(obj_raw.X, lam if lam > 0 else 1e-12)["L"]
    gd_raw = gradient_descent(obj_raw, w0_raw, eta=1.0 / L_raw,
                              max_iter=20000, tol=tol)

    return AffineInvariance(
        lam,
        kappa_std=res["std"][4], kappa_raw=k_raw,
        n_nonpos_std=res["std"][5], n_nonpos_raw=nnp_raw,
        newton_std=res["std"][2], newton_raw=nt_raw, gd_raw=gd_raw,
        f_star_std=res["std"][3], f_star_raw=fs_raw,
        iters_newton_std=_iters_to_tol(res["std"][2], res["std"][3], 1e-9),
        iters_newton_raw=_iters_to_tol(nt_raw, fs_raw, 1e-9))


# --------------------------------------------------------------------------- #
# T1.4 — SGD: batch-size sweep, epoch axis, 5-seed mean +/- std bands
# --------------------------------------------------------------------------- #
@dataclass
class SGDMultiSeed:
    variant: str
    lam: float
    eta0: float
    epochs: int
    n_seeds: int
    f_star: float
    epochs_axis: np.ndarray                # 0..epochs
    mean_gap: dict                         # batch -> mean (f-f*) per epoch
    std_gap: dict                          # batch -> std  (f-f*) per epoch
    gd_gap: np.ndarray                     # full-batch GD reference (per iter)
    dim_mean: np.ndarray                   # diminishing-step run, mean over seeds


def sgd_multiseed(
    variant: str = "ridge",
    lam: float = 1e-3,
    eta0: float = 0.5,
    batch_sizes: tuple[int, ...] = (32, 256, 4096),
    epochs: int = 60,
    n_seeds: int = 5,
    gamma_dim: float = 2e-4,
) -> SGDMultiSeed:
    """SGD honestly: constant step at several batch sizes, each averaged over
    n_seeds, plotted on the EPOCH axis with mean +/- std bands. Smaller batch = a
    higher variance floor (noise ball ~ eta/batch); the full-batch GD reference at
    the same eta is where the batches head as batch -> n (the self-check). One
    diminishing-step run shows annealing BELOW every constant-step floor.

    eta0=0.5 here is deliberately LARGER than the 0.1 that `sgd_hyper_sweep` picks
    for the benchmark: the floor is proportional to eta, so at the tuned step every
    batch size would pile up near 1e-5 and the whole point of the figure — that the
    floors separate by batch size — would be invisible. This figure is about the
    SHAPE of the floors, not about the best configuration."""
    ds = load_variant(variant)
    obj = LogisticObjective(ds.X_train, ds.y_train, lam=lam)
    w0 = np.zeros(obj.d)
    f_star, _ = _fstar(obj, w0)
    ax = np.arange(epochs + 1)

    # include the full batch (=n) so the sweep interpolates all the way to GD:
    # with batch=n, SGD is deterministic full-batch GD -> zero variance, and its
    # curve must overlay the GD reference (the free self-check).
    all_batches = list(batch_sizes) + [obj.n]

    mean_gap, std_gap = {}, {}
    for bs in all_batches:
        reps = 1 if bs >= obj.n else n_seeds        # full batch is deterministic
        stack = []
        for s in range(reps):
            r = sgd(obj, w0, eta0=eta0, batch_size=bs, epochs=epochs, gamma=0.0, seed=s)
            stack.append(np.maximum(np.asarray(r.f_history) - f_star, 1e-16))
        M = np.vstack(stack)
        mean_gap[bs] = M.mean(axis=0)
        std_gap[bs] = M.std(axis=0)

    # diminishing step (annealing) averaged over seeds, at the mid batch size
    mid = batch_sizes[len(batch_sizes) // 2]
    dim_stack = [np.maximum(
        np.asarray(sgd(obj, w0, eta0=eta0, batch_size=mid, epochs=epochs,
                       gamma=gamma_dim, seed=s).f_history) - f_star, 1e-16)
        for s in range(n_seeds)]
    dim_mean = np.vstack(dim_stack).mean(axis=0)

    # full-batch GD reference at the same eta (deterministic -> no band); log
    # epochs+1 points so it aligns with the SGD epoch axis (0..epochs).
    gd = gradient_descent(obj, w0, eta=eta0, max_iter=epochs + 1, tol=1e-14)
    gd_gap = np.maximum(np.asarray(gd.f_history) - f_star, 1e-16)

    return SGDMultiSeed(variant, lam, eta0, epochs, n_seeds, f_star,
                        ax, mean_gap, std_gap, gd_gap, dim_mean)


# --------------------------------------------------------------------------- #
# T1.5 — sklearn baselines WITH wall-clock time (objective value + seconds)
# --------------------------------------------------------------------------- #
@dataclass
class SklearnTiming:
    variant: str
    lam: float
    f_star: float
    rows: list[dict]         # solver, f, gap, iters, time


def _median_time(fn, reps: int = 3, warmup: bool = True) -> tuple:
    import time
    if warmup:
        fn()                 # warm-up
    ts, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out


def sklearn_timing(variant: str = "ridge", lam: float = 1e-3) -> SklearnTiming:
    """Objective value AND wall-clock for our Newton/AGD vs sklearn's lbfgs /
    newton-cholesky / saga on the identical objective (C = 1/(lambda*n),
    unpenalized intercept). f is our averaged form so all rows are comparable;
    gap = f - f* against the deep-Newton reference. Answers the assignment's
    'compare objective value AND computation time' for the library baseline."""
    from sklearn.linear_model import LogisticRegression

    ds = load_variant(variant)
    obj = LogisticObjective(ds.X_train, ds.y_train, lam=lam)
    w0 = np.zeros(obj.d)
    C = obj.sklearn_C()

    f_star = newton_reference(obj, w0).f_star

    rows = []

    # our hand-coded methods
    t_nt, nt = _median_time(lambda: newton(obj, w0, max_iter=100, tol=1e-12))
    rows.append(dict(solver="Newton (tự code)", f=float(obj.value(nt.w)),
                     gap=float(obj.value(nt.w) - f_star),
                     iters=len(nt.f_history), time=t_nt))
    L = conditioning(obj.X, lam)["L"]; mu = max(lam, 1e-12)

    # AGD and GD at the SAME hand-tuned settings the money plot uses, otherwise
    # this table and section 4.16 print two different iteration counts for the
    # same algorithm on the same problem.
    from .benchmark import gd_config, agd_config, sgd_config, SGD_EPOCHS
    acfg, gcfg, scfg = agd_config(L), gd_config(L), sgd_config()

    t_ag, ag = _median_time(lambda: accelerated_gd(obj, w0, L=acfg["L_eff"], mu=mu,
                                                   max_iter=20000, tol=1e-11))
    rows.append(dict(solver="AGD (tự code)", f=float(obj.value(ag.w)),
                     gap=float(obj.value(ag.w) - f_star),
                     iters=len(ag.f_history), time=t_ag))

    # GD gets ONE timed run and no warm-up: at 26 ms/iteration a 20,000-iteration
    # cap is ~9 min, so the usual median-of-3 would cost 35 min for a row whose
    # point is precisely that it never reaches the others' accuracy.
    t_gd, gd = _median_time(lambda: gradient_descent(obj, w0, eta=gcfg["eta"],
                                                     max_iter=20000, tol=1e-11),
                            reps=1, warmup=False)
    rows.append(dict(solver="GD (tự code)", f=float(obj.value(gd.w)),
                     gap=float(obj.value(gd.w) - f_star),
                     iters=len(gd.f_history), time=t_gd,
                     capped=not gd.converged))

    t_sg, sg = _median_time(lambda: sgd(obj, w0, eta0=scfg["eta0"],
                                        batch_size=scfg["batch"], epochs=SGD_EPOCHS,
                                        gamma=scfg["gamma"], seed=0))
    rows.append(dict(solver="SGD (tự code)", f=float(obj.value(sg.w)),
                     gap=float(obj.value(sg.w) - f_star),
                     iters=len(sg.f_history), time=t_sg, capped=True))

    def sk(solver):
        clf = LogisticRegression(C=C, penalty="l2", solver=solver,
                                 fit_intercept=True, tol=1e-10, max_iter=5000)
        clf.fit(ds.X_train, ds.y_train)
        w = np.concatenate([clf.coef_[0], clf.intercept_])
        return obj.value(w), int(clf.n_iter_[0])

    for label, solver in (("sklearn lbfgs", "lbfgs"),
                          ("sklearn newton-cholesky", "newton-cholesky"),
                          ("sklearn saga", "saga")):
        try:
            t, (fval, nit) = _median_time(lambda s=solver: sk(s))
            rows.append(dict(solver=label, f=float(fval), gap=float(fval - f_star),
                             iters=nit, time=t))
        except Exception as exc:                 # solver not in this sklearn build
            rows.append(dict(solver=label, f=float("nan"), gap=float("nan"),
                             iters=-1, time=float("nan"), error=str(exc)[:60]))

    return SklearnTiming(variant, lam, f_star, rows)


# --------------------------------------------------------------------------- #
# T2.4 — break-even dimension d* + BLAS-3/BLAS-2 accounting (mostly on paper)
# --------------------------------------------------------------------------- #
@dataclass
class Breakeven:
    kappa_star: float
    eps: float
    k_newton: int
    d_star: float            # sqrt(kappa*) ln(1/eps) / k_N
    n: int
    d_cube_ratio: dict       # d -> 3n/d  (how invisible d^3 is)
    blas: list               # gate.json blas rows (measured GEMM vs GEMV)


def breakeven_analysis(
    variant: str = "ridge",
    lam: float = 1e-3,
    eps: float = 1e-10,
    k_newton: int = 10,
    d_probe: tuple[int, ...] = (40, 160, 640),
) -> Breakeven:
    """Solve the crossover on paper: Newton k_N(nd^2+d^3/3) vs AGD sqrt(kappa)*nd.
    d* ~ sqrt(kappa*) ln(1/eps)/k_N. Uses the HONEST local kappa* (not the inflated
    bound), and reports 3n/d to show d^3 is negligible over the whole feasible
    range, plus the measured BLAS GEMM/GEMV numbers from the gate."""
    import json
    ds = load_variant(variant)
    obj = LogisticObjective(ds.X_train, ds.y_train, lam=lam)
    w_star = solve_deep(obj)
    ev = np.linalg.eigvalsh(obj.hessian(w_star))
    kappa_star = float(ev[-1] / ev[0])
    n = obj.n

    d_star = float(np.sqrt(kappa_star) * np.log(1.0 / eps) / k_newton)
    d_cube = {int(d): 3.0 * n / d for d in d_probe}

    blas = []
    gate = ARTIFACTS / "gate.json"
    if gate.exists():
        blas = json.loads(gate.read_text()).get("blas", [])

    return Breakeven(kappa_star, eps, k_newton, d_star, n, d_cube, blas)


# --------------------------------------------------------------------------- #
# T3.1 — adaptive family (AdaGrad / RMSprop / Adam) vs GD / AGD / Newton
# --------------------------------------------------------------------------- #
@dataclass
class AdaptiveFamily:
    variant: str
    lam: float
    f_star: float
    eta_grid: list[float]
    best_eta: dict          # method -> chosen eta (lowest final f - f*)
    sweep: list[dict]       # method, eta, iters, f_gap, converged
    runs: dict              # method -> OptResult at its best eta (incl. baselines)


def adaptive_family(
    variant: str = "ridge",
    lam: float = 1e-3,
    methods: tuple[str, ...] = ("AdaGrad", "RMSprop", "Adam"),
    eta_grid: tuple[float, ...] = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0),
    max_iter: int = 3000,
    tol: float = 1e-10,
) -> AdaptiveFamily:
    """Full-batch AdaGrad / RMSprop / Adam, each given its OWN best step size from
    a shared grid, then plotted against the GD / AGD / Newton baselines on the same
    axes (same objective, same w0=0, same f*).

    Full batch on purpose: it isolates the diagonal preconditioner from minibatch
    noise, so the comparison answers "does a cheap diagonal P help on THIS
    problem?" rather than re-running the SGD variance story. The answer is set up
    by the pipeline: X is already z-scored, so the per-coordinate rescaling these
    methods learn has largely been done in advance.

    EXPECTED WARNING: `_warn_if_on_grid_edge` fires for RMSprop every run, because
    its optimum is eta=0.003, the smallest point of the grid. That has been checked
    by hand rather than assumed — on ridge at n=75,026, d=415, budget 3000:

        eta=3e-4 -> f-f* = 1.6e-1     eta=1e-3 -> 2.9e-3     eta=3e-3 -> 8.7e-4

    so going lower is worse and 0.003 is a genuine interior optimum bracketed by
    1e-3 below and 0.01 above. The grid is not widened only because doing so costs
    a full 60-minute re-run of all five methods to add points that lose. Re-check
    this if the objective, lambda, or the matrix changes — do not just silence it.
    """
    from .optimizers import adaptive           # local import: keeps module import light
    ds = load_variant(variant)
    obj = LogisticObjective(ds.X_train, ds.y_train, lam=lam)
    w0 = np.zeros(obj.d)
    f_star, _ = _fstar(obj, w0)
    cond = conditioning(obj.X, lam)
    L, mu = cond["L"], max(cond["mu"], lam)

    sweep, best_eta, runs = [], {}, {}
    for m in methods:
        best = None
        for eta in eta_grid:
            r = adaptive(obj, w0, method=m, eta=eta, max_iter=max_iter, tol=tol)
            gap = float(r.f_history[-1] - f_star)
            if not np.isfinite(gap):
                gap = float("inf")
            sweep.append(dict(method=m, eta=float(eta), iters=len(r.f_history),
                              f_gap=gap, converged=bool(r.converged)))
            # Rank converged runs by ITERATIONS, not by final gap: several etas
            # bottom out at machine precision, where the gap differs only by
            # float noise and picking the "smallest" is meaningless. Runs that
            # never hit tol are ranked behind those that did, by final gap.
            key = (0, len(r.f_history), gap) if r.converged else (1, 0, gap)
            if best is None or key < best[0]:
                best = (key, float(eta), r)
        best_eta[m], runs[m] = best[1], best[2]
        _warn_if_on_grid_edge(best_eta[m], eta_grid, f"{m} eta")

    # Baselines on exactly the same objective / start / budget. GD gets the step
    # the trial-and-error study picked (benchmark.gd_config), NOT 1/L: every
    # adaptive method here is quoted at its own best eta from a sweep, so pinning
    # GD to a bound the deck itself shows to be pessimistic would compare tuned
    # methods against an untuned baseline. AGD keeps 1/L because its momentum
    # constant is derived with that step — see optim/optimizers/agd.py.
    from .benchmark import gd_config, agd_config
    runs["GD"] = gradient_descent(obj, w0, eta=gd_config(L)["eta"],
                                  max_iter=max_iter, tol=tol)
    # AGD likewise at its swept pair, fed in as L_eff = 1/t so beta is recomputed
    # to match the step (see benchmark.agd_config). Leaving it at 1/L here while
    # the money plot uses the tuned pair would print two different AGD iteration
    # counts on two slides of the same deck.
    runs["AGD"] = accelerated_gd(obj, w0, L=agd_config(L)["L_eff"], mu=mu,
                                 max_iter=max_iter, tol=tol)
    runs["Newton"] = newton(obj, w0, max_iter=100, tol=1e-13)
    return AdaptiveFamily(variant, lam, f_star, list(eta_grid), best_eta, sweep, runs)


# --------------------------------------------------------------------------- #
# T3.2 — SGD hyperparameter sweep: WHERE eta0 / batch / gamma COME FROM
# --------------------------------------------------------------------------- #
@dataclass
class SGDHyper:
    variant: str
    lam: float
    epochs: int
    tol: float
    f_star: float
    eta_grid: list[float]
    batch_grid: list[int]
    gamma_grid: list[float]
    grid: list[dict]         # eta0 x batch rows (at the chosen gamma)
    gamma_rows: list[dict]   # gamma rows at the chosen (eta0, batch)
    best: dict               # the config run_benchmark should use
    eta_max_stable: float    # largest eta0 in the grid that did not blow up


def sgd_hyper_sweep(
    variant: str = "ridge",
    lam: float = 1e-3,
    eta_grid: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
    batch_grid: tuple[int, ...] = (64, 256, 1024, 4096),
    gamma_grid: tuple[float, ...] = (0.0, 5e-5, 2e-4, 1e-3, 5e-3),
    gamma: float = 2e-4,
    epochs: int = 50,
    tol: float = 1e-4,
    seed: int = 0,
) -> SGDHyper:
    """Grid-search SGD's (eta0, batch, gamma) instead of hard-coding them.

    Every other method in this project gets its step size from theory: GD and AGD
    use 1/L, Newton uses t=1 damped by Armijo. SGD has no such formula — the
    Robbins-Monro conditions (sum eta_k = inf, sum eta_k^2 < inf) constrain the
    SCHEDULE, not the constant. So the constant has to be measured, and this is
    where the numbers in run_benchmark come from.

    Budget is fixed at `epochs` epochs for every cell, so the comparison is at
    equal cost (one epoch = one pass over the data = one full-gradient equivalent),
    and the ranking is simply the gap f - f* each cell ends at. `epochs_to_tol`
    (at the loose tol=1e-4) is reported alongside as the "how fast early on"
    column, but is NOT the criterion: SGD's whole story is that it stalls at a
    variance floor, so where it stalls is what the money plot shows.

    A cell whose objective goes non-finite is recorded as diverged, not dropped:
    the eta0 at which that starts is itself a reported number (eta_max_stable).
    """
    ds = load_variant(variant)
    obj = LogisticObjective(ds.X_train, ds.y_train, lam=lam)
    w0 = np.zeros(obj.d)
    f_star, _ = _fstar(obj, w0)

    def _row(eta0, bs, gm):
        t0 = time.perf_counter()
        r = sgd(obj, w0, eta0=eta0, batch_size=bs, epochs=epochs, gamma=gm, seed=seed)
        wall = time.perf_counter() - t0
        gap = float(r.f_history[-1] - f_star)
        diverged = not np.isfinite(gap)
        return dict(eta0=float(eta0), batch=int(bs), gamma=float(gm),
                    final_gap=(float("inf") if diverged else gap),
                    epochs_to_tol=_iters_to_tol(r, f_star, tol),
                    time_s=float(wall), diverged=bool(diverged))

    grid = [_row(e, bs, gamma) for bs in batch_grid for e in eta_grid]

    # rank by where the run ENDS after an equal budget (diverged cells sort last)
    def _key(row):
        return (1 if row["diverged"] else 0, row["final_gap"])

    best_cell = min(grid, key=_key)

    stable = [r["eta0"] for r in grid if not r["diverged"]]
    eta_max_stable = max(stable) if stable else float("nan")

    # decay-rate sweep at the winning (eta0, batch): gamma=0 is the constant step
    # that Robbins-Monro says must floor, and it should visibly lose here.
    gamma_rows = [_row(best_cell["eta0"], best_cell["batch"], g) for g in gamma_grid]
    best_gamma = min(gamma_rows, key=_key)

    best = dict(eta0=best_cell["eta0"], batch=best_cell["batch"],
                gamma=best_gamma["gamma"], epochs=epochs,
                final_gap=best_gamma["final_gap"],
                epochs_to_tol=best_gamma["epochs_to_tol"])
    _warn_if_on_grid_edge(best["eta0"], eta_grid, "SGD eta0")
    _warn_if_on_grid_edge(best["batch"], batch_grid, "SGD batch")
    _warn_if_on_grid_edge(best["gamma"], gamma_grid, "SGD gamma")
    return SGDHyper(variant, lam, epochs, tol, f_star, list(eta_grid),
                    list(batch_grid), list(gamma_grid), grid, gamma_rows, best,
                    float(eta_max_stable))
