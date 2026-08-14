"""Slide figures from a BenchmarkResult. Saves to artifacts/figures/.

Key plots (see CLAUDE.md):
  * convergence_curves -> log(f - f*) vs iteration AND vs cumulative work
    (n_grad / n_hess) — the core cost-vs-rate story.
  * l1_convergence     -> subgradient/ISTA/FISTA/CD suboptimality + sparsity path.
The parameter-sweep and honesty-study figures live alongside their drivers in
optim/experiments.py (step_sweep, quadratic_divergence, newton_init, sgd_sweep,
standardization, dim_scaling). Always log y-axis; linear scale hides the rates.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parent.parent / "artifacts" / "figures"

# stable, colour-blind-friendly assignment per optimizer
_STYLE = {
    "Newton": ("#d62728", "o"),
    "AGD":    ("#1f77b4", "s"),
    "GD":     ("#2ca02c", "^"),
    "SGD":    ("#9467bd", "d"),
}
_ITER_METHODS = ["Newton", "AGD", "GD", "SGD"]
_L1_STYLE = {"ISTA": "#ff7f0e", "FISTA": "#1f77b4", "CD": "#2ca02c",
             "Subgradient": "#7f7f7f"}


def _save(fig, name: str, save: bool):
    if not save:
        return fig
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def convergence_curves(bench, save: bool = True):
    """Two panels: suboptimality vs iteration (rate), and vs wallclock time
    (honest cost — this is where Newton's O(d^3) Hessian solve shows up, so a
    method that needs few iterations can still be slow)."""
    fs = bench.f_star
    fig, (ax_it, ax_work) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in _ITER_METHODS:
        res = bench.results.get(name)
        if res is None:
            continue
        colour, marker = _STYLE[name]
        sub = res.suboptimality(fs)
        ax_it.semilogy(range(len(sub)), sub, color=colour, label=name,
                       marker=marker, markevery=max(1, len(sub) // 12), ms=4)
        ax_work.semilogy(res.time_s, sub, color=colour, label=name,
                         marker=marker, markevery=max(1, len(sub) // 12), ms=4)
    kappa = bench.conditioning["kappa_upper"]
    ax_it.set(xlabel="iteration $k$", ylabel=r"$f(w_k)-f^\star$",
              title=f"{bench.variant}: convergence ($\\kappa\\approx${kappa:.0f})")
    ax_work.set(xlabel="wallclock time (s)",
                title="cost-adjusted (Newton pays for Hessians)")
    for ax in (ax_it, ax_work):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    if save:
        FIGURES.mkdir(parents=True, exist_ok=True)
        out = FIGURES / f"convergence_{bench.variant}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out
    return fig


def l1_convergence(bench, save: bool = True):
    """L1 (Lasso) figure: ISTA vs FISTA suboptimality log(F - F*) — showing the
    O(1/k) -> O(1/k^2) speedup — plus how many coefficients each drives to zero
    (sparsity path). `bench` is an L1BenchmarkResult."""
    # f* may be beaten by CD (it converges deeper than the FISTA reference), so
    # anchor the log-plot to the best value any method reached.
    fs = min([bench.f_star] + [min(r.f_history) for r in bench.results.values()])
    fig, (ax_sub, ax_nnz) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in ("Subgradient", "ISTA", "FISTA", "CD"):
        res = bench.results.get(name)
        if res is None:
            continue
        colour = _L1_STYLE[name]
        lbl = name
        if name == "Subgradient" and np.isfinite(getattr(bench, "subgrad_eta0", np.nan)):
            lbl = fr"Subgradient ($\eta_0$={bench.subgrad_eta0:g})"
        sub = np.maximum(np.asarray(res.f_history) - fs, 1e-16)
        ax_sub.semilogy(range(len(sub)), sub, color=colour, label=lbl,
                        ls="--" if name == "Subgradient" else "-")
        ax_nnz.plot(range(len(res.nnz)), res.nnz, color=colour, label=lbl,
                    ls="--" if name == "Subgradient" else "-")
    ax_sub.set(xlabel="iteration $k$ (epoch cho CD)", ylabel=r"$F(w_k)-F^\star$",
               title=(f"{bench.variant} L1: subgradient / ISTA / FISTA / CD "
                      f"($\\alpha$={bench.alpha:g})"))
    ax_nnz.axhline(bench.d_reg, ls=":", color="k", lw=1,
                   label=f"all {bench.d_reg} coefs")
    if bench.nnz_sklearn >= 0:
        ax_nnz.axhline(bench.nnz_sklearn, ls="--", color="grey", lw=1,
                       label=f"saga nnz={bench.nnz_sklearn}")
    ax_nnz.set(xlabel="iteration $k$", ylabel="nonzero coefficients",
               title="sparsity path (redundant columns zeroed)")
    for ax in (ax_sub, ax_nnz):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    return _save(fig, f"l1_{bench.variant}.png", save)


# --------------------------------------------------------------------------- #
# Experiment figures (see optim/experiments.py)
# --------------------------------------------------------------------------- #
def step_sweep_plot(sweep, save: bool = True):
    """GD suboptimality log(f - f*) vs iteration for each eta = mult/L. Runs that
    stay finite past mult=2 illustrate that the 2/L bound is conservative on the
    regularized logistic (curvature collapses from L to L_star off w0)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(sweep.mults)))
    for colour, m in zip(cmap, sweep.mults):
        res = sweep.runs[m]
        sub = res.suboptimality(sweep.f_star)
        ax.semilogy(range(len(sub)), sub, color=colour,
                    label=fr"$\eta={m:g}/L$")
    ax.set(xlabel="iteration $k$", ylabel=r"$f(w_k)-f^\star$",
           title=(f"{sweep.variant}: GD step-size sweep  "
                  fr"($L={sweep.L:.2f}$, $L_\star={sweep.L_star:.2f}$)"))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return _save(fig, f"gd_step_sweep_{sweep.variant}.png", save)


def quadratic_divergence_plot(qd, save: bool = True):
    """Objective of GD on the local quadratic model per eta. eta>2/L climbs to
    infinity (curvature is constant here) — the honest 'flies to infinity' plot."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for m in qd.mults:
        hist = qd.histories[m]
        stable = m < 2.0
        ax.semilogy(range(len(hist)), np.maximum(hist, 1e-16),
                    label=fr"$\eta={m:g}/L$",
                    lw=2 if not stable else 1.3,
                    ls="-" if stable else "--")
    ax.set(xlabel="iteration $k$", ylabel=r"$q(w_k)$ (quadratic model)",
           title=fr"{qd.variant}: $\eta>2/L$ diverges on the quadratic ($L={qd.L:.2f}$)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, f"quad_divergence_{qd.variant}.png", save)


def newton_init_plot(ni, save: bool = True):
    """Pure vs damped Newton from a far w0: objective per iteration. Pure may
    overshoot early; damped is monotone. Both hit quadratic convergence near w*."""
    fig, (ax_obj, ax_sub) = plt.subplots(1, 2, figsize=(12, 4.5))
    for res, colour, name in [(ni.pure, "#d62728", "pure (t=1)"),
                              (ni.damped, "#1f77b4", "damped (Armijo)")]:
        ax_obj.plot(res.f_history, color=colour, label=name, marker="o", ms=3)
        sub = np.maximum(np.asarray(res.f_history) - ni.f_star, 1e-16)
        ax_sub.semilogy(range(len(sub)), sub, color=colour, label=name,
                        marker="o", ms=3)
    ax_obj.axhline(ni.f_star, ls=":", color="k", lw=1, label=r"$f^\star$")
    ax_obj.set(xlabel="iteration $k$", ylabel=r"$f(w_k)$",
               title=fr"Newton from far $w_0$ ($\|w_0\|\approx{ni.w0_scale:g}\sqrt{{d}}$)")
    ax_sub.set(xlabel="iteration $k$", ylabel=r"$f(w_k)-f^\star$",
               title="quadratic convergence once close")
    for ax in (ax_obj, ax_sub):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    return _save(fig, f"newton_init_{ni.variant}.png", save)


def sgd_sweep_plot(sweep, save: bool = True):
    """Two panels: (left) constant steps plateau at a variance floor proportional
    to eta while the diminishing step anneals below them; (right) larger batches
    give a smoother tail."""
    fig, (ax_floor, ax_batch) = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, res in sweep.floor_runs.items():
        sub = np.maximum(np.asarray(res.f_history) - sweep.f_star, 1e-16)
        ls = "-" if "diminishing" in label else "--"
        lw = 2.2 if "diminishing" in label else 1.4
        ax_floor.semilogy(range(len(sub)), sub, label=label, ls=ls, lw=lw)
    ax_floor.set(xlabel="epoch", ylabel=r"$f(w_k)-f^\star$",
                 title=f"{sweep.variant}: constant step -> variance floor")
    for label, res in sweep.batch_runs.items():
        sub = np.maximum(np.asarray(res.f_history) - sweep.f_star, 1e-16)
        ax_batch.semilogy(range(len(sub)), sub, label=label)
    ax_batch.set(xlabel="epoch", title="diminishing step, varying batch size")
    for ax in (ax_floor, ax_batch):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, f"sgd_sweep_{sweep.variant}.png", save)


def standardization_plot(sc, save: bool = True):
    """Two panels, z-scored vs raw X, each with GD/AGD/Newton suboptimality. On
    raw X kappa explodes so GD/AGD stall; Newton is affine invariant and converges
    at the same rate on both — standardization is mandatory for first-order methods."""
    fig, (ax_std, ax_raw) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    panels = [(ax_std, sc.std_runs, sc.f_star_std, sc.kappa_std, "z-scored (ridge)"),
              (ax_raw, sc.raw_runs, sc.f_star_raw, sc.kappa_raw, "raw (unscaled)")]
    for ax, runs, fstar, kappa, title in panels:
        for name in ("Newton", "AGD", "GD"):
            res = runs.get(name)
            if res is None:
                continue
            colour, marker = _STYLE[name]
            sub = np.maximum(np.asarray(res.f_history) - fstar, 1e-16)
            ax.semilogy(range(len(sub)), sub, color=colour, label=name,
                        marker=marker, markevery=max(1, len(sub) // 10), ms=4)
        ax.set(xlabel="iteration $k$",
               title=f"{title}  ($\\kappa\\approx${kappa:.0f})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    ax_std.set_ylabel(r"$f(w_k)-f^\star$")
    fig.tight_layout()
    return _save(fig, "standardization_contrast.png", save)


def dim_scaling_plot(scan, save: bool = True):
    """Two cost axes vs dimension d. Left: per-iteration wallclock — Newton
    (O(nd^2+d^3)) rises far steeper than first-order (O(nd)), the robust
    'expensive per step' signal. Right: total time-to-tolerance — Newton still
    wins here because it needs ~8 iterations regardless of kappa (honest: the
    total-time reversal needs d^3 to dominate, i.e. d -> n)."""
    d = np.array([r["d"] for r in scan.rows])
    fig, (ax_pi, ax_tot) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_pi.plot(d, [r["pi_newton"] for r in scan.rows], color="#d62728",
               marker="o", label="Newton  $O(nd^2{+}d^3)$")
    ax_pi.plot(d, [r["pi_agd"] for r in scan.rows], color="#1f77b4",
               marker="s", label="AGD  $O(nd)$")
    ax_pi.plot(d, [r["pi_gd"] for r in scan.rows], color="#2ca02c",
               marker="^", label="GD  $O(nd)$")
    ax_pi.set(xlabel="dimension $d$", ylabel="wallclock per iteration (s)",
              title=f"{scan.variant}: per-iteration cost grows with $d$")

    ax_tot.plot(d, [r["t_newton"] for r in scan.rows], color="#d62728",
                marker="o", label="Newton")
    ax_tot.plot(d, [r["t_agd"] for r in scan.rows], color="#1f77b4",
                marker="s", label="AGD")
    ax_tot.set(xlabel="dimension $d$",
               ylabel=f"time to $f-f^\\star<{scan.tol:g}$  (s)",
               title="total time (Newton's few iterations still win)")
    for ax in (ax_pi, ax_tot):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, f"dim_scaling_{scan.variant}.png", save)


# ========================================================================== #
# REVISION LẦN 1 — figures for the new slides (see optim/experiments.py)
# ========================================================================== #
def kappa_honesty_plot(kh, save: bool = True):
    """T1.1 — GD suboptimality with the two theory rates overlaid: the deck bound
    (1-1/kappa_bound)^t is far too slow; the local (1-1/kappa*)^t tracks the real
    curve. The single most convincing 'theory meets data' figure of the talk."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    sub = np.maximum(np.asarray(kh.gd.f_history) - kh.f_star, 1e-16)
    t = np.arange(len(sub))
    y0 = sub[0]
    ax.semilogy(t, sub, color="#2ca02c", lw=2, label="GD thực nghiệm")
    ax.semilogy(t, np.maximum(y0 * (1 - 1 / kh.kappa_bound) ** t, 1e-16), "--",
                color="#d62728", lw=1.6,
                label=fr"lý thuyết $\kappa_{{bound}}={kh.kappa_bound:.0f}$")
    ax.semilogy(t, np.maximum(y0 * (1 - 1 / kh.kappa_star) ** t, 1e-16), "-.",
                color="#1f77b4", lw=1.6,
                label=fr"lý thuyết $\kappa_\star={kh.kappa_star:.0f}$")
    ax.axhline(kh.eps, ls=":", color="grey", lw=1)
    ax.set(xlabel="iteration $k$", ylabel=r"$f(w_k)-f^\star$",
           title=(f"{kh.variant}: GD vs hai cận tốc độ  "
                  fr"(đo {kh.meas_gd} vòng, dự đoán $\kappa_\star$≈{kh.pred_gd_star:.0f})"))
    ax.set_xlim(0, min(len(sub), int(3 * kh.meas_gd) if kh.meas_gd > 0 else len(sub)))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save(fig, f"kappa_honesty_{kh.variant}.png", save)


def kappa_sweep_plot(ks, save: bool = True):
    """T2.1 — log-log iterations-to-tolerance vs kappa*, fitted slopes annotated.
    GD slope ~1, AGD slope ~1/2: the direct empirical confirmation of Nesterov."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    kap = np.array(ks.kappa)
    ig = np.array(ks.it_gd, float)
    ia = np.array(ks.it_agd, float)
    mg, ma = ig > 0, ia > 0
    ax.loglog(kap[mg], ig[mg], "o", color="#2ca02c", ms=7, label="GD (đo)")
    ax.loglog(kap[ma], ia[ma], "s", color="#1f77b4", ms=7, label="AGD (đo)")
    xs = np.array([kap[mg].min(), kap[mg].max()])
    cg = np.polyfit(np.log10(kap[mg]), np.log10(ig[mg]), 1)
    ca = np.polyfit(np.log10(kap[ma]), np.log10(ia[ma]), 1)
    ax.loglog(xs, 10 ** np.polyval(cg, np.log10(xs)), "--", color="#2ca02c", lw=1.3)
    ax.loglog(xs, 10 ** np.polyval(ca, np.log10(xs)), "--", color="#1f77b4", lw=1.3)
    ax.annotate(fr"GD: độ dốc ${ks.slope_gd:.2f}$ (lý thuyết 1)",
                xy=(0.05, 0.90), xycoords="axes fraction", color="#2ca02c", fontsize=11)
    ax.annotate(fr"AGD: độ dốc ${ks.slope_agd:.2f}$ (lý thuyết 0.5)",
                xy=(0.05, 0.82), xycoords="axes fraction", color="#1f77b4", fontsize=11)
    ax.set(xlabel=r"số điều kiện cục bộ $\kappa_\star$",
           ylabel=r"số vòng đến $f-f^\star<10^{-8}$",
           title=f"{ks.variant}: quét $\\lambda$ $\\Rightarrow$ $\\kappa_\\star$; kiểm chứng định lý Nesterov")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, f"kappa_sweep_{ks.variant}.png", save)


# Zone -> (colormap, human label). The zone is carried by the colour FAMILY and
# the curves inside a zone are separated by shade + dash pattern: one flat colour
# per zone made the five gray-zone curves literally indistinguishable on the
# slide, which is the one thing this figure exists to show.
# Shade ranges are per-zone so the darkest orange never reaches the brown that
# reads as red: the gray zone stays amber, divergence owns red outright.
_REGIME_SHADES = {"an toàn": (0.50, 0.95), "xám": (0.45, 0.85), "phân kỳ": (0.72, 0.95)}
_REGIME_STYLE = {
    "an toàn": ("Greens", ["-", "--", "-.", ":"]),
    "xám":     ("Oranges", ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]),
    "phân kỳ": ("Reds", ["-", "--", "-.", ":"]),
}


def three_regimes_plot(tr, save: bool = True):
    """T2.2 — GD across the three step-size regimes on the REAL logistic: safe
    (eta<2/L), gray (2/L<eta<2/lambda, finite but wild), divergent (eta>2/lambda)."""
    def zone_of(eta):
        if eta < tr.two_over_L:
            return "an toàn"
        return "xám" if eta < tr.two_over_lam else "phân kỳ"

    # group first so shades can be spread across however many curves a zone has
    groups = {}
    for eta in tr.etas:
        groups.setdefault(zone_of(eta), []).append(eta)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    handles = []
    for zone in ("an toàn", "xám", "phân kỳ"):
        etas = groups.get(zone)
        if not etas:
            continue
        cmap_name, dashes = _REGIME_STYLE[zone]
        cmap = plt.get_cmap(cmap_name)
        # avoid the washed-out bottom and the near-black top of the ramp
        lo, hi = _REGIME_SHADES[zone]
        shades = np.linspace(lo, hi, len(etas)) if len(etas) > 1 else [(lo + hi) / 2]
        for i, (eta, shade) in enumerate(zip(etas, shades)):
            h = np.maximum(tr.histories[eta], 1e-16)
            (ln,) = ax.semilogy(range(len(h)), h, color=cmap(shade),
                                ls=dashes[i % len(dashes)], lw=1.6,
                                label=fr"$\eta={eta:g}$")
            handles.append((zone, ln))

    ax.axhline(1.0, color="0.6", lw=0.8, ls=":", zorder=0)

    # The divergent curves leave the top of the axes within a couple of iterations,
    # so on the slide they read as a single vertical tick and the "phân kỳ" legend
    # column looks unused. Say in words what the curve has no room to show.
    top = 1e6
    if "phân kỳ" in groups:
        firsts = []
        for eta in groups["phân kỳ"]:
            over = np.flatnonzero(np.asarray(tr.histories[eta]) > top)
            firsts.append(int(over[0]) if over.size else -1)
        if all(k >= 0 for k in firsts):
            ax.annotate(fr"$\eta\geq{min(groups['phân kỳ']):g}$: vượt $10^{{6}}$ "
                        fr"sau $\leq{max(firsts)}$ bước",
                        # the top strip is occupied by the eta=1500 sawtooth; the
                        # empty band under the converging curves is the only clear space
                        xy=(0.60, 0.10), xycoords="axes fraction", fontsize=9,
                        color=plt.get_cmap("Reds")(0.85), fontweight="bold")
    ax.set(xlabel="iteration $k$", ylabel=r"$f(w_k)-f^\star$",
           title=(fr"{tr.variant}: ba chế độ bước nhảy "
                  fr"($2/L={tr.two_over_L:.3f}$, $2/\lambda={tr.two_over_lam:.0f}$)"))
    ax.set_ylim(1e-12, 1e6)
    ax.grid(True, which="both", alpha=0.3)

    # One legend COLUMN per zone. matplotlib fills a multi-column legend
    # column-major, so each group is padded with invisible entries to the length
    # of the largest one — otherwise the columns straddle zone boundaries and an
    # eta from the gray zone ends up printed under the "an toàn" header.
    from matplotlib.lines import Line2D
    zones = [z for z in ("an toàn", "xám", "phân kỳ") if z in groups]
    rows = max(len(groups[z]) for z in zones)
    blank = lambda: Line2D([], [], ls="none", marker="none")
    ordered, labels = [], []
    for zone in zones:
        col = [ln for z, ln in handles if z == zone]
        head = Line2D([], [], ls="none", marker="none")
        ordered.append(head)
        labels.append(f"{zone} ({len(col)})")
        for ln in col:
            ordered.append(ln)
            labels.append(ln.get_label())
        for _ in range(rows - len(col)):
            ordered.append(blank())
            labels.append("")
    leg = ax.legend(ordered, labels, fontsize=8, ncol=len(zones), loc="lower left",
                    columnspacing=1.4, handlelength=2.4, borderpad=0.6,
                    labelspacing=0.35)
    for i, txt in enumerate(leg.get_texts()):
        if i % (rows + 1) == 0:          # the header row of each column
            txt.set_fontweight("bold")
    fig.tight_layout()
    return _save(fig, f"three_regimes_{tr.variant}.png", save)


def agd_schemes_plot(runs, f_star, fname="agd_schemes.png", save: bool = True):
    """Constant-beta AGD vs the (k-2)/(k+1) scheme, on iterations AND wall-clock.

    `runs` maps a label to an OptResult. The point of drawing both axes is that
    the two schemes have different per-iteration cost only through the objective
    evaluation, so a gap that survives the time axis is a real gap.
    """
    fig, (ax_it, ax_t) = plt.subplots(1, 2, figsize=(12, 4.5))
    colours = ["#1f77b4", "#9467bd", "#d95f02", "#2ca02c"]
    styles = ["-", "--", "-", "-."]
    for (label, r), c, ls in zip(runs.items(), colours, styles):
        g = np.maximum(np.asarray(r.f_history) - f_star, 1e-16)
        ax_it.semilogy(range(len(g)), g, color=c, ls=ls, lw=1.7, label=label)
        ax_t.loglog(np.maximum(r.time_s, 1e-4), g, color=c, ls=ls, lw=1.7, label=label)
    ax_it.set(xlabel="Số bước", ylabel=r"$f(w_k)-f^\star$",
              title="Hai lược đồ momentum: trục số bước")
    ax_t.set(xlabel="Thời gian (s)", title="cùng dữ liệu, trục thời gian")
    for ax in (ax_it, ax_t):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, fname, save)


def affine_plot(ai, save: bool = True):
    """T2.3 — at lambda=0 Newton takes the SAME steps on z-scored and raw designs
    (affine invariant), while GD on the raw (kappa~1e18) design barely moves."""
    fig, (ax_nt, ax_gd) = plt.subplots(1, 2, figsize=(12, 4.6))
    ns = np.maximum(np.asarray(ai.newton_std.f_history) - ai.f_star_std, 1e-16)
    nr = np.maximum(np.asarray(ai.newton_raw.f_history) - ai.f_star_raw, 1e-16)
    ax_nt.semilogy(range(len(ns)), ns, "o-", color="#1f77b4", ms=6,
                   label=fr"Newton z-scored ($\kappa$={ai.kappa_std:.1e})")
    ax_nt.semilogy(range(len(nr)), nr, "s--", color="#d62728", ms=6,
                   label=fr"Newton raw ($\kappa$={ai.kappa_raw:.1e})")
    ax_nt.set(xlabel="iteration $k$", ylabel=r"$f(w_k)-f^\star$",
              title=(f"Newton bất biến affine ($\\lambda$=0): "
                     f"{ai.iters_newton_std} vs {ai.iters_newton_raw} vòng"))
    gr = np.maximum(np.asarray(ai.gd_raw.f_history) - ai.f_star_raw, 1e-16)
    ax_gd.semilogy(range(len(nr)), nr, "s--", color="#d62728", ms=5, label="Newton raw")
    ax_gd.semilogy(range(len(gr)), gr, color="#2ca02c", lw=1.6, label="GD raw")
    ax_gd.set(xlabel="iteration $k$",
              title="GD KHÔNG bất biến: bò trên raw (điều kiện xấu)")
    for ax in (ax_nt, ax_gd):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, "affine_invariance.png", save)


def sgd_multiseed_plot(sm, save: bool = True):
    """T1.4 — SGD on the epoch axis, mean +/- std over seeds. Smaller batch = higher
    variance floor; batch=n overlays GD (self-check); diminishing step anneals below."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(sm.mean_gap)))
    for colour, bs in zip(cmap, sm.mean_gap):
        m, s = sm.mean_gap[bs], sm.std_gap[bs]
        is_full = bs == list(sm.mean_gap)[-1]
        ax.semilogy(sm.epochs_axis, m, color=colour, lw=2,
                    label=(f"batch={bs}" + (" (=n, ~GD)" if is_full else "")))
        ax.fill_between(sm.epochs_axis, np.maximum(m - s, 1e-16), m + s,
                        color=colour, alpha=0.22)
    ax.semilogy(sm.epochs_axis[:len(sm.dim_mean)], sm.dim_mean, "k-.", lw=1.8,
                label="bước giảm dần")
    ax.set(xlabel="epoch $= t\\,b/n$", ylabel=r"$f(w_k)-f^\star$",
           title=f"{sm.variant}: SGD — sàn phương sai $\\propto \\eta/$batch ({sm.n_seeds} seed)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, f"sgd_multiseed_{sm.variant}.png", save)


def money_plots(bench, save: bool = True):
    """T1.2 — the assignment's two required money plots as SEPARATE figures:
    objective vs iteration, and objective vs wall-clock (both log y)."""
    fs = bench.f_star
    outs = []
    for axis, fname, xlabel, title in (
        ("iter", f"money_iter_{bench.variant}.png", "iteration $k$",
         "Trên trục số vòng lặp"),
        ("time", f"money_time_{bench.variant}.png", "wall-clock (s)",
         "Trên trục thời gian (wall-clock)")):
        fig, ax = plt.subplots(figsize=(7, 4.6))
        # On the ITERATION axis, SGD's per-epoch step is not one unit of work
        # comparable to a Newton/GD/AGD iteration, so it is shown only on the
        # wall-clock axis (seconds ARE comparable) and on its own epoch slide.
        methods = _ITER_METHODS if axis == "time" else ["Newton", "AGD", "GD"]
        for name in methods:
            res = bench.results.get(name)
            if res is None:
                continue
            colour, marker = _STYLE[name]
            sub = res.suboptimality(fs)
            x = range(len(sub)) if axis == "iter" else res.time_s
            ax.semilogy(x, sub, color=colour, label=name, marker=marker,
                        markevery=max(1, len(sub) // 12), ms=4)
        if axis == "time":
            ax.set_xscale("log")
        kappa = bench.conditioning["kappa_upper"]
        ax.set(xlabel=xlabel, ylabel=r"$f(w_k)-f^\star$",
               title=f"{bench.variant}: {title}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        outs.append(_save(fig, fname, save))
    return outs


def breakeven_plot(be, save: bool = True):
    """T2.4 — left: analytic per-run cost of Newton (k_N n d^2) vs AGD
    (sqrt(kappa) n d) vs d, with the crossover d*; right: measured BLAS-3 (GEMM)
    vs BLAS-2 (GEMV) throughput explaining why Newton's FLOP disadvantage shrinks."""
    fig, (ax_c, ax_b) = plt.subplots(1, 2, figsize=(12, 4.6))
    d = np.logspace(1, 3.5, 100)
    n = be.n
    cost_newton = be.k_newton * (n * d * d + d ** 3 / 3)
    cost_agd = np.sqrt(be.kappa_star) * np.log(1 / be.eps) * n * d
    ax_c.loglog(d, cost_newton, color="#d62728", label=r"Newton $k_N(nd^2+d^3/3)$")
    ax_c.loglog(d, cost_agd, color="#1f77b4",
                label=r"AGD $\sqrt{\kappa_\star}\ln(1/\epsilon)\,nd$")
    ax_c.axvline(be.d_star, ls=":", color="k",
                 label=fr"$d^\star\approx{be.d_star:.0f}$")
    ax_c.set(xlabel="dimension $d$", ylabel="FLOP mô hình (per run)",
             title=fr"Điểm hòa vốn $d^\star$ ($\kappa_\star$={be.kappa_star:.0f})")
    ax_c.grid(True, which="both", alpha=0.3); ax_c.legend(fontsize=8)

    if be.blas:
        ds = [str(b["d"]) for b in be.blas]
        gemm = [b["gemm_gflops"] for b in be.blas]
        gemv = [b["gemv_gflops"] for b in be.blas]
        x = np.arange(len(ds))
        ax_b.bar(x - 0.2, gemm, 0.4, color="#d62728", label="GEMM (Hessian, BLAS-3)")
        ax_b.bar(x + 0.2, gemv, 0.4, color="#1f77b4", label="GEMV (gradient, BLAS-2)")
        ax_b.set_xticks(x); ax_b.set_xticklabels([f"d={s}" for s in ds])
        ax_b.set(ylabel="GFLOP/s (đo, 1 luồng)",
                 title="BLAS-3 nhanh hơn/FLOP $\\Rightarrow$ Newton rẻ hơn kỳ vọng")
        ax_b.grid(True, axis="y", alpha=0.3); ax_b.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, "breakeven.png", save)


_ADA_STYLE = {
    "AdaGrad": ("#ff7f0e", "v"),
    "RMSprop": ("#8c564b", "P"),
    "Adam":    ("#e377c2", "X"),
    "AdamW":   ("#7f7f7f", "*"),
    "AMSGrad": ("#17becf", "h"),
}


def adaptive_family_plot(af, save: bool = True):
    """T3.1 — the adaptive family (each at its own best step) against GD/AGD/Newton,
    on the iteration axis and the wall-clock axis. Same f*, same w0, same budget."""
    style = {**_STYLE, **_ADA_STYLE}
    # Follow whatever `adaptive_family` actually ran: a hard-coded list silently
    # drops a method the moment someone adds one to `methods=`.
    baselines = [m for m in ("Newton", "AGD", "GD") if m in af.runs]
    adaptive_methods = [m for m in _ADA_STYLE if m in af.runs]
    order = baselines + adaptive_methods
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, axis, xlabel in ((axes[0], "iter", "số bước $k$"),
                             (axes[1], "time", "thời gian (s)")):
        for name in order:
            res = af.runs.get(name)
            if res is None:
                continue
            colour, marker = style[name]
            sub = res.suboptimality(af.f_star)
            x = range(len(sub)) if axis == "iter" else res.time_s
            label = name if name not in af.best_eta else \
                f"{name} ($\\eta$={af.best_eta[name]:g})"
            ax.semilogy(x, sub, color=colour, label=label, marker=marker,
                        markevery=max(1, len(sub) // 10), ms=4, lw=1.4)
        ax.set(xlabel=xlabel, ylabel=r"$f(w_k)-f^\star$")
        ax.grid(True, which="both", alpha=0.3)
    axes[1].set_xscale("log")
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"{af.variant}: họ Ada/Adam so với GD, AGD, Newton "
                 f"($\\lambda$={af.lam:g}, mỗi thuật toán ở bước tốt nhất)", fontsize=11)
    fig.tight_layout()
    return _save(fig, f"adaptive_family_{af.variant}.png", save)


def sgd_hyper_plot(hp, save: bool = True):
    """Where SGD's hyperparameters come from — the figure that answers "why 0.5?".

    Left: final f - f* after an equal 50-epoch budget, as a function of eta0, one
    curve per batch size (the U shape: too small = too slow, too large = noise
    floor / blow-up). Right: the decay rate gamma at the winning (eta0, batch),
    including gamma=0, the constant step Robbins-Monro says cannot converge.
    """
    fig, (ax_eta, ax_gam) = plt.subplots(1, 2, figsize=(12, 4.5))

    floor = 1e-16
    for bs in hp.batch_grid:
        rows = sorted([r for r in hp.grid if r["batch"] == bs],
                      key=lambda r: r["eta0"])
        etas = [r["eta0"] for r in rows]
        gaps = [max(r["final_gap"], floor) if np.isfinite(r["final_gap"]) else np.nan
                for r in rows]
        ax_eta.loglog(etas, gaps, marker="o", ms=4, label=f"batch={bs}")
        # mark the cells that blew up, so divergence is visible rather than absent
        div = [r["eta0"] for r in rows if r["diverged"]]
        if div:
            top = ax_eta.get_ylim()[1]
            ax_eta.loglog(div, [top] * len(div), marker="x", ls="none",
                          color="k", ms=8, clip_on=False)
    best = hp.best
    ax_eta.plot([best["eta0"]], [max(best["final_gap"], floor)], marker="*",
                ms=17, color="k", ls="none", zorder=5,
                label=rf"chọn: $\eta_0={best['eta0']:g}$, batch$={best['batch']}$")
    ax_eta.set(xlabel=r"$\eta_0$", ylabel=rf"$f-f^\star$ sau {hp.epochs} epoch",
               title=f"{hp.variant}: quét $\\eta_0$ $\\times$ batch")

    rows = sorted(hp.gamma_rows, key=lambda r: r["gamma"])
    labels = [("0 (bước cố định)" if r["gamma"] == 0 else f"{r['gamma']:g}")
              for r in rows]
    vals = [max(r["final_gap"], floor) for r in rows]
    bars = ax_gam.bar(range(len(rows)), vals,
                      color=["#d62728" if r["gamma"] == 0 else "#1f77b4" for r in rows])
    ax_gam.set_yscale("log")
    ax_gam.set_xticks(range(len(rows)))
    ax_gam.set_xticklabels(labels, fontsize=8, rotation=20)
    ax_gam.set(xlabel=r"$\gamma$ (nhịp giảm bước, đếm theo mini-batch)",
               ylabel=rf"$f-f^\star$ sau {hp.epochs} epoch",
               title=rf"$\eta_0={best['eta0']:g}$, batch$={best['batch']}$: quét $\gamma$")
    for b, r in zip(bars, rows):
        ax_gam.annotate(f"{r['final_gap']:.1e}", (b.get_x() + b.get_width() / 2,
                        b.get_height()), ha="center", va="bottom", fontsize=7)

    ax_eta.grid(True, which="both", alpha=0.3)
    ax_gam.grid(True, which="both", axis="y", alpha=0.3)
    ax_eta.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, f"sgd_hyper_{hp.variant}.png", save)


# --------------------------------------------------------------------------- #
# Thử-và-sai chọn tham số (optim/tuning.py) — bố cục theo bài mẫu
# --------------------------------------------------------------------------- #
def choice_plot(ch, save: bool = True, fname: str | None = None):
    """Hai panel như bài mẫu: THÔ (f tuyến tính) | TINH (log(f - f*)).

    Panel trái để trục thẳng có chủ đích: ở lưới thô cái ta cần thấy là bước nào
    làm hàm mục tiêu VỌT LÊN, mà thang log sẽ nén mất chuyện đó. Panel phải mới
    dùng log, vì khi các bước đều hợp lý rồi thì chúng chỉ khác nhau ở tốc độ.
    """
    fig, (ax_c, ax_f) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    for v in ch.coarse.grid:
        h = np.asarray(ch.coarse.runs[v].f_history)
        h = np.where(np.isfinite(h), h, np.nan)
        ax_c.plot(range(len(h)), h, label=f"{v:g}", lw=1.6)
    ax_c.set(xlabel="Số bước", ylabel="$f(w_k)$",
             title=f"{ch.label}: lưới THÔ")
    # Trần của panel THÔ phải theo DỮ LIỆU, không cố định. Nó từng là f(w_0)*2.2 để
    # chừa chỗ nhìn đường vọt lên (lưới GD có t=10 phân kỳ); nhưng ở lưới nào không
    # có đường nào vọt — Newton chẳng hạn — thì 2,2 lần đó là khoảng trắng thuần.
    finite = [np.asarray(ch.coarse.runs[v].f_history) for v in ch.coarse.grid]
    finite = [h[np.isfinite(h)] for h in finite]
    lo = min(float(h.min()) for h in finite if h.size)
    hi_data = max(float(h.max()) for h in finite if h.size)
    f0 = float(ch.coarse.runs[ch.coarse.grid[0]].f_history[0])
    if hi_data <= f0 * 1.02:            # không đường nào vọt lên: bám sát dữ liệu
        hi = f0 + 0.08 * (f0 - lo)
    else:                               # có đường vọt: chừa chỗ, nhưng vẫn chặn trên
        hi = min(hi_data * 1.05, f0 * 2.2)
    ax_c.set_ylim(lo - 0.05 * (hi - lo), hi)

    for v in ch.fine.grid:
        g = ch.fine.gaps(v)
        lw, z = (2.4, 5) if v == ch.chosen else (1.3, 2)
        ax_f.semilogy(range(len(g)), g, label=f"{v:g}", lw=lw, zorder=z)
    ax_f.set(xlabel="Số bước", ylabel=r"$f(w_k)-f^\star$",
             title=f"{ch.label}: lưới TINH")

    for ax in (ax_c, ax_f):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, title=ch.coarse.param, title_fontsize=8)
    # KHÔNG nhúng dòng "-> Chọn ..." vào hình: trên slide nó nằm sát tiêu đề mục và
    # bị đè lên. Bài mẫu cũng đặt dòng đó là chữ của slide, không phải của hình.
    # Giá trị được chọn đã được tô đậm bằng nét dày trong panel tinh.
    fig.tight_layout()
    return _save(fig, fname or f"tune_{ch.label.split()[0].lower()}.png", save)


def bt_grid_plot(grid_runs, f_star, title, chosen, save: bool = True,
                 fname: str = "tune_backtracking_grid.png"):
    """Lưới (rho, c) của backtracking trên CẢ HAI trục: số bước | thời gian.

    Hai trục là bắt buộc chứ không phải cho đẹp: backtracking đổi số bước và số
    lần tính f theo hai chiều ngược nhau, nên cách nào thắng phụ thuộc vào việc
    ta đếm gì.
    """
    fig, (ax_it, ax_t) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(grid_runs)))
    for colour, ((rho, c), res) in zip(cmap, sorted(grid_runs.items())):
        g = np.maximum(np.asarray(res.f_history) - f_star, 1e-16)
        best = (rho, c) == chosen
        kw = dict(color=colour, lw=2.6 if best else 1.2, zorder=5 if best else 2,
                  label=rf"$\rho$={rho:g}, c={c:g}" + (" ←" if best else ""))
        ax_it.semilogy(range(len(g)), g, **kw)
        ax_t.semilogy(res.time_s, g, **kw)
    ax_it.set(xlabel="Số bước", ylabel=r"$f(w_k)-f^\star$", title=title)
    ax_t.set(xlabel="Thời gian (s)", title="cùng dữ liệu, trục thời gian")
    for ax in (ax_it, ax_t):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return _save(fig, fname, save)


def step_trace_plot(res, title, save: bool = True,
                    fname: str = "tune_step_trace.png"):
    """Độ dài bước backtracking THỰC SỰ chọn ở mỗi vòng.

    Đây là hình bài mẫu có mà ta chưa có, và nó trả lời đúng câu hỏi tự nhiên
    nhất về backtracking: nó có thật sự đổi bước không, hay chỉ luôn lấy t_init?
    """
    steps = np.asarray(res.steps, dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(range(len(steps)), steps, width=0.9)
    ax.axhline(steps.max(), ls=":", color="0.4", lw=1,
               label=f"lớn nhất = {steps.max():g}")
    ax.axhline(steps.mean(), ls="--", color="#d62728", lw=1.2,
               label=f"trung bình = {steps.mean():.3g}")
    ax.set(xlabel="Bước", ylabel="Độ dài bước", title=title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, fname, save)


def variants_plot(runs, f_star, title, save: bool = True, fname="tune_variants.png"):
    """So sánh các cách chọn bước của CÙNG một thuật toán, hai trục."""
    fig, (ax_it, ax_t) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for name, res in runs.items():
        g = np.maximum(np.asarray(res.f_history) - f_star, 1e-16)
        ax_it.semilogy(range(len(g)), g, label=name, lw=1.8)
        ax_t.semilogy(res.time_s, g, label=name, lw=1.8)
    ax_it.set(xlabel="Số bước", ylabel=r"$f(w_k)-f^\star$", title=title)
    ax_t.set(xlabel="Thời gian (s)", title="cùng dữ liệu, trục thời gian")
    for ax in (ax_it, ax_t):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, fname, save)


def money7_plot(runs, f_star, save: bool = True, fname="money7_iter.png",
                x="iter", title=None):
    """All seven hand-tuned configurations on one axis.

    Family carries the COLOUR (Newton / GD / AGD / SGD), the step rule inside a
    family carries the DASH pattern, so a reader can separate "which family" from
    "which step rule" without reading the legend twice. `x` selects the axis:
    "iter" (one point per outer iteration) or "time" (wall-clock seconds).
    """
    fam_colour = {"Newton": "#d62728", "GD": "#1f77b4",
                  "AGD": "#2ca02c", "SGD": "#9467bd"}
    dashes = ["-", "--", "-.", ":"]
    seen = {}
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for name, res in runs.items():
        fam = name.split()[0]
        i = seen.get(fam, 0); seen[fam] = i + 1
        g = np.maximum(np.asarray(res.f_history) - f_star, 1e-16)
        xs = range(len(g)) if x == "iter" else np.maximum(res.time_s, 1e-3)
        plot = ax.semilogy if x == "iter" else ax.loglog
        plot(xs, g, label=name, lw=1.7, color=fam_colour.get(fam, "0.4"),
             ls=dashes[i % len(dashes)])
    ax.set(xlabel="Số bước" if x == "iter" else "Thời gian (s)",
           ylabel=r"$f(w_k)-f^\star$",
           title=title or ("Bảy cấu hình đã dò tay: trục số bước" if x == "iter"
                           else "Bảy cấu hình đã dò tay: trục thời gian"))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    return _save(fig, fname, save)


def l_study_plot(ls, save: bool = True, fname: str = "tune_L.png"):
    """Xác định L: power iteration hội tụ | bốn ước lượng L đặt cạnh nhau."""
    fig, (ax_p, ax_b) = plt.subplots(1, 2, figsize=(11.5, 4.0))

    h = np.asarray(ls.power_iters)
    ax_p.semilogy(range(1, len(h) + 1), np.abs(h - h[-1]) + 1e-18, marker="o", ms=3)
    ax_p.set(xlabel="Vòng lặp power iteration", ylabel=r"$|\lambda^{(k)}-\lambda^{(\infty)}|$",
             title=f"Power iteration → $\\lambda_{{\\max}}$ = {ls.lambda_max_gram:.4g}")
    ax_p.grid(True, which="both", alpha=0.3)

    names = [r"Công thức" "\n" r"$\frac{1}{4}\lambda_{max}+\lambda$",
             r"Cục bộ tại $w^\star$",
             r"Thực nghiệm" "\n" r"$2/t_{max}$"]
    vals = [ls.L_formula, ls.L_local, ls.L_empirical]
    bars = ax_b.bar(names, vals, color=["#1f77b4", "#2ca02c", "#d62728"])
    for b, v in zip(bars, vals):
        ax_b.annotate(f"{v:.3g}", (b.get_x() + b.get_width() / 2, b.get_height()),
                      ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_b.set(ylabel="$L$", title="Ba ước lượng của cùng một hằng số $L$")
    ax_b.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, fname, save)


def sgd_schedule_panels(panels, f_star, log_every, save: bool = True,
                        fname: str = "tune_sgd_schedules.png"):
    """Một panel MỖI cỡ mini-batch, mỗi panel là các lịch bước — bố cục bài mẫu.

    `panels` là {batch: {nhãn: OptResult}}.
    """
    bs = sorted(panels)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0))
    for ax, b in zip(axes.ravel(), bs):
        for label, res in panels[b].items():
            g = np.maximum(np.asarray(res.f_history) - f_star, 1e-16)
            x = np.arange(len(g)) * log_every
            ax.semilogy(x, g, label=label, lw=1.3)
        ax.set(xlabel="Số bước", ylabel=r"$f(w_k)-f^\star$",
               title=f"SGD — batch = {b}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, fname, save)


def sgd_batch_compare(best, f_star, log_every, save: bool = True,
                      fname: str = "tune_sgd_batches.png"):
    """Lịch tốt nhất của từng cỡ batch, đặt cạnh nhau trên hai trục."""
    fig, (ax_it, ax_t) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for label, res in best.items():
        g = np.maximum(np.asarray(res.f_history) - f_star, 1e-16)
        ax_it.semilogy(np.arange(len(g)) * log_every, g, label=label, lw=1.6)
        ax_t.semilogy(res.time_s, g, label=label, lw=1.6)
    ax_it.set(xlabel="Số bước", ylabel=r"$f(w_k)-f^\star$",
              title="SGD — thử nghiệm độ lớn các mini-batch")
    ax_t.set(xlabel="Thời gian (s)", title="cùng dữ liệu, trục thời gian")
    for ax in (ax_it, ax_t):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, fname, save)
