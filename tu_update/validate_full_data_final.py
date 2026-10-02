"""Validate one completed full-data final stage without repeating the sweep."""
from __future__ import annotations

import argparse
import hashlib
import json

import run_full_data_stage as stage
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=stage.LABELS, required=True)
    args = parser.parse_args()
    directory = stage.OUT / args.method / "final"
    result = json.loads((directory / "results.json").read_text())
    locked = json.loads((stage.OUT / "selected_configurations.json").read_text())
    X, y, data = stage.load_input()
    fingerprint = {"data_sha256": data["sha256"], "source_sha256": stage.source_hash()}
    assert result["status"] == "complete" and result["stage"] == "final"
    assert result["fingerprint"] == locked["fingerprint"] == fingerprint
    assert result["protocol"]["stop_on_convergence"]
    assert len(result["candidates"]) == stage.TIMING_REPEATS == 3
    obj = stage.LogisticObjective(X, y, lam=stage.LAMBDA)
    reference = stage.reference_for(obj, fingerprint)
    fine = stage.previous_stage(args.method, "fine", fingerprint)
    selected_fine = fine["candidates"][fine["selected"]["candidate_index"]]
    times, checks = [], []
    first = result["candidates"][0]
    for index, row in enumerate(result["candidates"]):
        trace = row["trace"]
        n = row["iterations_run"]
        assert row["candidate_index"] == index and row["finite"]
        assert row["parameters"] == locked["methods"][args.method]["parameters"]
        assert trace["iteration"] == list(range(n + 1))
        assert 0 < n <= stage.CONVERGENCE_MAX_ITER
        for key in ("objective", "gradient_norm", "time_s"):
            assert len(trace[key]) == n + 1 and np.isfinite(trace[key]).all()
        assert np.all(np.diff(trace["time_s"]) >= 0)
        assert row["elapsed_s"] == trace["time_s"][-1]
        w = np.asarray(row["weights"])
        assert np.isfinite(w).all()
        exact_value = obj.value(w)
        exact_norm = float(np.linalg.norm(obj.grad(w)))
        assert abs(exact_value - row["final_objective"]) <= 1e-12
        assert abs(exact_norm - row["final_grad_norm"]) <= 1e-12
        assert abs(exact_value - trace["objective"][-1]) <= 1e-12
        assert abs(max(exact_value - reference["f_star"], 0) - row["final_f_gap"]) <= 1e-12
        assert exact_norm <= stage.CONVERGENCE_TOL
        assert row["first_converged_at"] == n
        if args.method == "gd_fixed":
            assert row["gradient_evals"] == row["objective_evals"] == n + 1
            assert len(trace["accepted_steps"]) == 0
            assert np.all(np.asarray(trace["gradient_norm"][:-1]) > stage.CONVERGENCE_TOL)
        elif args.method == "gd_backtracking":
            assert row["gradient_evals"] == n + 1
            assert row["objective_evals"] == 1 + row["backtracking_trials"]
            assert len(trace["accepted_steps"]) == n
        else:
            assert row["objective_evals"] == n + 1 and row["gradient_evals"] >= n + 2
            assert len(trace["accepted_steps"]) == n
        prefix = min(n + 1, len(selected_fine["trace"]["iteration"]))
        for key in ("objective", "gradient_norm"):
            np.testing.assert_allclose(trace[key][:prefix], selected_fine["trace"][key][:prefix], rtol=0, atol=1e-12)
            np.testing.assert_allclose(trace[key], first["trace"][key], rtol=0, atol=1e-12)
        np.testing.assert_allclose(w, first["weights"], rtol=0, atol=1e-12)
        times.append(row["elapsed_s"])
        checks.append({"candidate_index": index, "parameters": row["parameters"],
                       "iterations": n, "gradient_norm_recomputed": exact_norm,
                       "raw_objective_gap": exact_value - reference["f_star"],
                       "distance_to_newton_weights": float(np.linalg.norm(w - reference["w_star"])),
                       "objective_increases": int(np.sum(np.diff(trace["objective"]) > 1e-12)),
                       "fine_prefix_updates_reproduced": prefix - 1})
    assert result["median_time_s"] == float(np.median(times))
    representative = int(np.argsort(times)[len(times) // 2])
    assert result["representative_candidate_index"] == representative
    weights = np.asarray(result["candidates"][representative]["weights"])
    for saved, recomputed in (
        (result["model_metrics_in_sample"], stage.classification_metrics(X, y, weights)),
        (result["model_metrics_w0_in_sample"], stage.classification_metrics(X, y, np.zeros(obj.d))),
    ):
        for key in saved:
            assert abs(saved[key] - recomputed[key]) <= 1e-12, key
    observer = json.loads((directory / "observer.json").read_text())
    assert observer["source_sha256"] == hashlib.sha256((stage.REPO / observer["script"]).read_bytes()).hexdigest()
    assert [row["candidate_index"] for row in observer["repeats"]] == list(range(3))
    validation = {"status": "passed", "method": args.method, "stage": "final",
                  "all_repeats_certified_converged": True, "timing_repeats_verified": 3,
                  "data_source_and_locked_parameters_verified": True,
                  "fine_prefix_and_all_repeat_numeric_traces_reproduced": True,
                  "exact_endpoint_objective_and_gradient_verified": True,
                  "metrics_recomputed": True, "model_evaluation_scope": "in-sample; original test is empty",
                  "median_time_s": result["median_time_s"],
                  "checkpoint_overhead_s": [row["checkpoint_overhead_s"] for row in observer["repeats"]],
                  "checks": checks}
    stage.write_json(directory / "validation.json", validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
