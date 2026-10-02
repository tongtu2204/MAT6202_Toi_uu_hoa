"""Run one final stage with observational checkpoints; optimizer math is unchanged.

Usage: python tu_update/run_full_data_with_progress.py --method gd_fixed --stage final
Completed timing repeats resume through run_full_data_stage.py. progress.json
also preserves the current repeat's weights, counters and full numerical trace.
The measured runtime includes checkpoint overhead, reported separately below.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import run_full_data_stage as stage  # Sets single-thread BLAS before numpy.
import gd_agd_runner as runners

INTERVAL = 2000


def execute():
    method = sys.argv[sys.argv.index("--method") + 1]
    if "--stage" not in sys.argv or sys.argv[sys.argv.index("--stage") + 1] != "final":
        raise ValueError("Progress wrapper is for the final stage only")
    directory = stage.OUT / method / "final"
    path = directory / "results.json"
    completed = len(json.loads(path.read_text())["candidates"]) if path.exists() else 0
    observer = {"script": str(Path(__file__).relative_to(stage.REPO)),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "interval_updates": INTERVAL, "timing_includes_checkpoint_overhead": True,
                "repeats": []}
    previous_meta = directory / "observer.json"
    if previous_meta.exists():
        old = json.loads(previous_meta.read_text())
        if old["source_sha256"] != observer["source_sha256"]:
            raise ValueError("Observer differs from the recorded checkpoint")
        observer["repeats"] = old["repeats"]
    active = {"candidate_index": None, "object": None, "checkpoint_s": 0.0}
    original_objective = stage.LogisticObjective
    original_record = runners._record
    original_candidate = stage.run_candidate

    class ObservedObjective(original_objective):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            active["object"] = self

        def value(self, w):
            value = super().value(w)
            self.last_value_weights = w.copy()
            return value

        def grad(self, w):
            gradient = super().grad(w)
            self.last_gradient_weights = w.copy()
            self.last_gradient = gradient.copy()
            return gradient

    def record(run, iteration, value, grad_norm, started):
        original_record(run, iteration, value, grad_norm, started)
        if active["candidate_index"] is None or iteration % INTERVAL:
            return
        obj = active["object"]
        snapshot = {
            "status": "in_progress", "method": method, "stage": "final",
            "candidate_index": active["candidate_index"], "parameters": run.parameters,
            "iteration": iteration, "objective": value, "gradient_norm": grad_norm,
            "gradient_location": "momentum point" if method.startswith("agd") else "current weights",
            "elapsed_s": run.elapsed_s, "checkpoint_overhead_s_so_far": active["checkpoint_s"],
            "weights": obj.last_value_weights.tolist(),
            "gradient_weights": obj.last_gradient_weights.tolist(),
            "gradient": obj.last_gradient.tolist(),
            "objective_evals": run.objective_evals, "gradient_evals": run.gradient_evals,
            "backtracking_trials": run.backtracking_trials,
            "trace": {"iteration": run.iterations, "objective": run.objectives,
                      "gradient_norm": run.grad_norms, "time_s": run.times_s,
                      "accepted_steps": run.steps},
            "observer_source_sha256": observer["source_sha256"],
        }
        checkpoint_started = time.perf_counter()
        stage.write_json(directory / "progress.json", snapshot)
        active["checkpoint_s"] += time.perf_counter() - checkpoint_started
        if iteration:
            gap = value - json.loads((stage.OUT / "reference.json").read_text())["f_star"]
            print(f"  repeat {active['candidate_index'] + 1}: k={iteration}; "
                  f"gap={gap:.3e}; gradient={grad_norm:.3e}; time={run.elapsed_s:.1f}s", flush=True)

    def candidate(method_arg, obj, parameters, stage_arg):
        nonlocal completed
        if stage_arg != "final":
            return original_candidate(method_arg, obj, parameters, stage_arg)
        active["candidate_index"] = completed
        active["checkpoint_s"] = 0.0
        try:
            run = original_candidate(method_arg, obj, parameters, stage_arg)
            observer["repeats"].append({"candidate_index": completed,
                                        "checkpoint_overhead_s": active["checkpoint_s"],
                                        "elapsed_s": run.elapsed_s})
            stage.write_json(directory / "observer.json", observer)
            completed += 1
            return run
        finally:
            active["candidate_index"] = None

    stage.LogisticObjective = ObservedObjective
    runners._record = record
    stage.run_candidate = candidate
    try:
        stage.main()
        snapshot = json.loads((directory / "progress.json").read_text())
        snapshot["status"] = "stage_complete; use results.json for final endpoints"
        stage.write_json(directory / "progress.json", snapshot)
    finally:
        stage.LogisticObjective = original_objective
        runners._record = original_record
        stage.run_candidate = original_candidate


if __name__ == "__main__":
    execute()
