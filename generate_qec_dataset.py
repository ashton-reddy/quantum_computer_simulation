"""Generate fresh QEC trajectories and export trajectory-disjoint ML splits.

Run from a checkout with: python generate_qec_dataset.py --output qec_dataset
This uses the existing five-qubit circuit and depolarizing/readout noise model.
It does not reproduce the paper's unavailable original data or train a CNN.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from qexp.core.noise import NoiseParams, build_noise_model, noise_to_log_dict
from qexp.core.utils import circuit_features
from qexp.qec.bitflip_syndrome import build_bitflip_syndrome_round, syndrome_metrics
from qec_trajectory_split import make_splits

OUTCOMES = ("00", "01", "10", "11")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("qec_dataset"))
    parser.add_argument("--trajectories", type=int, default=15)
    parser.add_argument("--cycles", type=int, default=400)
    parser.add_argument("--shots", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.08)
    args = parser.parse_args(argv)
    if (args.trajectories < 5 or args.window < 1 or args.cycles <= args.window
            or args.shots < 1 or args.seed < 0 or args.split_seed < 0
            or not 0 <= args.threshold <= 1):
        parser.error("Need >=5 trajectories, cycles > window >=1, shots >=1, "
                     "nonnegative seeds, and threshold in [0,1].")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be new or empty; existing runs are preserved.")
    args.output.mkdir(parents=True, exist_ok=True)

    # Single-thread density-matrix simulation is efficient for this five-qubit
    # noisy circuit. Aer still samples the requested number of measurement shots.
    simulator = AerSimulator(method="density_matrix", max_parallel_threads=1)
    circuit = transpile(build_bitflip_syndrome_round(), simulator,
                        optimization_level=1, seed_transpiler=args.seed)
    features = circuit_features(circuit)
    start = time.perf_counter()
    probabilities, rates, groups, cycles = [], [], [], []
    trajectories = []
    flat_fields = ["raw_row", "trajectory_id", "cycle", "shots", "simulator_seed",
                   "p1", "p2", "pm", "count_00", "count_01", "count_10", "count_11",
                   "prob_00", "prob_01", "prob_10", "prob_11", "error_rate_this_cycle"]

    with (args.output / "dataset.jsonl").open("w") as raw_file, \
            (args.output / "cycles.csv").open("w", newline="") as flat_file:
        writer = csv.DictWriter(flat_file, fieldnames=flat_fields)
        writer.writeheader()
        for trajectory_id in range(args.trajectories):
            # Every trajectory starts a fresh RNG and random walk. IDs are
            # assigned during simulation, never inferred from concatenated rows.
            trajectory_seed = args.seed + trajectory_id * (args.cycles + 1)
            rng = np.random.default_rng(trajectory_seed)
            rw_p2 = rw_pm = 0.0
            previous_error = None
            trajectories.append({"trajectory_id": trajectory_id,
                                 "noise_seed": trajectory_seed,
                                 "cycles": args.cycles})
            for cycle in range(args.cycles):
                rw_p2 += float(rng.normal(0, 2e-4))
                rw_pm += float(rng.normal(0, 1e-4))
                noise = NoiseParams(
                    p1=0.001,
                    p2=float(np.clip(0.01 + 2e-5 * cycle + rw_p2, 0, 1)),
                    pm=float(np.clip(0.02 + 1e-5 * cycle + rw_pm, 0, 1)),
                )
                simulator_seed = trajectory_seed + cycle + 1
                result = simulator.run(circuit, shots=args.shots,
                                       seed_simulator=simulator_seed,
                                       noise_model=build_noise_model(noise)).result()
                if not result.success:
                    raise RuntimeError(str(result.status))
                observed = result.get_counts()
                counts = {key: int(observed.get(key, 0)) for key in OUTCOMES}
                if sum(counts.values()) != args.shots:
                    raise RuntimeError("Measurement counts do not sum to requested shots.")
                metrics = syndrome_metrics(counts)
                error = metrics["error_rate_this_cycle"]
                row_index = len(rates)
                row = {"raw_row": row_index, "domain": "quantum_error_correction",
                       "experiment": "bitflip_code_syndrome",
                       "trajectory_id": trajectory_id, "cycle": cycle,
                       "trajectory_seed": trajectory_seed,
                       "simulator_seed": simulator_seed, "shots": args.shots,
                       "noise": noise_to_log_dict(noise), "circuit": features,
                       "counts": counts, "metrics": metrics,
                       "derived": {"temporal_error_delta": None if previous_error is None
                                   else error - previous_error}}
                raw_file.write(json.dumps(row, sort_keys=True) + "\n")
                flat = {"raw_row": row_index, "trajectory_id": trajectory_id,
                        "cycle": cycle, "shots": args.shots,
                        "simulator_seed": simulator_seed,
                        "p1": noise.p1, "p2": noise.p2, "pm": noise.pm,
                        "error_rate_this_cycle": error}
                flat.update({f"count_{key}": counts[key] for key in OUTCOMES})
                flat.update({f"prob_{key}": counts[key] / args.shots for key in OUTCOMES})
                writer.writerow(flat)
                probabilities.append([metrics["syndrome_probs"][key] for key in OUTCOMES])
                rates.append(error)
                groups.append(trajectory_id)
                cycles.append(cycle)
                previous_error = error
                if (cycle + 1) % 100 == 0 or cycle + 1 == args.cycles:
                    raw_file.flush()
                    flat_file.flush()
                    elapsed = time.perf_counter() - start
                    print(f"Trajectory {trajectory_id + 1}/{args.trajectories}, "
                          f"cycle {cycle + 1}/{args.cycles}; "
                          f"{len(rates)} total cycles; {elapsed:.1f}s", flush=True)

    splits = make_splits(probabilities, rates, groups, cycles, window=args.window,
                         threshold=args.threshold, seed=args.split_seed)
    split_summary = {}
    with (args.output / "windows.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["split", "split_row", "trajectory_id", "input_start_cycle",
                         "input_end_cycle", "target_cycle", "target_raw_row", "y"])
        for name, part in splits.items():
            np.savez_compressed(args.output / f"{name}.npz", **part)
            split_summary[name] = {
                "trajectory_ids": np.unique(part["trajectory_ids"]).tolist(),
                "windows": int(len(part["y"])),
                "class_counts": np.bincount(part["y"], minlength=2).tolist(),
                "positive_fraction": float(part["y"].mean()),
                "X_shape": list(part["X"].shape),
            }
            for index, (group, target, source, label) in enumerate(zip(
                    part["trajectory_ids"], part["target_cycle"],
                    part["source_rows"], part["y"])):
                writer.writerow([name, index, group, target - args.window,
                                 target - 1, target, source[-1], label])

    majority = int(np.bincount(splits["train"]["y"], minlength=2).argmax())
    summary = {
        "dataset_kind": "fresh_simulation_not_original_paper_data",
        "config": {key: value for key, value in vars(args).items() if key != "output"},
        "noise": {"p1": 0.001, "base_p2": 0.01, "base_pm": 0.02,
                  "slope_p2": 2e-5, "slope_pm": 1e-5,
                  "rw_sigma_p2": 2e-4, "rw_sigma_pm": 1e-4,
                  "reset_drift_each_trajectory": True},
        "simulation_method": "density_matrix",
        "raw_cycles": len(rates), "windowed_samples": sum(len(s["y"]) for s in splits.values()),
        "probability_order": list(OUTCOMES),
        "label": "1 if next cycle's observed nonzero-syndrome rate is > threshold, else 0",
        "trajectories": trajectories, "splits": split_summary,
        "training_majority_class": majority,
        "majority_baseline_test_accuracy": float(np.mean(splits["test"]["y"] == majority)),
        "model_trained": False,
        "limitations": [
            "Fresh 15 x 400-cycle run; original 5,765-cycle dataset was not available.",
            "Each cycle prepares a fresh state; temporal dependence comes from noise drift.",
            "Thresholded syndrome rate is a proxy, not a decoded logical-failure measurement.",
            "Only depolarizing gate noise and symmetric readout noise are simulated.",
            "Grover benchmark rows are excluded from this QEC dataset.",
            "Only three independent test trajectories; windows within a split overlap.",
            "No CNN evaluation is included; the paper's 98.3% does not apply to these splits.",
        ],
        "versions": {name: importlib.metadata.version(name)
                     for name in ("qiskit", "qiskit-aer", "numpy", "scikit-learn")},
        "python_version": platform.python_version(),
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__), ROOT / "qec_trajectory_split.py",
                                       ROOT / "src/qexp/core/noise.py",
                                       ROOT / "src/qexp/qec/bitflip_syndrome.py")},
        "elapsed_seconds": time.perf_counter() - start,
    }
    summary["file_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sorted(args.output.iterdir()) if path.is_file()}
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Done: {len(rates)} cycles; {summary['windowed_samples']} windows. "
          f"Saved to {args.output.resolve()}", flush=True)
    return summary


if __name__ == "__main__":
    main()
