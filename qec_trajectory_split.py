"""Leakage-safe splitting for the QEC next-cycle prediction experiment.

Dependencies: numpy, scikit-learn (already used by the original project).
Input: raw QEC cycles only; exclude Grover rows. All four arrays must align.
Use the ORIGINAL trajectory/run IDs, not IDs inferred from equal-size chunks.

Integration (replace the old global windowing and random window split):

    from qec_trajectory_split import make_splits, report_results

    splits = make_splits(
        probabilities,    # (N, 4): probabilities for 00, 01, 10, 11
        error_rates,     # (N,): measured error rate of each cycle
        trajectory_ids,  # (N,): independent simulation run ID
        cycle_indices,   # (N,): consecutive cycle number within each run
    )
    train, val, test = (splits[name] for name in ("train", "val", "test"))

    # Recreate and compile your existing CNN with FRESH weights and optimizer.
    # Set NumPy/TensorFlow seeds before creating it. Do not load old weights.
    model.fit(
        train["X"], train["y"], epochs=10, batch_size=32,
        validation_data=(val["X"], val["y"]),
    )
    # This matches the paper's two-output CNN and integer labels.
    predicted = model.predict(test["X"], verbose=0).argmax(axis=1)
    results = report_results(splits, predicted)

Default allocation with 15 trajectories: 9 train, 3 validation, 3 test.
Keep seed=42 and this allocation fixed; do not choose a split by test accuracy.
Any learned scaling, class weights, or resampling must use training data only.
Overlapping windows INSIDE one split are allowed. Test metrics remain correlated
within trajectories; windows are not independent replicates for uncertainty.
"""

import warnings

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit


def make_splits(probabilities, error_rates, trajectory_ids, cycle_indices,
                *, window=8, threshold=0.08, seed=42):
    """Split raw trajectories, then build windows separately inside each run.

    Return train/val/test dictionaries with X, y, trajectory_ids, target_cycle,
    and source_rows. source_rows contains original input row indices for each
    window's eight input cycles PLUS its next-cycle target, for auditing.
    Trajectories must have unique, consecutive integer cycle numbers; missing
    cycles or duplicate rows raise an error instead of being silently bridged.
    """
    p = np.asarray(probabilities, dtype=np.float32)
    rates = np.asarray(error_rates, dtype=float)
    groups = np.asarray(trajectory_ids)
    cycles = np.asarray(cycle_indices, dtype=float)
    if rates.ndim != 1 or groups.ndim != 1 or cycles.ndim != 1:
        raise ValueError("error_rates, trajectory_ids, and cycle_indices must be 1D.")
    n = len(rates)
    if p.shape != (n, 4) or len(groups) != n or len(cycles) != n:
        raise ValueError("Expected aligned (N, 4) probabilities and three (N,) arrays.")
    if not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("window must be a positive integer.")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Syndrome probabilities must be finite and between 0 and 1.")
    if not np.allclose(p.sum(axis=1), 1, atol=1e-4):
        raise ValueError("Each syndrome probability vector must sum to 1.")
    if not np.isfinite(rates).all() or (rates < 0).any() or (rates > 1).any():
        raise ValueError("Error rates must be finite and between 0 and 1.")
    if not np.isfinite(cycles).all() or not np.equal(cycles, np.floor(cycles)).all():
        raise ValueError("Cycle indices must be finite integers.")
    if any(g is None or str(g).strip().lower() in {"", "nan", "none", "<na>"}
           for g in groups):
        raise ValueError("Every row needs its original trajectory ID.")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 5:
        raise ValueError("This 60/20/20 allocation requires at least 5 trajectories.")

    # Validate each full run BEFORE splitting; never join different runs.
    ordered_rows = {}
    for group in unique_groups:
        rows = np.flatnonzero(groups == group)
        rows = rows[np.argsort(cycles[rows], kind="stable")]
        if len(rows) <= window:
            raise ValueError(f"Trajectory {group!r} needs at least {window + 1} cycles.")
        if not np.all(np.diff(cycles[rows]) == 1):
            raise ValueError(f"Trajectory {group!r} has duplicate or missing cycle numbers.")
        ordered_rows[group] = rows

    # GroupShuffleSplit fractions refer to trajectories, not individual rows.
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    remaining, test_rows = next(outer.split(p, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_local, val_local = next(inner.split(p[remaining], groups=groups[remaining]))
    row_splits = {
        "train": remaining[train_local],
        "val": remaining[val_local],
        "test": test_rows,
    }
    output = {}
    for name, rows in row_splits.items():
        sources = []
        for group in np.unique(groups[rows]):
            ordered = ordered_rows[group]
            # Eight input cycles, followed by ONE unseen target cycle.
            sources.extend(ordered[t - window:t + 1]
                           for t in range(window, len(ordered)))
        sources = np.stack(sources)
        targets = sources[:, -1]
        y = (rates[targets] > threshold).astype(np.int64)
        output[name] = {
            "X": p[sources[:, :-1]][..., np.newaxis],
            "y": y,
            "trajectory_ids": groups[targets],
            "target_cycle": cycles[targets].astype(np.int64),
            "source_rows": sources,
        }
        print(f"{name}: IDs={np.unique(groups[targets]).tolist()}, "
              f"windows={len(y)}, failure fraction={y.mean():.3f}")
        if len(np.unique(y)) < 2:
            warnings.warn(
                f"{name} contains only one class. Report this limitation; "
                "do not hunt for a seed that improves test results.",
                stacklevel=2,
            )

    # Include BOTH feature and target cycles in the overlap checks.
    names = list(output)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if np.intersect1d(output[left]["source_rows"],
                              output[right]["source_rows"]).size:
                raise RuntimeError(f"Raw-cycle overlap between {left} and {right}.")
            if np.intersect1d(output[left]["trajectory_ids"],
                              output[right]["trajectory_ids"]).size:
                raise RuntimeError(f"Trajectory overlap between {left} and {right}.")
    return output


def report_results(splits, predicted_classes):
    """Report held-out metrics with a majority baseline selected on TRAIN only."""
    y_train, y_test = splits["train"]["y"], splits["test"]["y"]
    pred = np.asarray(predicted_classes)
    if pred.shape != y_test.shape or not np.isin(pred, [0, 1]).all():
        raise ValueError("Pass one predicted class (0 or 1) per test window.")
    majority = int(np.bincount(y_train, minlength=2).argmax())
    report = classification_report(
        y_test, pred, labels=[0, 1], target_names=["No failure", "Failure"],
        output_dict=True, zero_division=0,
    )
    result = {
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "training_majority_class": majority,
        "majority_baseline_test_accuracy": float(np.mean(y_test == majority)),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        "classification_report": report,
    }
    if len(np.unique(y_test)) == 2:
        result["balanced_accuracy"] = float(
            (report["No failure"]["recall"] + report["Failure"]["recall"]) / 2
        )
    print(f"Test accuracy: {result['test_accuracy']:.4f}")
    print(f"Training-majority baseline: {result['majority_baseline_test_accuracy']:.4f}")
    print("Confusion matrix (rows=actual, columns=predicted; order=0,1):")
    print(np.array(result["confusion_matrix"]))
    print(classification_report(
        y_test, pred, labels=[0, 1], target_names=["No failure", "Failure"],
        digits=4, zero_division=0,
    ))
    return result


if __name__ == "__main__":
    print(__doc__)
