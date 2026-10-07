QEC TRAJECTORY-SPLIT DATASET — 2026-10-07

Fresh Qiskit Aer simulation: 15 trajectories x 400 cycles x 4,000 shots.
The full simulation completed in 78 seconds. This is not the original
5,765-cycle dataset from the research paper.

Training:   9 trajectories, 3,528 windows, 878 class 0 / 2,650 class 1
Validation: 3 trajectories, 1,176 windows, 327 class 0 / 849 class 1
Test:       3 trajectories, 1,176 windows, 530 class 0 / 646 class 1

FILES
- dataset.jsonl: complete per-cycle records with counts, noise and provenance.
- cycles.csv: flattened raw cycles; raw_row is a zero-based JSONL row index.
- train.npz, val.npz, test.npz: ready-to-use NumPy arrays.
- windows.csv: split and next-cycle target mapping for every window.
- manifest.json: parameters, seeds, class counts, versions and SHA-256 hashes.
- provenance.json: source commit and verification results.

LOADING
import numpy as np
train = np.load('train.npz', allow_pickle=False)
X_train, y_train = train['X'], train['y']

X shape is (number_of_windows, 8, 4, 1), with columns 00, 01, 10, 11.
Each NPZ also contains trajectory_ids, target_cycle, and source_rows.
source_rows includes all eight input rows plus the next-cycle target row.
The label is 1 when the next cycle's measured nonzero-syndrome rate > 0.08.

VERIFICATION
Four unit tests passed. Full-data checks confirmed counts total 4,000 shots,
correct features and labels, consecutive cycles, and zero cross-split overlap
of either trajectory IDs or raw rows (including target cycles).

INTERPRETATION
These labels measure a syndrome-rate threshold, not decoded logical failures.
Each circuit prepares a fresh state; temporal dependence comes from noise drift.
Only depolarizing and readout noise are used. Grover rows are excluded.
No CNN has been trained on this dataset. The previous 98.3% accuracy does not
apply. The training-majority baseline on this test set is 54.93%.
Use validation for tuning and evaluate the test set only after training.
There are only three independent test trajectories; overlapping windows are
not independent samples for uncertainty calculations.

REPRODUCE
python -m pip install -r requirements-qec.txt
python generate_qec_dataset.py --output qec_dataset

Source: https://github.com/ashton-reddy/quantum_comupter_simulation
Commit: 4e3bab2c97ab85ae19da1d6b08966d79af4dd3f4
