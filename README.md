# Quantum Computer Simulation
Simulated quantum circuits for research experimentation.

Repository: https://github.com/ashton-reddy/quantum_computer_simulation

## Saved trajectory-held-out evaluation

The reference dataset and completed CNN evaluation are committed here as ordinary
files, including the binary NumPy arrays and Keras model. They are the exact
artifacts used for the revised paper's 89.63% test-accuracy result.

| Artifact | Location |
| --- | --- |
| Training and evaluation script | `evaluate_trajectory_cnn.py` |
| Raw dataset | `qec_dataset/dataset.jsonl`, `qec_dataset/cycles.csv` |
| Train / validation / test arrays | `qec_dataset/train.npz`, `qec_dataset/val.npz`, `qec_dataset/test.npz` |
| Window-to-cycle mapping | `qec_dataset/windows.csv` |
| Dataset settings, seeds and provenance | `qec_dataset/manifest.json`, `qec_dataset/provenance.json` |
| Saved final model | `cnn_results/cnn_final.keras` |
| Per-window predictions | `cnn_results/test_predictions.csv` |
| Metrics, including baselines and per-trajectory results | `cnn_results/metrics.json` |
| Training protocol and architecture | `cnn_results/protocol.json`, `cnn_results/architecture.json` |
| Epoch history and execution log | `cnn_results/history.json`, `cnn_results/training.log` |
| Recorded package versions | `requirements-reproduce.txt` |
| Artifact integrity hashes | `artifact_sha256.json` |

The CNN achieved **89.63% accuracy** and **88.71% balanced accuracy** on 1,176
windows from three held-out trajectories. The training-majority baseline achieved
54.93% accuracy and last-cycle persistence achieved 91.24%. The CNN confusion
matrix, with actual rows and predicted columns ordered as classes 0 and 1, is
`[[421, 109], [13, 633]]`. This run does not establish an accuracy advantage over
the persistence baseline.

This is a fresh dataset, not a re-split of the original paper's unavailable data.
The original training implementation was unavailable, so the CNN was reconstructed
from the manuscript, with explicit `same` padding in both convolutions. Training
used fresh weights, seed 42, Adam at learning rate 0.001, batch size 32, and exactly
10 epochs. The final epoch was evaluated without checkpoint selection or seed
search. Training, validation, and test sets contain 9, 3, and 3 whole trajectories;
no raw input or target row is shared across sets.

Reproduce training from the committed split arrays with Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproduce.txt
python evaluate_trajectory_cnn.py --data qec_dataset --output reproduced_results
```

The output directory must not already exist. The committed reference files remain
in `cnn_results/`; new runs go into the ignored `reproduced_results/` directory.
`qec_dataset/README.txt` and `manifest.json` describe the dataset-generation step,
so their `model_trained=false` statement is historical. The completed evaluation
is recorded separately in `cnn_results/`.

The dataset manifest records file/source SHA-256 hashes; the evaluation metrics
record the training-script and dataset-manifest hashes. `artifact_sha256.json`
also lists SHA-256 hashes for every committed reference artifact.

Only three independent test trajectories are available. Overlapping windows
within each split are correlated. Labels indicate a syndrome-rate threshold,
not decoded logical failure; each cycle prepares a fresh quantum state.

## QEC dataset with trajectory-disjoint splits

The supplied `qec_trajectory_split.py` is included unchanged. It splits original
trajectory IDs before constructing eight-cycle windows, assigns 9/3/3 of 15
trajectories to train/validation/test with seed 42, and checks that neither raw
feature/target cycles nor trajectory IDs overlap between splits.

Generate a fresh dataset from the existing five-qubit bit-flip syndrome circuit:

```bash
python -m pip install -r requirements-qec.txt
python generate_qec_dataset.py --output qec_dataset_regenerated
python -m unittest discover -s tests
```

The default run generates 15 independent noise trajectories of 400 cycles, with
4,000 shots per cycle. Each trajectory resets the linear drift and random walk.
Noise settings come from the old sweep: p1=0.001, p2=0.01, pm=0.02; slopes
2e-5 and 1e-5 and random-walk standard deviations 2e-4 and 1e-4 for p2 and pm.
Only the depolarizing and readout channels implemented in `build_noise_model`
are applied. Density-matrix simulation uses one thread and logs progress every
100 cycles. Raw output is flushed at each progress update. Existing output
directories containing files are never overwritten.

This new entry point avoids the old sweep's unsupported `NoiseParams` fields.
It does not execute the Grover benchmark or alter the original `main.py`.

Outputs:

- `dataset.jsonl`: raw counts, noise settings, metrics, trajectory IDs and seeds.
- `cycles.csv`: flattened raw cycles, in the same row order.
- `train.npz`, `val.npz`, `test.npz`: X, y, trajectory_ids, target_cycle, source_rows.
- `windows.csv`: window-to-trajectory/split/target mapping.
- `manifest.json`: settings, class counts, baseline, package versions and hashes.

```python
import numpy as np
train = np.load("qec_dataset/train.npz", allow_pickle=False)
val = np.load("qec_dataset/val.npz", allow_pickle=False)
test = np.load("qec_dataset/test.npz", allow_pickle=False)
# train["X"] has shape (3528, 8, 4, 1); labels are integer 0 or 1.
# Probability columns are 00, 01, 10, 11.
```

`source_rows` uses zero-based raw-row indices and includes the eight input cycles
plus the next-cycle target. The label is 1 when that target's observed nonzero
syndrome rate is strictly greater than 0.08. It is a syndrome-rate proxy, not a
measurement of decoded logical failure. Each circuit invocation prepares a
fresh state; temporal dependence comes from drifting noise.

This is a new 6,000-cycle dataset, not the paper's unavailable 5,765-cycle data.
It has 5,880 windows: 3,528 train, 1,176 validation, and 1,176 test. The generator
does not train a CNN; use `evaluate_trajectory_cnn.py` for that step. The completed
reference evaluation is described above. The old 98.3% does not apply.
Overlapping windows within each split are correlated, and only three independent
trajectories are held out for testing.

