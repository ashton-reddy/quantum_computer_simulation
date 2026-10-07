# Quantum_Comupter_Simulation
Simulated Quatum Computer for Research Experimentation

## QEC dataset with trajectory-disjoint splits

The supplied `qec_trajectory_split.py` is included unchanged. It splits original
trajectory IDs before constructing eight-cycle windows, assigns 9/3/3 of 15
trajectories to train/validation/test with seed 42, and checks that neither raw
feature/target cycles nor trajectory IDs overlap between splits.

Generate a fresh dataset from the existing five-qubit bit-flip syndrome circuit:

```bash
python -m pip install -r requirements-qec.txt
python generate_qec_dataset.py --output qec_dataset
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
It has 5,880 windows: 3,528 train, 1,176 validation, and 1,176 test. No CNN is
trained by this generator. Retrain from fresh weights using only validation for
tuning before reporting new held-out accuracy; the old 98.3% does not apply.
Overlapping windows within each split are correlated, and only three independent
trajectories are held out for testing.

