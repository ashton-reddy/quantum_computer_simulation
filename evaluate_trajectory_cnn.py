"""Fixed-protocol re-evaluation of the paper's CNN on trajectory-held-out data.

Usage: python evaluate_trajectory_cnn.py --data qec_dataset --output cnn_results
The architecture is reconstructed from the paper, with explicit same padding.
No hyperparameter or seed search is performed; the final epoch is evaluated.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '2'

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score)


def score(y, pred, prob=None):
    result = {
        'accuracy': float(accuracy_score(y, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        'classification_report': classification_report(
            y, pred, labels=[0, 1], target_names=['No failure', 'Failure'],
            output_dict=True, zero_division=0),
    }
    if prob is not None and len(np.unique(y)) == 2:
        result['roc_auc'] = float(roc_auc_score(y, prob))
        result['average_precision'] = float(average_precision_score(y, prob))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        'seed': 42, 'epochs': 10, 'batch_size': 32,
        'optimizer': 'Adam', 'learning_rate': 0.001,
        'loss': 'SparseCategoricalCrossentropy(from_logits=True)',
        'padding': 'same on both convolutions; valid on both pools',
        'checkpoint': 'final epoch 10, no early stopping or model selection',
        'class_weighting': False, 'normalization': 'none; input probabilities unchanged',
        'deterministic_ops': True, 'intra_op_threads': 2, 'inter_op_threads': 1,
        'model_origin': 'reconstructed from manuscript architecture; original training code unavailable',
        'test_use': 'evaluation only after fixed training protocol',
    }
    (args.output / 'protocol.json').write_text(json.dumps(config, indent=2) + '\n')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(config['seed'])
    tf.config.experimental.enable_op_determinism()
    train = dict(np.load(args.data / 'train.npz', allow_pickle=False))
    val = dict(np.load(args.data / 'val.npz', allow_pickle=False))
    manifest = json.loads((args.data / 'manifest.json').read_text())
    for filename, digest in manifest['file_sha256'].items():
        assert hashlib.sha256((args.data / filename).read_bytes()).hexdigest() == digest

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8, 4, 1)),
        tf.keras.layers.Conv2D(32, (3, 2), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 1)),
        tf.keras.layers.Conv2D(64, (3, 2), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(2),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    (args.output / 'architecture.json').write_text(model.to_json() + '\n')
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.experimental_deterministic = True
    train_ds = tf.data.Dataset.from_tensor_slices((train['X'], train['y']))
    train_ds = train_ds.shuffle(len(train['y']), seed=42, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(32).with_options(options)
    val_ds = tf.data.Dataset.from_tensor_slices((val['X'], val['y'])).batch(32).with_options(options)
    start = time.perf_counter()
    history = model.fit(train_ds, validation_data=val_ds, epochs=10, verbose=2)
    elapsed = time.perf_counter() - start
    model.save(args.output / 'cnn_final.keras')
    (args.output / 'history.json').write_text(json.dumps(history.history, indent=2) + '\n')

    # Test data is first loaded for model evaluation after the final model is saved.
    test = dict(np.load(args.data / 'test.npz', allow_pickle=False))
    parts = {'train': train, 'val': val, 'test': test}
    for left, right in itertools.combinations(parts, 2):
        assert not np.intersect1d(parts[left]['trajectory_ids'], parts[right]['trajectory_ids']).size
        assert not np.intersect1d(parts[left]['source_rows'], parts[right]['source_rows']).size
    test_ds = tf.data.Dataset.from_tensor_slices(test['X']).batch(32).with_options(options)
    logits = model.predict(test_ds, verbose=0)
    probabilities = tf.nn.softmax(logits, axis=-1).numpy()[:, 1]
    predicted = logits.argmax(axis=1)

    raw = [json.loads(line) for line in (args.data / 'dataset.jsonl').read_text().splitlines()]
    rates = np.asarray([r['metrics']['error_rate_this_cycle'] for r in raw])
    for part in parts.values():
        source = part['source_rows']
        np.testing.assert_array_equal(part['y'], rates[source[:, -1]] > manifest['config']['threshold'])
    persistence = (rates[test['source_rows'][:, -2]] > manifest['config']['threshold']).astype(int)
    majority = int(np.bincount(train['y'], minlength=2).argmax())
    y = test['y']
    result = {
        'protocol': config,
        'source_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'dataset_manifest_sha256': hashlib.sha256((args.data / 'manifest.json').read_bytes()).hexdigest(),
        'versions': {name: importlib.metadata.version(name) for name in
                     ['tensorflow-cpu', 'keras', 'numpy', 'scikit-learn']},
        'python': platform.python_version(), 'parameters': int(model.count_params()),
        'training_seconds': elapsed, 'test_windows': int(len(y)),
        'cnn': score(y, predicted, probabilities),
        'training_majority_class': majority,
        'majority_baseline': score(y, np.full_like(y, majority)),
        'persistence_baseline': score(y, persistence),
        'per_test_trajectory': [],
        'split_audit_passed': True,
    }
    for trajectory in np.unique(test['trajectory_ids']):
        mask = test['trajectory_ids'] == trajectory
        result['per_test_trajectory'].append({
            'trajectory_id': int(trajectory), 'windows': int(mask.sum()),
            'class_counts': np.bincount(y[mask], minlength=2).tolist(),
            'cnn': score(y[mask], predicted[mask], probabilities[mask]),
            'persistence': score(y[mask], persistence[mask]),
        })
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.output / 'test_predictions.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['test_row','trajectory_id','target_cycle','target_raw_row','actual',
                         'cnn_prediction','cnn_probability_class_1','persistence_prediction'])
        for i in range(len(y)):
            writer.writerow([i, test['trajectory_ids'][i], test['target_cycle'][i],
                             test['source_rows'][i,-1], y[i], predicted[i],
                             float(probabilities[i]), persistence[i]])
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
