"""
FlowSOM clustering baseline for CytoGate-Bench.

FlowSOM = SOM + consensus metaclustering of the SOM codebook:

  1. Train a square SOM on the (x, y) plane (grid × grid neurons).
  2. Hierarchically cluster the SOM codebook (neuron weight vectors) into
     K metaclusters (average linkage — the canonical FlowSOM choice).
  3. Each cell is assigned to its winning neuron, then to that neuron's
     metacluster.
  4. Greedy majority-vote mapping for metacluster → GT label.
  5. Output prediction.json.

Usage:
    python -m src.baselines.flowsom_baseline \
        --benchmark benchmark/ \
        --data-dir data/ \
        --output-dir results/flowsom/ \
        --datasets FR-FCM-Z74D_hc \
        --flowsom-grid 10 --flowsom-k 20
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from src.bench import BenchmarkLoader, TaskInput


def hungarian_match(
    cluster_labels: np.ndarray,
    gt_labels: np.ndarray,
    categories: list[str],
) -> dict[int, str | None]:
    """Greedy cluster → GT category mapping (many-to-one, by majority vote).

    - All categories including "Unassigned" are valid match targets.
    - Noise labels (-1) are mapped to None (NaN in output).
    """
    non_noise = cluster_labels != -1
    mapping: dict[int, str | None] = {-1: None}
    if not non_noise.any():
        return mapping

    ct = pd.crosstab(cluster_labels[non_noise], gt_labels[non_noise])
    ct = ct.reindex(columns=categories, fill_value=0)
    best_cat = ct.idxmax(axis=1)
    max_count = ct.max(axis=1)
    for cid, cat, cnt in zip(best_cat.index, best_cat.values, max_count.values):
        mapping[int(cid)] = cat if cnt > 0 else None
    return mapping


SUBSAMPLE_THRESHOLD = 15_000
SUBSAMPLE_SIZE = 15_000
SUBSAMPLE_SEED = 0
DEFAULT_GRID = 10
DEFAULT_K = 20
DEFAULT_ITERATIONS = 10_000


def _train_som(
    X: np.ndarray,
    grid: int,
    seed: int,
    n_iterations: int,
):
    """Train a square MiniSom on ``X`` and return the trained SOM."""
    try:
        from minisom import MiniSom
    except ImportError as e:
        raise ImportError(
            "minisom is not installed. Run: uv add minisom"
        ) from e
    som = MiniSom(
        grid, grid, X.shape[1],
        sigma=1.0, learning_rate=0.5, random_seed=seed,
    )
    som.random_weights_init(X)
    som.train(X, num_iteration=n_iterations, verbose=False)
    return som


def _consensus_metacluster(codebook: np.ndarray, k: int) -> np.ndarray:
    """Hierarchical (average linkage) metaclustering of SOM neuron weights.

    Returns ``meta_of_node`` (n_nodes,) — metacluster id per SOM neuron.
    Average linkage matches FlowSOM's canonical ConsensusClusterPlus call.
    """
    n_nodes = codebook.shape[0]
    k_eff = max(1, min(int(k), n_nodes))
    if k_eff >= n_nodes:
        return np.arange(n_nodes, dtype=np.int32)
    return AgglomerativeClustering(
        n_clusters=k_eff, linkage='average',
    ).fit_predict(codebook).astype(np.int32)


def _assign_neurons(som, X: np.ndarray, grid: int) -> np.ndarray:
    """Return flat neuron-id (∈ [0, grid²)) per row of ``X``."""
    winners = np.array([som.winner(x) for x in X], dtype=np.int32)
    return (winners[:, 0] * grid + winners[:, 1]).astype(np.int32)


def run_flowsom_task(
    X_2d: np.ndarray,
    gt_labels: np.ndarray,
    categories: list[str],
    grid: int = DEFAULT_GRID,
    k: int = DEFAULT_K,
    n_iterations: int = DEFAULT_ITERATIONS,
) -> tuple[np.ndarray, dict]:
    """Train SOM + consensus metacluster on 2D data and return predicted labels.

    For samples larger than SUBSAMPLE_THRESHOLD, the SOM is trained on a
    uniform random subsample of SUBSAMPLE_SIZE points; consensus metaclustering
    runs on the SOM codebook (independent of N), and every parent cell is
    then assigned to its winning neuron directly (argmax over neuron
    weights — equivalent to 1-NN on the SOM codebook).
    """
    n_total = len(X_2d)

    # MiniSom needs at least 1 cell to train; fall back to all-None for
    # truly empty parents so the step is scored.
    if n_total < 2:
        pred = np.array([None] * n_total, dtype=object)
        info = {
            "flowsom_grid":      grid,
            "flowsom_k":         k,
            "n_total_neurons":   grid * grid,
            "n_clusters":        0,
            "subsampled":        False,
            "n_fit":             n_total,
            "n_iterations":      n_iterations,
            "cluster_mapping":   {},
            "skipped":           "parent < 2 cells",
        }
        return pred, info

    subsampled = n_total > SUBSAMPLE_THRESHOLD
    if subsampled:
        rng = np.random.default_rng(SUBSAMPLE_SEED)
        fit_idx = rng.choice(n_total, SUBSAMPLE_SIZE, replace=False)
        X_fit = X_2d[fit_idx]
    else:
        X_fit = X_2d

    som = _train_som(X_fit, grid=grid, seed=SUBSAMPLE_SEED,
                     n_iterations=n_iterations)
    codebook = som.get_weights().reshape(-1, X_fit.shape[1])  # (grid², d)

    meta_of_node = _consensus_metacluster(codebook, k=k)
    neuron_ids = _assign_neurons(som, X_2d, grid=grid)
    cluster_labels = meta_of_node[neuron_ids].astype(np.int32)

    n_clusters = int(len(np.unique(cluster_labels)))

    mapping = hungarian_match(cluster_labels, gt_labels, categories)
    pred_labels = pd.Series(cluster_labels).map(mapping).to_numpy(dtype=object)

    info = {
        "flowsom_grid":    grid,
        "flowsom_k":       k,
        "n_total_neurons": grid * grid,
        "n_clusters":      n_clusters,
        "subsampled":      subsampled,
        "n_fit":           len(X_fit),
        "n_iterations":    n_iterations,
        "cluster_mapping": {str(k_): v for k_, v in mapping.items()},
    }
    return pred_labels, info


def run_on_task(
    ti: TaskInput, output_dir: Path,
    grid: int, k: int, n_iterations: int,
) -> None:
    pred_labels, info = run_flowsom_task(
        ti.X_2d, ti.y, ti.categories,
        grid=grid, k=k, n_iterations=n_iterations,
    )
    ti.save_prediction(pred_labels, output_dir=output_dir, method="flowsom", info=info)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="FlowSOM clustering baseline")
    parser.add_argument("--output-dir", type=Path, default=Path("results/flowsom/"))
    parser.add_argument("--flowsom-grid", type=int, default=DEFAULT_GRID,
                        help=f"SOM grid size; total SOM neurons = grid². Default: {DEFAULT_GRID}.")
    parser.add_argument("--flowsom-k", type=int, default=DEFAULT_K,
                        help=f"Consensus metacluster count (final K). Default: {DEFAULT_K}.")
    parser.add_argument("--n-iterations", type=int, default=DEFAULT_ITERATIONS,
                        help=f"SOM training iterations. Default: {DEFAULT_ITERATIONS}.")
    BenchmarkLoader.add_cli_args(parser)
    args = parser.parse_args()

    loader = BenchmarkLoader.from_cli(args)
    print(f"FlowSOM baseline (grid={args.flowsom_grid}² = "
          f"{args.flowsom_grid ** 2} neurons → K={args.flowsom_k}) → {args.output_dir}")

    n_tasks = 0
    current_ds = current_sample = None
    for ti in loader.iter_tasks_from_cli(args):
        if ti.dataset != current_ds:
            current_ds, current_sample = ti.dataset, None
            print(f"\n{'='*60}\nDataset: {ti.dataset}\n{'='*60}")
        if ti.sample != current_sample:
            current_sample = ti.sample
            print(f"  {ti.sample}")

        run_on_task(
            ti, args.output_dir,
            grid=args.flowsom_grid, k=args.flowsom_k,
            n_iterations=args.n_iterations,
        )
        n_tasks += 1

    print(f"\nDone. Total: {n_tasks} tasks")


if __name__ == "__main__":
    main()
