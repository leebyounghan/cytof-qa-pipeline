"""Check imports and small data operations without models, GPUs, or API calls."""

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["inference", "preprocess"], required=True)
    args = parser.parse_args()
    names = ["numpy", "pandas", "pyarrow", "yaml"]
    if args.profile == "preprocess":
        names += ["fcsparser"]
    else:
        names += ["scipy", "sklearn", "skimage", "matplotlib", "minisom",
                  "openai", "cv2", "numba", "torch", "torchvision", "torchaudio", "vllm"]
    for name in names:
        importlib.import_module(name)

    import numpy as np
    import pandas as pd
    checks = ["imports"]
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        frame = pd.DataFrame({"marker": np.array([0.1, 1.0, 3.0])})
        path = directory / "cells.parquet"
        frame.to_parquet(path, index=False)
        pd.testing.assert_frame_equal(frame, pd.read_parquet(path))
        checks.append("parquet_roundtrip")
        if args.profile == "preprocess":
            import fcsparser
            examples = sorted(Path(fcsparser.__file__).parent.rglob("*.fcs"))
            if not examples:
                raise RuntimeError("No bundled FCS fixture found for parsing validation")
            metadata, cells = fcsparser.parse(str(examples[0]), reformat_meta=True)
            if not metadata or cells.empty:
                raise RuntimeError("Bundled FCS fixture returned no cells")
            checks.append("bundled_fcs_parse")
        else:
            from scipy.stats import gaussian_kde
            from sklearn.cluster import KMeans
            from minisom import MiniSom
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import torch
            values = np.random.default_rng(42).normal(size=(100, 2))
            density = gaussian_kde(values[:, 0])(np.linspace(-2, 2, 10))
            if not np.isfinite(density).all():
                raise RuntimeError("Nonfinite KDE values")
            KMeans(n_clusters=2, random_state=42, n_init=2).fit(values)
            som = MiniSom(2, 2, 2, random_seed=42)
            som.train_random(values, 10)
            plt.scatter(values[:, 0], values[:, 1], s=2)
            plt.savefig(directory / "scatter.png")
            plt.close()
            torch.testing.assert_close(torch.ones(2, 2) @ torch.ones(2, 2), torch.full((2, 2), 2.0))
            checks.extend(["kde", "kmeans", "minisom", "scatter_render", "torch_cpu"])
    print(json.dumps({"profile": args.profile, "passed": checks,
                      "numpy": importlib.metadata.version("numpy")}, indent=2))


if __name__ == "__main__":
    main()
