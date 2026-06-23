import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=str, default="runs")
    parser.add_argument("--prefix", type=str, default="spatial_blocks_seed_")
    args = parser.parse_args()

    run_root = Path(args.run_root)

    metric_files = sorted(run_root.glob(f"{args.prefix}*/**/outputs/test_metrics.json"))

    if len(metric_files) == 0:
        raise RuntimeError(
            f"No test_metrics.json files found under {run_root} "
            f"with prefix {args.prefix}"
        )

    rows = []

    for path in metric_files:
        with open(path, "r") as f:
            metrics = json.load(f)

        rows.append(
            {
                "path": str(path),
                "test_mae": float(metrics["test_mae"]),
                "test_rmse": float(metrics["test_rmse"]),
                "test_bias": float(metrics["test_bias"]),
            }
        )

    for row in rows:
        print(
            f"{row['path']}: "
            f"MAE={row['test_mae']:.4f}, "
            f"RMSE={row['test_rmse']:.4f}, "
            f"bias={row['test_bias']:.4f}"
        )

    print("")
    print("Summary:")

    for key in ["test_mae", "test_rmse", "test_bias"]:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        print(
            f"{key}: "
            f"mean={values.mean():.4f}, "
            f"std={values.std(ddof=1):.4f}, "
            f"min={values.min():.4f}, "
            f"max={values.max():.4f}"
        )


if __name__ == "__main__":
    main()