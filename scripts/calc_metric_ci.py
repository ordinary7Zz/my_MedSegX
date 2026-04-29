import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_ci(values, n_boot=2000, ci=95, seed=42):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    n = values.size
    boot_means = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = (100 - ci) / 2
    mean = values.mean()
    lower = np.percentile(boot_means, alpha)
    upper = np.percentile(boot_means, 100 - alpha)
    return mean, lower, upper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="playground/MedSegX/external",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="playground/MedSegX/external/thyroid_ci_summary.csv",
    )
    parser.add_argument("--pattern", type=str, default="*-RealWorld-site.csv")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--ci", type=float, default=95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    csv_files = sorted(input_dir.glob(args.pattern))

    rows = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        hd_col = "HD95" if "HD95" in df.columns else "HD"

        if "DSC" not in df.columns or hd_col not in df.columns:
            continue

        dsc_mean, dsc_lower, dsc_upper = bootstrap_ci(
            df["DSC"].to_numpy(),
            n_boot=args.n_boot,
            ci=args.ci,
            seed=args.seed,
        )
        hd_mean, hd_lower, hd_upper = bootstrap_ci(
            df[hd_col].to_numpy(),
            n_boot=args.n_boot,
            ci=args.ci,
            seed=args.seed,
        )

        rows.append(
            {
                "dataset": csv_file.stem.replace("-RealWorld-site", ""),
                "n_case": len(df),
                "DSC_mean": dsc_mean,
                "DSC_ci_lower": dsc_lower,
                "DSC_ci_upper": dsc_upper,
                "HD95_mean": hd_mean,
                "HD95_ci_lower": hd_lower,
                "HD95_ci_upper": hd_upper,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)
    print(out_df.to_string(index=False))
    print(f"\nsaved to: {args.output}")


if __name__ == "__main__":
    main()
