"""Generate results graphs from saved metrics without rerunning experiments."""
import argparse
from pathlib import Path

from _bootstrap import ROOT
import pandas as pd
from hedging.viz.research_plots import results_summary_plots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=ROOT / "results/v2/evaluation/metrics.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/v2/evaluation/figures")
    args = parser.parse_args()
    results_summary_plots(pd.read_csv(args.metrics), args.output_dir)
    print(f"Wrote results_comparison and results_by_setting (PNG and PDF) to {args.output_dir}")


if __name__ == "__main__":
    main()
