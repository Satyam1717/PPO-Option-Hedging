"""Publication-friendly figures for risk, costs and distribution shifts."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "ppo_band": "PPO band",
    "ppo_direct": "PPO direct",
    "ppo_band_blind_cost": "PPO band (cost masked)",
    "delta_band": "Tuned delta band",
    "adaptive_bs": "BS rolling volatility",
    "quote_bs": "BS inception volatility",
    "static_bs": "BS fixed volatility",
    "periodic_delta": "Five-day delta",
    "constant_half": "Constant half hedge",
    "no_hedge": "No hedge",
}

REGIME_LABELS = {
    "gbm_low_vol": "GBM 15% (training)",
    "gbm_high_vol": "GBM 40% (training)",
    "heston": "Heston (training)",
    "gbm_interpolation": "GBM 25% (held out)",
    "gbm_extrapolation": "GBM 50% (held out)",
    "heston_stress": "Heston stress (held out)",
    "volatility_switch": "Volatility switch (held out)",
}


def results_summary_plots(metrics, directory):
    """Plot saved evaluation metrics with the same averaging used in RESULTS.md."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    settings = metrics.groupby(["regime", "cost_bps", "group"], as_index=False).mean(numeric_only=True)
    summary = settings.groupby("group").mean(numeric_only=True).sort_values("mean_utility", ascending=False)
    labels = [METHOD_LABELS.get(group, group) for group in summary.index]
    colors = ["#137d91" if group.startswith("ppo_") else "#8497ab" for group in summary.index]

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
        panels = [("mean_utility", "Terminal utility", "Higher is better"),
                  ("rmse", "Hedging error (RMSE)", "Lower is better"),
                  ("mean_total_cost", "Transaction cost", "Lower is better")]
        positions = np.arange(len(summary))
        for ax, (column, title, direction) in zip(axes, panels):
            values = summary[column].to_numpy()
            bars = ax.barh(positions, values, color=colors, height=.66)
            ax.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=5, fontsize=9)
            ax.axvline(0, color="#536173", linewidth=.8)
            ax.set_title(f"{title}\n{direction}", loc="left", pad=16, fontsize=12, weight="bold")
            ax.set_xlabel("Per option, in simulated currency units" if column != "mean_utility" else "Mean quadratic utility")
            ax.grid(axis="x", alpha=.16)
            ax.set_axisbelow(True)
            ax.margins(x=.22)
        axes[0].set_yticks(positions, labels)
        axes[0].invert_yaxis()
        fig.suptitle("Option hedging: measured strategy comparison", fontsize=18, weight="bold", y=.98)
        fig.text(.5, .915, "Training seeds averaged within each setting; all 28 market/cost settings weighted equally.",
                 ha="center", color="#536173")
        fig.text(.5, .025, "Teal: PPO variants. Grey: comparison strategies. Descriptive averages; no confidence intervals shown.",
                 ha="center", fontsize=10, color="#536173")
        fig.tight_layout(rect=(0, .065, 1, .88), w_pad=2)
        for extension in ("png", "pdf"):
            fig.savefig(directory / f"results_comparison.{extension}", dpi=200, bbox_inches="tight")
        plt.close(fig)

        ppo = settings[settings.group == "ppo_band"]
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
        titles = [("delta_band", "Band PPO vs tuned delta band"),
                  ("ppo_band_blind_cost", "Cost feature: included vs masked")]
        for ax, (reference, title) in zip(axes, titles):
            comparison = ppo.merge(settings[settings.group == reference], on=["regime", "cost_bps"],
                                   suffixes=("_ppo", "_reference"), validate="one_to_one")
            comparison["difference"] = comparison.mean_utility_ppo - comparison.mean_utility_reference
            pivot = comparison.pivot(index="regime", columns="cost_bps", values="difference")
            order = [regime for regime in REGIME_LABELS if regime in pivot.index]
            pivot = pivot.reindex(order).sort_index(axis=1)
            values = pivot.to_numpy()
            limit = max(.001, np.abs(values).max())
            im = ax.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
            ax.set_xticks(range(len(pivot.columns)), [f"{value:g}" for value in pivot.columns])
            ax.set_yticks(range(len(pivot.index)), [REGIME_LABELS[regime] for regime in pivot.index])
            ax.set_xlabel("Transaction cost (basis points)")
            wins = int((values > 0).sum())
            ax.set_title(f"{title}\nPositive in {wins}/{values.size} settings", fontsize=12, pad=14, weight="bold")
            for row, column in np.ndindex(values.shape):
                value = values[row, column]
                label = f"{value:+.3f}" if abs(value) >= .0005 else f"{value:+.2g}"
                ax.text(column, row, label, ha="center", va="center", fontsize=10,
                        color="white" if abs(value) > .6 * limit else "#152334")
            fig.colorbar(im, ax=ax, shrink=.8, label="Mean utility difference")
        fig.suptitle("Where band PPO helps, and where it falls behind", fontsize=18, weight="bold", y=.98)
        fig.text(.5, .025, "Seed-averaged utility differences. Blue: positive. Red: negative. Each panel has its own colour scale.",
                 ha="center", color="#536173", fontsize=10)
        fig.tight_layout(rect=(0, .06, 1, .90), w_pad=2)
        for extension in ("png", "pdf"):
            fig.savefig(directory / f"results_by_setting.{extension}", dpi=200, bbox_inches="tight")
        plt.close(fig)


def research_plots(metrics, comparisons, examples, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    averages = metrics.groupby(["regime", "cost_bps", "group"], as_index=False).mean(numeric_only=True)
    band = averages[averages.group == "delta_band"][["regime", "cost_bps", "mean_utility"]]
    band = band.rename(columns={"mean_utility": "reference_utility"})
    ppo = averages[averages.group == "ppo_band"].merge(band, on=["regime", "cost_bps"])
    if len(ppo):
        ppo["advantage"] = ppo.mean_utility - ppo.reference_utility
        pivot = ppo.pivot(index="regime", columns="cost_bps", values="advantage")
        fig, ax = plt.subplots(figsize=(7, 4))
        limit = max(0.01, np.abs(pivot.to_numpy()).max())
        im = ax.imshow(pivot, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), [f"{x:g}" for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_xlabel("Transaction cost (bps)")
        ax.set_title("Band PPO utility minus validation-tuned delta band")
        for i in range(len(pivot)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f"{pivot.iloc[i,j]:+.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="Mean utility difference (positive favors PPO)")
        fig.tight_layout(); fig.savefig(directory / "generalization.png", dpi=160); plt.close(fig)
    # Exclude no-hedge/constant controls from the plot to keep the hedge comparison legible.
    fig, ax = plt.subplots(figsize=(7, 4))
    for group, g in averages.groupby("group"):
        if group in ("no_hedge", "constant_half"):
            continue
        ax.scatter(g.mean_total_cost, g.rmse, s=22, alpha=0.6, label=group)
    ax.set_xlabel("Mean transaction cost"); ax.set_ylabel("Terminal hedge RMSE")
    ax.set_title("Risk and cost across all evaluation settings")
    ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(directory / "risk_cost.png", dpi=160); plt.close(fig)
    for name, methods in examples.items():
        fig, axes = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True)
        axes[0].plot(next(iter(methods.values()))["prices"], color="black")
        axes[0].set_ylabel("Stock price"); axes[0].set_title(name)
        for method, ex in methods.items():
            axes[1].step(range(len(ex["hedge_ratios"])), ex["hedge_ratios"], where="post", label=method)
        axes[1].set(xlabel="Trading day", ylabel="Stock holding", ylim=(-0.02, 1.02))
        axes[1].legend(fontsize=7); fig.tight_layout()
        fig.savefig(directory / f"path_{name}.png", dpi=160); plt.close(fig)
