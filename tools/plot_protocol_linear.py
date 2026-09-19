"""Plot the declared F504 text+core-minus-core contrasts from the saved CSV only.

The plot consumes saved point estimates, pointwise confidence intervals and
Bonferroni-adjusted p-values. It does not recompute inference or select models.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ESTIMATORS = ["ols", "ridge", "lasso", "enet"]
LABELS = {"ols": "OLS", "ridge": "Ridge", "lasso": "Lasso", "enet": "Elastic net"}
COLORS = {"raw": "#24699B", "dgtw": "#AD542E"}
TARGETS = {"raw": "Raw", "dgtw": "DGTW"}
METRICS = ["rank_ic", "ew_spread_bp"]


def selected_contrasts(source):
    data = pd.read_csv(source)
    chosen = []
    for target in TARGETS:
        for estimator in ESTIMATORS:
            model = f"{estimator}_textcore_{target}_fit504_val126"
            baseline = f"{estimator}_core_{target}_fit504_val126"
            for metric in METRICS:
                rows = data[(data.model_id == model) & (data.benchmark_id == baseline)
                            & (data.target == target) & (data.metric == metric)
                            & (data.family == "feature") & (data.comparison == "versus_core")
                            & (data.period == "full") & (data.hac_lags == 5)]
                if len(rows) != 1:
                    raise ValueError(f"Expected one registered contrast for {model}, {metric}")
                row = rows.iloc[0]
                values = row[["mean", "ci_low", "ci_high", "p_bonferroni"]].to_numpy(dtype=float)
                if not np.isfinite(values).all() or not row.ci_low <= row["mean"] <= row.ci_high:
                    raise ValueError(f"Invalid saved estimate or interval for {model}, {metric}")
                if int(row.family_size) != 48:
                    raise ValueError("This figure requires the declared 48-comparison linear feature family")
                chosen.append(dict(target=target, estimator=estimator, metric=metric,
                                   mean=float(row["mean"]), ci_low=float(row.ci_low),
                                   ci_high=float(row.ci_high), adjusted_p=float(row.p_bonferroni)))
    return pd.DataFrame(chosen)


def plot(source, output):
    source, output = Path(source), Path(output)
    frame = selected_contrasts(source)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 13, "axes.labelsize": 11,
                         "svg.hashsalt": "protocol_v1_1_linear_textcore_f504",
                         "svg.fonttype": "none", "axes.unicode_minus": True})
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.8), sharey=True)
    fig.subplots_adjust(left=.17, right=.975, top=.80, bottom=.285, wspace=.21)
    fig.suptitle("Adding text to sentiment and attention", x=.17, y=.97,
                 ha="left", fontsize=19, fontweight="bold", color="#172B3A")
    fig.text(.17, .91, "Paired differences: text + core − core, within each linear estimator",
             ha="left", fontsize=11, color="#52616B")
    positions = [8, 7, 6, 5, 3, 2, 1, 0]
    order = [(target, estimator) for target in TARGETS for estimator in ESTIMATORS]
    row_labels = [f"{TARGETS[target]} · {LABELS[estimator]}" for target, estimator in order]
    titles = ["A   Rank correlation", "B   Equal-weighted decile spread"]
    xlabels = ["Change in mean daily rank IC", "Change in gross spread (basis points/day)"]
    for ax, metric, title, xlabel in zip(axes, METRICS, titles, xlabels):
        ax.set_title(title, loc="left", pad=16, fontweight="bold", color="#172B3A")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E6EBEF", linewidth=.8)
        ax.axvline(0, color="#44535E", linestyle=(0, (3, 3)), linewidth=1.15, zorder=2)
        ax.axhline(4, color="#E0E6EA", linewidth=.9)
        for y, (target, estimator) in zip(positions, order):
            row = frame[(frame.target == target) & (frame.estimator == estimator)
                        & (frame.metric == metric)].iloc[0]
            significant = row.adjusted_p < .05
            ax.errorbar(row["mean"], y,
                        xerr=[[row["mean"]-row.ci_low], [row.ci_high-row["mean"]]],
                        fmt="o", markersize=7.7, markeredgewidth=1.6,
                        markerfacecolor=COLORS[target] if significant else "white",
                        markeredgecolor=COLORS[target], ecolor=COLORS[target],
                        elinewidth=1.8, capsize=3.1, capthick=1.3, zorder=3)
        ax.set_ylim(-.65, 8.65)
        ax.set_yticks(positions, labels=row_labels)
        ax.tick_params(axis="y", length=0, pad=12)
        ax.tick_params(axis="x", length=3, color="#A1ABB3", labelcolor="#344854")
        ax.set_xlabel(xlabel, labelpad=13)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#A1ABB3")
    for tick, (target, _) in zip(axes[0].get_yticklabels(), order):
        tick.set_color(COLORS[target])
    axes[0].set_xlim(-.00065, .0085)
    axes[0].xaxis.set_major_locator(MultipleLocator(.002))
    axes[0].xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axes[1].set_xlim(-13.1, 1.05)
    axes[1].xaxis.set_major_locator(MultipleLocator(2))
    handles = [Line2D([], [], marker="o", color="none", markeredgecolor="#4E5C65",
                      markerfacecolor=face, markeredgewidth=1.5, markersize=7,
                      label=label) for face, label in [("#4E5C65", "Adjusted p < 0.05"),
                                                      ("white", "Adjusted p ≥ 0.05")]]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.165, .135),
               frameon=False, ncol=2, handletextpad=.65, columnspacing=2.1)
    caption = ("Bars show pointwise 95% HAC intervals (5 lags). Marker fill uses Bonferroni-adjusted p-values;\n"
               "the correction family has 48 feature comparisons per target and metric. F504 / V126, 2014–2022.\n"
               "Gross spreads are before costs. Source: protocol_v1_1_linear_contrasts.csv.")
    fig.text(.17, .107, caption, va="top", ha="left", fontsize=9.1,
             color="#52616B", linespacing=1.55)
    output.parent.mkdir(parents=True, exist_ok=True)
    png, svg = output.with_suffix(".png"), output.with_suffix(".svg")
    description = "F504 text+core minus core; eight registered estimator/target contrasts; saved pointwise HAC95% intervals, family48 adjusted p-values. Source SHA256: " + source_hash
    fig.savefig(png, dpi=220, facecolor="white", metadata={"Software": "plot_protocol_linear.py", "Description": description})
    fig.savefig(svg, facecolor="white", metadata={"Date": None, "Title": "Adding text to sentiment and attention", "Description": description})
    plt.close(fig)
    print(f"Saved {png}\nSaved {svg}\nSource SHA256: {source_hash}")
    return png, svg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT/"reports/data/protocol_v1_1_linear_contrasts.csv")
    parser.add_argument("--out", type=Path, default=ROOT/"reports/figures/protocol_v1_1_linear_textcore_f504")
    args = parser.parse_args(argv)
    return plot(args.source, args.out)


if __name__ == "__main__":
    main()
