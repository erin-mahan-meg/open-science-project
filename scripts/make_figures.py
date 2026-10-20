#!/usr/bin/env python3
"""Create the two main figures for the Quarto report."""
"""Create figures for the Quarto report."""

import os
import tempfile
from pathlib import Path

# Use a writable temporary directory for Matplotlib's font cache.
cache_dir = Path(tempfile.gettempdir()) / "epilepsy-figures-matplotlib"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal, spearmanr


# This script is stored in project_folder/scripts/ and the CSV is in data/.
project_dir = Path(__file__).resolve().parent.parent
data_file = project_dir / "data" / "simulated_epilepsy_surgery_clean.csv"

data = pd.read_csv(data_file)

engel_order = ["I", "II", "III", "IV"]
engel_colors = {
    "I": "#0072B2",
    "II": "#56B4E9",
    "III": "#E69F00",
    "IV": "#D55E00",
}

sns.set_theme(style="whitegrid", context="notebook")


def format_p_value(p_value):
    """Format a p-value for compact display in a figure."""
    return "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"


def run_statistical_tests():
    """Return the overall statistical tests used in the two figures."""
    concordance_data = data.dropna(
        subset=["multimodal_concordance_count", "engel_outcome_1yr"]
    ).copy()
    concordance_data["engel_outcome_numeric"] = concordance_data[
        "engel_outcome_1yr"
    ].map({"I": 1, "II": 2, "III": 3, "IV": 4})
    rho, spearman_p = spearmanr(
        concordance_data["multimodal_concordance_count"],
        concordance_data["engel_outcome_numeric"],
    )

    duration_groups = [
        data.loc[
            data["engel_outcome_1yr"] == engel_class,
            "epilepsy_duration_years",
        ].dropna()
        for engel_class in engel_order
    ]
    h_statistic, kw_p = kruskal(*duration_groups)

    return {
        "concordance": {
            "test": "Spearman rank correlation",
            "statistic": rho,
            "p_value": spearman_p,
            "sample_size": len(concordance_data),
        },
        "duration": {
            "test": "Kruskal-Wallis test",
            "statistic": h_statistic,
            "degrees_of_freedom": len(duration_groups) - 1,
            "p_value": kw_p,
        },
    }


def plot_concordance_by_outcome():
    """Return a 100% stacked bar chart of Engel outcome by concordance count."""
    test = run_statistical_tests()["concordance"]
    counts = pd.crosstab(
        data["multimodal_concordance_count"],
        data["engel_outcome_1yr"],
    ).reindex(columns=engel_order, fill_value=0)

    percentages = counts.div(counts.sum(axis=1), axis=0) * 100

    figure, axis = plt.subplots(figsize=(8, 5.5))
    percentages.plot(
        kind="bar",
        stacked=True,
        color=[engel_colors[label] for label in engel_order],
        width=0.78,
        edgecolor="white",
        linewidth=0.7,
        ax=axis,
    )

    # Add percentage labels only when a segment is large enough to read.
    for container in axis.containers:
        labels = [
            f"{bar.get_height():.0f}%" if bar.get_height() >= 8 else ""
            for bar in container
        ]
        axis.bar_label(
            container,
            labels=labels,
            label_type="center",
            color="white",
            fontsize=9,
            fontweight="bold",
        )

    # Show the number of patients represented by each stacked bar.
    for position, sample_size in enumerate(counts.sum(axis=1)):
        axis.text(
            position,
            102,
            f"n={sample_size}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axis.set_title("Engel outcome by multimodal concordance", fontweight="bold")
    axis.text(
        0.5,
        1.02,
        (
            f"Spearman: ρ = {test['statistic']:.2f}, "
            f"{format_p_value(test['p_value'])}"
        ),
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
    )
    axis.set_xlabel("Number of concordant presurgical modalities")
    axis.set_ylabel("Patients within concordance group")
    axis.set_ylim(0, 109)
    axis.yaxis.set_major_formatter(PercentFormatter(100))
    axis.legend(
        title="Engel class",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )
    axis.grid(axis="x", visible=False)
    sns.despine(ax=axis)
    figure.tight_layout()
    return figure


def plot_duration_by_outcome():
    """Return a boxplot and patient-level points for epilepsy duration."""
    test = run_statistical_tests()["duration"]
    plot_data = data.dropna(
        subset=["engel_outcome_1yr", "epilepsy_duration_years"]
    ).copy()

    figure, axis = plt.subplots(figsize=(8, 5.5))

    sns.boxplot(
        data=plot_data,
        x="engel_outcome_1yr",
        y="epilepsy_duration_years",
        order=engel_order,
        color="#B8D4E8",
        width=0.55,
        showfliers=False,
        linewidth=1.2,
        ax=axis,
    )

    # Set a seed so the point jitter is reproducible when the report is rerendered.
    np.random.seed(20261020)
    sns.stripplot(
        data=plot_data,
        x="engel_outcome_1yr",
        y="epilepsy_duration_years",
        order=engel_order,
        color="#24445C",
        alpha=0.48,
        size=3.5,
        jitter=0.22,
        ax=axis,
    )

    sample_sizes = plot_data["engel_outcome_1yr"].value_counts()
    axis.set_xticks(
        range(len(engel_order)),
        [f"Engel {label}\nn={sample_sizes.get(label, 0)}" for label in engel_order],
    )
    axis.set_title("Epilepsy duration by Engel outcome", fontweight="bold")
    axis.text(
        0.5,
        1.02,
        (
            f"Kruskal–Wallis: H({test['degrees_of_freedom']}) = "
            f"{test['statistic']:.2f}, {format_p_value(test['p_value'])}"
        ),
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
    )
    axis.set_xlabel("")
    axis.set_ylabel("Epilepsy duration (years)")
    axis.grid(axis="x", visible=False)
    sns.despine(ax=axis)
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    plot_concordance_by_outcome()
    plot_duration_by_outcome()
    plt.show()