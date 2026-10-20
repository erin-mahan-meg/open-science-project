#!/usr/bin/env python3
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


def calculate_modality_coverage():
    """Return counts and percentages of patients receiving each modality."""
    modality_columns = {
        "MRI": "mri_performed",
        "Scalp EEG": "scalp_eeg_performed",
        "PET": "pet_performed",
        "MEG": "meg_performed",
        "Intracranial EEG": "intracranial_eeg_performed",
    }
    total_patients = len(data)

    coverage = pd.DataFrame(
        [
            {
                "modality": modality,
                "patients": data[column].eq("Yes").sum(),
                "total_patients": total_patients,
            }
            for modality, column in modality_columns.items()
        ]
    )
    coverage["percentage"] = (
        coverage["patients"] / coverage["total_patients"] * 100
    )
    return coverage.sort_values("percentage", ascending=True).reset_index(drop=True)


def plot_modality_coverage():
    """Return a horizontal bar chart of presurgical modality coverage."""
    coverage = calculate_modality_coverage()

    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.barh(
        coverage["modality"],
        coverage["percentage"],
        color="#3B82A0",
        edgecolor="white",
        height=0.68,
    )

    labels = [
        f"{row.percentage:.1f}% ({row.patients}/{row.total_patients})"
        for row in coverage.itertuples()
    ]
    axis.bar_label(
        bars,
        labels=labels,
        padding=5,
        fontsize=10,
    )

    axis.set_title("Presurgical modality coverage", fontweight="bold")
    axis.set_xlabel("Patients who received the modality")
    axis.set_ylabel("")
    axis.set_xlim(0, 112)
    axis.xaxis.set_major_formatter(PercentFormatter(100))
    axis.grid(axis="y", visible=False)
    sns.despine(ax=axis)
    figure.tight_layout()
    return figure


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

if __name__ == "__main__":
    plot_modality_coverage()
    plot_concordance_by_outcome()
    plt.show()
