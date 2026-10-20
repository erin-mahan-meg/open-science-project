#!/usr/bin/env python3
"""Inspect the raw simulated data and report subject-level quality issues."""

from pathlib import Path

import pandas as pd


project_dir = Path(__file__).resolve().parent.parent
data_file = project_dir / "data" / "simulated_epilepsy_surgery_raw.csv"
output_dir = project_dir / "outputs"
output_dir.mkdir(exist_ok=True)

data = pd.read_csv(data_file)
issues = []


def add_issue(patient_id, issue_type, variable, observed_value, explanation):
    """Add one subject-level data-quality issue to the results table."""
    issues.append(
        {
            "patient_id": patient_id,
            "issue_type": issue_type,
            "variable": variable,
            "observed_value": observed_value,
            "explanation": explanation,
        }
    )


# Flag missing values in clinical variables.
for variable in data.columns:
    if variable == "patient_id":
        continue

    for patient_id in data.loc[data[variable].isna(), "patient_id"]:
        if variable.endswith("localization_result"):
            explanation = (
                "Investigation result is missing rather than marked as Not performed."
            )
        else:
            explanation = "Required value is missing."
        add_issue(patient_id, "Missing value", variable, "NA", explanation)


# Flag impossible clinical chronology.
invalid_onset = data[
    data["age_at_epilepsy_onset_years"] > data["age_at_surgery_years"]
]
for _, row in invalid_onset.iterrows():
    add_issue(
        row["patient_id"],
        "Implausible value",
        "age_at_epilepsy_onset_years",
        row["age_at_epilepsy_onset_years"],
        "Epilepsy onset occurs after the recorded age at surgery.",
    )


# Flag negative epilepsy duration.
negative_duration = data[data["epilepsy_duration_years"] < 0]
for _, row in negative_duration.iterrows():
    add_issue(
        row["patient_id"],
        "Implausible value",
        "epilepsy_duration_years",
        row["epilepsy_duration_years"],
        "Epilepsy duration cannot be negative.",
    )


# Flag an extreme seizure frequency for review.
extreme_frequency = data[data["monthly_seizure_frequency"] > 100]
for _, row in extreme_frequency.iterrows():
    add_issue(
        row["patient_id"],
        "Implausible value",
        "monthly_seizure_frequency",
        row["monthly_seizure_frequency"],
        "Value exceeds the prespecified review threshold of 100 seizures per month.",
    )


# Flag unexpected capitalization or category values.
valid_localizations = {
    "Temporal",
    "Frontal",
    "Parietal/occipital",
    "Hemispheric/multilobar",
}
invalid_localization = data[~data["epilepsy_localization"].isin(valid_localizations)]
for _, row in invalid_localization.iterrows():
    add_issue(
        row["patient_id"],
        "Invalid category",
        "epilepsy_localization",
        row["epilepsy_localization"],
        "Value does not match one of the permitted category labels.",
    )


# Flag patient identifiers that occur more than once.
duplicate_ids = data.loc[data["patient_id"].duplicated(keep=False), "patient_id"].unique()
for patient_id in duplicate_ids:
    number_of_records = int((data["patient_id"] == patient_id).sum())
    add_issue(
        patient_id,
        "Duplicate record",
        "patient_id",
        f"{number_of_records} records",
        "Patient identifier occurs more than once in the raw dataset.",
    )


issue_table = pd.DataFrame(issues).sort_values(
    ["patient_id", "issue_type", "variable"]
)
output_file = output_dir / "data_quality_issues.csv"
issue_table.to_csv(output_file, index=False)

print(f"Checked {len(data)} rows representing {data['patient_id'].nunique()} IDs.")
print(f"Flagged {len(issue_table)} subject-and-issue combinations.")
print(f"Affected subjects: {issue_table['patient_id'].nunique()}")
print(f"Saved data-quality table to: {output_file}")
