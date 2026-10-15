#!/usr/bin/env python3
"""Import the simulated epilepsy data and make a descriptive-statistics table."""

from pathlib import Path
import pandas as pd


# File locations are relative to this script, so the project folder can be moved.
project_dir = Path(__file__).resolve().parent.parent
data_file = project_dir / "data" / "simulated_epilepsy_surgery_raw.csv"
output_dir = project_dir / "outputs"
output_dir.mkdir(exist_ok=True)

# Import the data and define columns
data = pd.read_csv(data_file)
outcome = "engel_outcome_1yr"
engel_classes = ["I", "II", "III", "IV"]
issues = []

# Define the groups that will become columns in the table.
groups = {f"Engel {label}": data[data[outcome] == label] for label in engel_classes}
groups["Total"] = data

# Add the sample size to each column heading.
column_names = {
    name: f"{name}\n(n={len(group)})" for name, group in groups.items()
}

# functions to get the descriptive stats that we care about
def mean_sd(values: pd.Series) -> str:
    values = values.dropna()
    if values.empty:
        return ""
    return f"{values.mean():.1f} ({values.std():.1f})"


def median_iqr(values: pd.Series) -> str:
    values = values.dropna()
    if values.empty:
        return ""
    return f"{values.median():.1f} [{values.quantile(0.25):.1f}, {values.quantile(0.75):.1f}]"


def count_percent(values: pd.Series, level: object) -> str:
    denominator = values.notna().sum()
    if denominator == 0:
        return ""
    count = (values == level).sum()
    return f"{count} ({100 * count / denominator:.1f}%)"


def missing_count_percent(values: pd.Series) -> str:
    count = values.isna().sum()
    return f"{count} ({100 * count / len(values):.1f}%)"

# for data issues report: 
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

rows = []
row_types = []

# Start with the number of observations in each outcome group.
cohort_row = {"Variable": "Cohort, n"}
for name, group in groups.items():
    cohort_row[column_names[name]] = str(len(group))
rows.append(cohort_row)
row_types.append("cohort")

# Exclude the identifier and outcome because the outcome defines the columns.
variables = [
    column for column in data.columns if column not in ["patient_id", outcome]
]

for variable in variables:
    label = variable.replace("_", " ")

    if pd.api.types.is_numeric_dtype(data[variable]):
        mean_row = {"Variable": f"{label}, mean (SD)"}
        median_row = {"Variable": "    median [Q1, Q3]"}

        for name, group in groups.items():
            mean_row[column_names[name]] = mean_sd(group[variable])
            median_row[column_names[name]] = median_iqr(group[variable])

        rows.extend([mean_row, median_row])
        row_types.extend(["numeric_main", "detail"])

    else:
        # Add a section heading, then one row for every observed category.
        header_row = {"Variable": f"{label}, n (%)"}
        for name in groups:
            header_row[column_names[name]] = ""
        rows.append(header_row)
        row_types.append("categorical_header")

        levels = data[variable].dropna().drop_duplicates().tolist()
        for level in levels:
            level_row = {"Variable": f"    {level}"}
            for name, group in groups.items():
                level_row[column_names[name]] = count_percent(group[variable], level)
            rows.append(level_row)
            row_types.append("detail")

    # Show missingness only where the raw dataset actually contains missing data.
    if data[variable].isna().any():
        missing_row = {"Variable": "    Missing"}
        for name, group in groups.items():
            missing_row[column_names[name]] = missing_count_percent(group[variable])
        rows.append(missing_row)
        row_types.append("missing")


summary = pd.DataFrame(rows)

# Save the underlying table as a CSV file.
summary_file = output_dir / "descriptive_statistics_by_engel.csv"
summary.to_csv(summary_file, index=False)

print(summary.to_string(index=False))
print(f"Saved table to: {summary_file}")

# Detecting the issues in the data for fixing
# Flag missing values in clinical variables.
for variable in data.columns:
    if variable == "patient_id":
        continue

    for patient_id in data.loc[data[variable].isna(), "patient_id"]:
        if variable.endswith("localization_result"):
            explanation = "Investigation result is missing rather than marked as Not performed."
        else:
            explanation = "Required value is missing."

        add_issue(
            patient_id,
            "Missing value",
            variable,
            "NA",
            explanation,
        )


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

print(f"Checked {len(data)} rows.")
print(f"Flagged {len(issue_table)} subject-and-issue combinations.")
print(f"Affected subjects: {issue_table['patient_id'].nunique()}")
print(f"Saved data-quality table to: {output_file}")
