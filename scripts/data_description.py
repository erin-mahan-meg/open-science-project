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