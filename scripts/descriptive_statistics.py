#!/usr/bin/env python3
"""Create an Engel-stratified descriptive table from the cleaned dataset."""

from pathlib import Path

import pandas as pd


project_dir = Path(__file__).resolve().parent.parent
data_file = project_dir / "data" / "simulated_epilepsy_surgery_clean.csv"
output_dir = project_dir / "outputs"
output_dir.mkdir(exist_ok=True)

data = pd.read_csv(data_file)
outcome = "engel_outcome_1yr"
engel_classes = ["I", "II", "III", "IV"]

groups = {f"Engel {label}": data[data[outcome] == label] for label in engel_classes}
groups["Total"] = data
column_names = {
    name: f"{name}\n(n={len(group)})" for name, group in groups.items()
}


def mean_sd(values: pd.Series) -> str:
    values = values.dropna()
    if values.empty:
        return ""
    return f"{values.mean():.1f} ({values.std():.1f})"


def median_iqr(values: pd.Series) -> str:
    values = values.dropna()
    if values.empty:
        return ""
    return (
        f"{values.median():.1f} "
        f"[{values.quantile(0.25):.1f}, {values.quantile(0.75):.1f}]"
    )


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
cohort_row = {"Variable": "Cohort, n"}
for name, group in groups.items():
    cohort_row[column_names[name]] = str(len(group))
rows.append(cohort_row)

variables = [
    column for column in data.columns if column not in ["patient_id", outcome]
]

for variable in variables:
    label = variable.replace("_", " ")

    if pd.api.types.is_numeric_dtype(data[variable]):
        mean_row = {"Variable": f"{label}, mean (SD)"}
        for name, group in groups.items():
            mean_row[column_names[name]] = mean_sd(group[variable])
        rows.extend([mean_row])
    else:
        header_row = {"Variable": f"{label}, n (%)"}
        for name in groups:
            header_row[column_names[name]] = ""
        rows.append(header_row)

        levels = data[variable].dropna().drop_duplicates().tolist()
        for level in levels:
            level_row = {"Variable": f"    {level}"}
            for name, group in groups.items():
                level_row[column_names[name]] = count_percent(group[variable], level)
            rows.append(level_row)

    if data[variable].isna().any():
        missing_row = {"Variable": "    Missing"}
        for name, group in groups.items():
            missing_row[column_names[name]] = missing_count_percent(group[variable])
        rows.append(missing_row)


summary = pd.DataFrame(rows)
summary_file = output_dir / "descriptive_statistics_by_engel.csv"
summary.to_csv(summary_file, index=False)

print(summary.to_string(index=False))
print(f"Saved table to: {summary_file}")
