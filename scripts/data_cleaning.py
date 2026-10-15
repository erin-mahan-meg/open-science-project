#!/usr/bin/env python3
"""Clean the simulated epilepsy-surgery dataset and save a new CSV file."""

from pathlib import Path
import pandas as pd


# This script is stored in project_folder/scripts/ and the CSV files are in data/.
project_dir = Path(__file__).resolve().parent.parent
raw_file = project_dir / "data" / "simulated_epilepsy_surgery_raw.csv"
clean_file = project_dir / "data" / "simulated_epilepsy_surgery_clean.csv"


# Import the raw data. The original file is never overwritten.
raw_data = pd.read_csv(raw_file)
clean_data = raw_data.copy()


# 1. Remove rows that are exact duplicates across every column.
rows_before = len(clean_data)
clean_data = clean_data.drop_duplicates().copy()
duplicates_removed = rows_before - len(clean_data)


# 2. Standardize the known inconsistent localization label.
localization_replacements = {"temporal": "Temporal"}
labels_corrected = clean_data["epilepsy_localization"].isin(
    localization_replacements
).sum()
clean_data["epilepsy_localization"] = clean_data[
    "epilepsy_localization"
].replace(localization_replacements)


# 3. Replace onset ages occurring after surgery with missing values.
invalid_onset = (
    clean_data["age_at_epilepsy_onset_years"]
    > clean_data["age_at_surgery_years"]
)
invalid_onsets_replaced = int(invalid_onset.sum())
clean_data.loc[invalid_onset, "age_at_epilepsy_onset_years"] = pd.NA


# 4. Recalculate epilepsy duration wherever both ages are available.
valid_ages = (
    clean_data["age_at_surgery_years"].notna()
    & clean_data["age_at_epilepsy_onset_years"].notna()
)
clean_data.loc[valid_ages, "epilepsy_duration_years"] = (
    clean_data.loc[valid_ages, "age_at_surgery_years"]
    - clean_data.loc[valid_ages, "age_at_epilepsy_onset_years"]
)

# Duration cannot be calculated when either age is missing.
clean_data.loc[~valid_ages, "epilepsy_duration_years"] = pd.NA


# 5. Replace seizure frequencies above the review threshold with missing values.
extreme_frequency = clean_data["monthly_seizure_frequency"] > 100
extreme_frequencies_replaced = int(extreme_frequency.sum())
clean_data.loc[extreme_frequency, "monthly_seizure_frequency"] = pd.NA


# 6. Validate the cleaned dataset before saving it.
assert clean_data["patient_id"].is_unique, "Duplicate patient IDs remain."
assert not (
    clean_data["age_at_epilepsy_onset_years"]
    > clean_data["age_at_surgery_years"]
).any(), "An onset age still occurs after surgery."
assert not (clean_data["epilepsy_duration_years"] < 0).any(), (
    "A negative epilepsy duration remains."
)
assert not (clean_data["monthly_seizure_frequency"] > 100).any(), (
    "An extreme seizure frequency remains."
)
assert "temporal" not in clean_data["epilepsy_localization"].values, (
    "The inconsistent temporal label remains."
)


# Save a separate cleaned dataset for tables, figures, and later analysis.
clean_data.to_csv(clean_file, index=False)


print(f"Raw rows: {len(raw_data)}")
print(f"Clean rows: {len(clean_data)}")
print(f"Exact duplicate rows removed: {duplicates_removed}")
print(f"Localization labels corrected: {labels_corrected}")
print(f"Invalid onset ages replaced with missing: {invalid_onsets_replaced}")
print(f"Extreme seizure frequencies replaced with missing: {extreme_frequencies_replaced}")
print(f"Saved cleaned data to: {clean_file}")