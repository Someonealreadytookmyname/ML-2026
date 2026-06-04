# Data cleaning utilities
import pandas as pd
from data_helpers import ensure_file

# Ensure the required dataset exists in the working directory
ensure_file("students_dataset.csv")

df = pd.read_csv("students_dataset.csv")

print("Original Dataset")
print(df.head())

# Missing values

df.fillna(df.mean(numeric_only=True), inplace=True)

# Duplicate removal

df.drop_duplicates(inplace=True)

print("\nClean Dataset")
print(df.head())

df.to_csv(
    "clean_students.csv",
    index=False
)

print("Dataset Saved")