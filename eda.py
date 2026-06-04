# Exploratory Data Analysis module
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_helpers import ensure_file

# Ensure the cleaned dataset exists (create/copy if needed)
ensure_file("students.csv")

df = pd.read_csv("../data/students.csv")

# CGPA Distribution

sns.histplot(
    df["CGPA"],
    kde=True
)

plt.title("CGPA Distribution")
plt.show()

# Correlation Heatmap

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True
)

plt.title("Correlation Heatmap")
plt.show()

