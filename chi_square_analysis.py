"""
chi_square_analysis.py
Tests whether reported automation engagement is associated with crash
severity in a public automated-vehicle crash dataset.

Usage:    python chi_square_analysis.py path/to/dataset.csv
Requires: pandas, scipy  (pip install pandas scipy)
"""
import sys

import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

# Replace these placeholders with the documented field names of the dataset
AUTOMATION_COL = "automation_status"
SEVERITY_COL = "severity_category"
REPORT_ID_COL = "report_id"  # used to remove repeat reports of one crash


def load_and_clean(path):
    """Load the CSV, report data quality and return the analytic sample."""
    df = pd.read_csv(path)
    print(f"Raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Missing values (top 20 columns):")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    if REPORT_ID_COL in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=REPORT_ID_COL)
        print(f"Duplicate reports removed: {before - len(df)}")

    clean = df.dropna(subset=[AUTOMATION_COL, SEVERITY_COL]).copy()
    print(f"Analytic sample: N = {len(clean)}")
    return clean


def run_test(clean):
    """Run the chi-square test of independence and report Cramer's V."""
    table = pd.crosstab(clean[AUTOMATION_COL], clean[SEVERITY_COL])
    print("\nObserved frequencies:\n", table)

    chi2, p_value, dof, expected = chi2_contingency(table)
    expected = pd.DataFrame(expected, index=table.index, columns=table.columns)
    print("\nExpected frequencies:\n", expected.round(2))

    n = table.to_numpy().sum()
    cramers_v = (chi2 / (n * (min(table.shape) - 1))) ** 0.5
    print(f"\nchi2({dof}, N = {n}) = {chi2:.3f}, p = {p_value:.4g}, "
          f"Cramer's V = {cramers_v:.3f}")

    if (expected < 5).any().any():
        print("Warning: at least one expected count is below 5.")
        if table.shape == (2, 2):
            odds_ratio, p_exact = fisher_exact(table)
            print(f"Fisher's exact test: OR = {odds_ratio:.3f}, "
                  f"p = {p_exact:.4g}")
        else:
            print("Merge sparse categories on a documented basis and re-run.")


if __name__ == "__main__":
    data_path = (sys.argv[1] if len(sys.argv) > 1
                 else "verified_public_dataset.csv")
    run_test(load_and_clean(data_path))
