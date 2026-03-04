"""
utils.py — Helper Functions
============================
Utility functions used across the project.
"""

import numpy as np
import pandas as pd
from datetime import datetime


def create_date_range(start_year=2010, end_year=2025):
    """Create a year-month DataFrame."""
    records = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            records.append({'year': year, 'month': month})
    return pd.DataFrame(records)


def normalize_to_100(series):
    """Normalize a pandas Series to 0-100 scale."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50, index=series.index)
    return ((series - mn) / (mx - mn) * 100).round(1)


def safe_divide(a, b, fill=0):
    """Safe division avoiding divide-by-zero."""
    return np.where(b != 0, a / b, fill)


def print_section(title, char='=', width=70):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def format_pvalue(p):
    """Format p-value for display."""
    if p < 0.001:
        return "< 0.001 ***"
    elif p < 0.01:
        return f"{p:.3f} **"
    elif p < 0.05:
        return f"{p:.3f} *"
    elif p < 0.10:
        return f"{p:.3f} †"
    else:
        return f"{p:.3f}"


def correlation_matrix(df, columns, method='pearson'):
    """Compute and format a correlation matrix."""
    return df[columns].corr(method=method).round(4)


def describe_feature(df, col):
    """Print descriptive statistics for a feature."""
    s = df[col]
    print(f"  {col}:")
    print(f"    Mean: {s.mean():.2f} | Std: {s.std():.2f}")
    print(f"    Min: {s.min():.2f} | Max: {s.max():.2f}")
    print(f"    Skew: {s.skew():.2f} | Kurt: {s.kurtosis():.2f}")


if __name__ == "__main__":
    print("Utils module loaded successfully.")
    print(f"Available functions: create_date_range, normalize_to_100,")
    print(f"  safe_divide, print_section, format_pvalue, correlation_matrix")
