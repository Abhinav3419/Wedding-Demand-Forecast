"""
real_trends_loader.py — Load Real Google Trends Data
=====================================================
Loads Google Trends CSV exports for:
  1. "wedding lehenga" (India, 2004-present)
  2. "sherwani" (India, 2004-present)
  3. "bridal saree" (India, 2004-present)

Creates a composite wedding apparel demand index.
"""

import pandas as pd
import numpy as np
import os

# Project root = parent of src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")


def load_single_trends_csv(filepath, col_name):
    """Load a Google Trends CSV export, handling the header rows."""
    df = pd.read_csv(filepath, skiprows=2)
    old_col = df.columns[1]
    df = df.rename(columns={df.columns[0]: 'date', old_col: col_name})
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0)
    return df


def load_all_trends():
    """
    Load all three Google Trends files and create composite index.

    Returns DataFrame with columns:
      year, month, wedding_lehenga, sherwani, bridal_saree, demand_index
    """
    print("=" * 70)
    print("LOADING REAL GOOGLE TRENDS DATA")
    print("=" * 70)

    lehenga = load_single_trends_csv(
        os.path.join(DATA_RAW, "google_trends_wedding_lehenga.csv"),
        "wedding_lehenga"
    )
    sherwani = load_single_trends_csv(
        os.path.join(DATA_RAW, "google_trends_sherwani.csv"),
        "sherwani"
    )
    saree = load_single_trends_csv(
        os.path.join(DATA_RAW, "google_trends_bridal_saree.csv"),
        "bridal_saree"
    )

    print(f"  wedding lehenga: {len(lehenga)} months, "
          f"{lehenga['date'].min().strftime('%Y-%m')} to {lehenga['date'].max().strftime('%Y-%m')}")
    print(f"  sherwani:        {len(sherwani)} months, "
          f"{sherwani['date'].min().strftime('%Y-%m')} to {sherwani['date'].max().strftime('%Y-%m')}")
    print(f"  bridal saree:    {len(saree)} months, "
          f"{saree['date'].min().strftime('%Y-%m')} to {saree['date'].max().strftime('%Y-%m')}")

    merged = lehenga[['year', 'month', 'wedding_lehenga']].merge(
        sherwani[['year', 'month', 'sherwani']], on=['year', 'month']
    ).merge(
        saree[['year', 'month', 'bridal_saree']], on=['year', 'month']
    )

    merged['demand_index'] = (
        merged['wedding_lehenga'] + merged['sherwani'] + merged['bridal_saree']
    ) / 3.0
    merged['demand_index'] = merged['demand_index'].round(1)

    merged = merged[(merged['year'] >= 2010) & (merged['year'] <= 2025)]
    merged = merged.sort_values(['year', 'month']).reset_index(drop=True)

    print(f"\n  Composite index: {len(merged)} months "
          f"({merged['year'].min()}-{merged['month'].iloc[0]:02d} to "
          f"{merged['year'].max()}-{merged['month'].iloc[-1]:02d})")
    print(f"  Demand range: {merged['demand_index'].min():.1f} - {merged['demand_index'].max():.1f}")

    monthly_avg = merged.groupby('month')['demand_index'].mean()
    peak_month = monthly_avg.idxmax()
    trough_month = monthly_avg.idxmin()
    month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                   7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    print(f"  Peak month:   {month_names[peak_month]} (avg: {monthly_avg[peak_month]:.1f})")
    print(f"  Trough month: {month_names[trough_month]} (avg: {monthly_avg[trough_month]:.1f})")

    return merged


def build_master_with_real_trends():
    """
    Build master dataset using REAL Google Trends + Muhurat/Hijri/Economic data.
    """
    import sys
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
    from muhurat_data import (generate_muhurat_data, generate_hijri_data,
                               generate_gold_prices, generate_cpi_data)

    print("\n[1/5] Loading REAL Google Trends data...")
    trends = load_all_trends()

    print("\n[2/5] Generating Hindu Muhurat data...")
    muhurat = generate_muhurat_data()

    print("[3/5] Generating Hijri calendar data...")
    hijri = generate_hijri_data()

    print("[4/5] Generating gold price data...")
    gold = generate_gold_prices()

    print("[5/5] Generating CPI data...")
    cpi = generate_cpi_data()

    master = trends.merge(muhurat, on=['year', 'month'])
    master = master.merge(hijri, on=['year', 'month'])
    master = master.merge(gold, on=['year', 'month'])
    master = master.merge(cpi, on=['year', 'month'])

    master = master.sort_values(['year', 'month']).reset_index(drop=True)

    print(f"\nMaster dataset: {master.shape[0]} rows x {master.shape[1]} columns")
    return master


if __name__ == "__main__":
    master = build_master_with_real_trends()
    out_path = os.path.join(DATA_RAW, "master.csv")
    master.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print("\nSample (first 12 months):")
    print(master[['year','month','wedding_lehenga','sherwani','bridal_saree',
                   'demand_index','hindu_muhurat_count']].head(12).to_string(index=False))
