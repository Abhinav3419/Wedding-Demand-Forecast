"""
real_trends_loader.py — Load REAL Google Trends Data
=====================================================
Loads actual Google Trends CSV exports for:
  1. "wedding lehenga" (India, 2004-present)
  2. "sherwani" (India, 2004-present)
  3. "bridal saree" (India, 2004-present)

Creates a composite wedding apparel demand index by averaging
the three normalized search terms.
"""

import pandas as pd
import numpy as np


def load_single_trends_csv(filepath, col_name):
    """Load a Google Trends CSV export, handling the header rows."""
    # Google Trends CSVs have 2 header rows before data
    df = pd.read_csv(filepath, skiprows=2)
    
    # Rename columns
    old_col = df.columns[1]  # e.g., "wedding lehenga: (India)"
    df = df.rename(columns={df.columns[0]: 'date', old_col: col_name})
    
    # Parse date
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Handle '<1' values (Google Trends uses this for very low volume)
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
    
    # Load each file
    lehenga = load_single_trends_csv(
        "/mnt/user-data/uploads/multiTimeline.csv", 
        "wedding_lehenga"
    )
    sherwani = load_single_trends_csv(
        "/mnt/user-data/uploads/multiTimeline__1_.csv",
        "sherwani"
    )
    saree = load_single_trends_csv(
        "/mnt/user-data/uploads/multiTimeline__2_.csv",
        "bridal_saree"
    )
    
    print(f"  wedding lehenga: {len(lehenga)} months, "
          f"{lehenga['date'].min().strftime('%Y-%m')} to {lehenga['date'].max().strftime('%Y-%m')}")
    print(f"  sherwani:        {len(sherwani)} months, "
          f"{sherwani['date'].min().strftime('%Y-%m')} to {sherwani['date'].max().strftime('%Y-%m')}")
    print(f"  bridal saree:    {len(saree)} months, "
          f"{saree['date'].min().strftime('%Y-%m')} to {saree['date'].max().strftime('%Y-%m')}")
    
    # Merge on year + month
    merged = lehenga[['year', 'month', 'wedding_lehenga']].merge(
        sherwani[['year', 'month', 'sherwani']], on=['year', 'month']
    ).merge(
        saree[['year', 'month', 'bridal_saree']], on=['year', 'month']
    )
    
    # Create composite demand index (average of three terms)
    # This is more robust than any single search term
    merged['demand_index'] = (
        merged['wedding_lehenga'] + merged['sherwani'] + merged['bridal_saree']
    ) / 3.0
    merged['demand_index'] = merged['demand_index'].round(1)
    
    # Filter to 2010-2025 for our study period (muhurat data starts 2010)
    merged = merged[(merged['year'] >= 2010) & (merged['year'] <= 2025)]
    merged = merged.sort_values(['year', 'month']).reset_index(drop=True)
    
    print(f"\n  Composite index: {len(merged)} months "
          f"({merged['year'].min()}-{merged['month'].iloc[0]:02d} to "
          f"{merged['year'].max()}-{merged['month'].iloc[-1]:02d})")
    print(f"  Demand range: {merged['demand_index'].min():.1f} - {merged['demand_index'].max():.1f}")
    
    # Quick seasonal check
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
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
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
    
    # Merge everything
    master = trends.merge(muhurat, on=['year', 'month'])
    master = master.merge(hijri, on=['year', 'month'])
    master = master.merge(gold, on=['year', 'month'])
    master = master.merge(cpi, on=['year', 'month'])
    
    master = master.sort_values(['year', 'month']).reset_index(drop=True)
    
    print(f"\n✅ Master dataset: {master.shape[0]} rows × {master.shape[1]} columns")
    print(f"   Date range: {master['year'].min()}-{master['month'].iloc[0]:02d} to "
          f"{master['year'].max()}-{master['month'].iloc[-1]:02d}")
    
    return master


if __name__ == "__main__":
    master = build_master_with_real_trends()
    master.to_csv("/home/claude/wedding-demand-forecast/data/raw/master_real.csv", index=False)
    print(f"\n💾 Saved to data/raw/master_real.csv")
    print("\nSample (first 12 months):")
    print(master[['year','month','wedding_lehenga','sherwani','bridal_saree',
                   'demand_index','hindu_muhurat_count']].head(12).to_string(index=False))
