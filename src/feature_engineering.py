"""
feature_engineering.py — Cultural Calendar Feature Engineering Pipeline
========================================================================
This is the CORE NOVEL CONTRIBUTION of the project.

Takes raw master data and engineers three layers of features:
  Layer 1: Temporal baseline (what any standard model would use)
  Layer 2: Cultural calendar features (OUR INNOVATION)
  Layer 3: Economic context features

Also implements lead-lag analysis to determine optimal shopping lead time
before muhurat dates.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    Input: raw master dataset with columns from muhurat_data.py
    Output: fully featured dataset ready for modeling
    """
    df = df.copy()
    df = df.sort_values(['year', 'month']).reset_index(drop=True)
    
    # Create date column for convenience
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    
    print("=" * 70)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 70)
    
    # ================================================================
    # LAYER 1: TEMPORAL BASELINE FEATURES
    # ================================================================
    print("\n[LAYER 1] Temporal Baseline Features...")
    
    # Quarter
    df['quarter'] = df['month'].map(lambda m: (m - 1) // 3 + 1)
    
    # Cyclical encoding of month (captures circular nature: Dec is close to Jan)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Diwali proximity (Diwali falls in Oct-Nov each year)
    # Approximate: Diwali is typically in late October or early November
    diwali_months = {
        2010: 11, 2011: 10, 2012: 11, 2013: 11, 2014: 10,
        2015: 11, 2016: 10, 2017: 10, 2018: 11, 2019: 10,
        2020: 11, 2021: 11, 2022: 10, 2023: 11, 2024: 11, 2025: 10
    }
    df['diwali_month'] = df['year'].map(diwali_months)
    df['months_to_diwali'] = (df['diwali_month'] - df['month']).apply(
        lambda x: min(abs(x), 12 - abs(x))  # Circular distance
    )
    df['is_diwali_month'] = (df['month'] == df['diwali_month']).astype(int)
    df.drop('diwali_month', axis=1, inplace=True)
    
    # Lag features (demand history)
    df['demand_lag_1m'] = df['demand_index'].shift(1)
    df['demand_lag_2m'] = df['demand_index'].shift(2)
    df['demand_lag_3m'] = df['demand_index'].shift(3)
    df['demand_lag_12m'] = df['demand_index'].shift(12)  # Same month last year
    
    # Rolling statistics
    df['demand_rolling_mean_3m'] = df['demand_index'].rolling(3, min_periods=1).mean()
    df['demand_rolling_std_3m'] = df['demand_index'].rolling(3, min_periods=1).std().fillna(0)
    df['demand_rolling_mean_6m'] = df['demand_index'].rolling(6, min_periods=1).mean()
    
    # Year-over-year change
    df['demand_yoy_change'] = df['demand_index'] - df['demand_lag_12m']
    
    # Trend component (linear year progression normalized)
    df['year_trend'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
    
    n_layer1 = 14  # approximate count
    print(f"      → Added ~{n_layer1} temporal features")
    
    # ================================================================
    # LAYER 2: CULTURAL CALENDAR FEATURES (NOVEL CONTRIBUTION)
    # ================================================================
    print("\n[LAYER 2] Cultural Calendar Features (NOVEL)...")
    
    # --- Hindu Muhurat Features ---
    
    # Raw muhurat count (already in data)
    # Year-over-year change in muhurat count for same month
    df['muhurat_yoy_change'] = df.groupby('month')['hindu_muhurat_count'].diff()
    
    # Muhurat density vs historical average for that month
    month_avg_muhurat = df.groupby('month')['hindu_muhurat_count'].transform('mean')
    df['muhurat_density_vs_avg'] = df['hindu_muhurat_count'] / month_avg_muhurat.replace(0, 1)
    
    # Muhurat concentration: is this month in top-3 muhurat months of its year?
    year_rank = df.groupby('year')['hindu_muhurat_count'].rank(ascending=False, method='min')
    df['is_top3_muhurat_month'] = (year_rank <= 3).astype(int)
    
    # Cumulative muhurat days in the quarter
    df['quarter_muhurat_cumsum'] = df.groupby(['year', 'quarter'])['hindu_muhurat_count'].cumsum()
    
    # Rolling 3-month muhurat average (smoothed cultural signal)
    df['muhurat_rolling_3m'] = df['hindu_muhurat_count'].rolling(3, min_periods=1).mean()
    
    # Lead features: muhurat count in NEXT month and NEXT 2 months
    # (people shop BEFORE the wedding, so future muhurats drive current demand)
    df['muhurat_lead_1m'] = df['hindu_muhurat_count'].shift(-1)
    df['muhurat_lead_2m'] = df['hindu_muhurat_count'].shift(-2)
    df['muhurat_next_3m_total'] = (
        df['hindu_muhurat_count'].shift(-1).fillna(0) +
        df['hindu_muhurat_count'].shift(-2).fillna(0) +
        df['hindu_muhurat_count'].shift(-3).fillna(0)
    )
    
    # --- Hijri Calendar Features ---
    
    # Inverse of restricted days (more available days = more weddings)
    days_in_month_map = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30,
                         7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
    df['days_in_month'] = df['month'].map(days_in_month_map)
    df['hijri_available_days'] = df['days_in_month'] - df['hijri_total_restricted_days']
    df['hijri_availability_ratio'] = df['hijri_available_days'] / df['days_in_month']
    
    # --- Combined Cultural Score ---
    # This is the key composite feature
    df['combined_favorable_score'] = (
        df['hindu_muhurat_count'] * df['hijri_availability_ratio']
    )
    
    # Favorable score vs 12-month rolling average
    df['favorable_score_vs_rolling'] = (
        df['combined_favorable_score'] / 
        df['combined_favorable_score'].rolling(12, min_periods=1).mean().replace(0, 1)
    )
    
    n_layer2 = 14
    print(f"      → Added ~{n_layer2} cultural calendar features")
    print("      → Key features: muhurat_count, muhurat_density_vs_avg,")
    print("        combined_favorable_score, muhurat_lead_1m/2m, muhurat_next_3m_total")
    
    # ================================================================
    # LAYER 3: ECONOMIC CONTEXT FEATURES
    # ================================================================
    print("\n[LAYER 3] Economic Context Features...")
    
    # Gold price momentum (rising gold = more wedding spending sentiment)
    df['gold_price_mom_3m'] = df['gold_price_inr_10g'].pct_change(3)
    df['gold_price_mom_12m'] = df['gold_price_inr_10g'].pct_change(12)
    
    # Gold price normalized (min-max within dataset)
    gold_min, gold_max = df['gold_price_inr_10g'].min(), df['gold_price_inr_10g'].max()
    df['gold_price_normalized'] = (df['gold_price_inr_10g'] - gold_min) / (gold_max - gold_min)
    
    # CPI change (inflation rate proxy)
    df['cpi_yoy_change'] = df['cpi_index'].pct_change(12)
    
    # Interaction: high muhurat count × gold momentum
    # (many wedding dates + rising gold = very strong demand signal)
    df['muhurat_x_gold_mom'] = df['hindu_muhurat_count'] * df['gold_price_mom_3m'].fillna(0)
    
    n_layer3 = 5
    print(f"      → Added ~{n_layer3} economic features")
    
    # ================================================================
    # CLEANUP
    # ================================================================
    
    # Drop rows with NaN from lag/shift operations (first 12 months)
    initial_rows = len(df)
    # We keep NaNs for now and handle in modeling (to preserve data)
    # But mark which rows are usable for training
    df['is_trainable'] = (~df['demand_lag_12m'].isna()).astype(int)
    
    print(f"\n{'='*70}")
    print(f"FEATURE ENGINEERING COMPLETE")
    print(f"{'='*70}")
    print(f"Total rows: {len(df)}")
    print(f"Trainable rows (with all lags): {df['is_trainable'].sum()}")
    print(f"Total features: {len(df.columns)}")
    print(f"Feature list:")
    for i, col in enumerate(df.columns):
        print(f"   {i+1:2d}. {col}")
    
    return df


def get_feature_sets(df: pd.DataFrame) -> dict:
    """
    Return the three feature sets for model comparison.
    
    This is CRITICAL for the research methodology:
    - Baseline uses ONLY temporal features
    - Enhanced adds cultural calendar features
    - Full adds everything
    
    If Enhanced > Baseline with statistical significance,
    the hypothesis is supported.
    """
    
    layer1_features = [
        'month_sin', 'month_cos', 'quarter', 'year_trend',
        'months_to_diwali', 'is_diwali_month',
        'demand_lag_1m', 'demand_lag_2m', 'demand_lag_3m', 'demand_lag_12m',
        'demand_rolling_mean_3m', 'demand_rolling_std_3m',
        'demand_rolling_mean_6m', 'demand_yoy_change',
    ]
    
    layer2_features = [
        'hindu_muhurat_count', 'muhurat_yoy_change', 'muhurat_density_vs_avg',
        'is_top3_muhurat_month', 'quarter_muhurat_cumsum', 'muhurat_rolling_3m',
        'muhurat_lead_1m', 'muhurat_lead_2m', 'muhurat_next_3m_total',
        'is_pitru_paksha_month', 'is_kharmas_month', 'is_mal_maas_year',
        'hijri_total_restricted_days', 'hijri_availability_ratio',
        'combined_favorable_score', 'favorable_score_vs_rolling',
    ]
    
    layer3_features = [
        'gold_price_normalized', 'gold_price_mom_3m', 'gold_price_mom_12m',
        'cpi_yoy_change', 'muhurat_x_gold_mom',
    ]
    
    return {
        'baseline': layer1_features,
        'enhanced': layer1_features + layer2_features,
        'full': layer1_features + layer2_features + layer3_features,
    }


def prepare_train_data(df: pd.DataFrame, feature_cols: list
                       ) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Prepare clean training data by dropping NaN rows.
    Returns X, y, and the indices of valid rows.
    """
    valid_mask = df['is_trainable'] == 1
    df_valid = df[valid_mask].copy()
    
    # Fill any remaining NaN in features with 0
    X = df_valid[feature_cols].fillna(0).values
    y = df_valid['demand_index'].values
    valid_indices = df_valid.index.tolist()
    
    return X, y, valid_indices


if __name__ == "__main__":
    # Load raw data
    raw = pd.read_csv("/home/claude/wedding-demand-forecast/data/raw/master_raw.csv")
    
    # Engineer features
    featured = engineer_features(raw)
    
    # Save processed data
    featured.to_csv("/home/claude/wedding-demand-forecast/data/processed/features_master.csv", index=False)
    print(f"\n💾 Saved to data/processed/features_master.csv")
    
    # Show feature sets
    fsets = get_feature_sets(featured)
    print(f"\n📊 Feature Set Sizes:")
    print(f"   Baseline:  {len(fsets['baseline'])} features")
    print(f"   Enhanced:  {len(fsets['enhanced'])} features (+ {len(fsets['enhanced']) - len(fsets['baseline'])} cultural)")
    print(f"   Full:      {len(fsets['full'])} features (+ {len(fsets['full']) - len(fsets['enhanced'])} economic)")
    
    # Quick correlation check
    trainable = featured[featured['is_trainable'] == 1]
    corr_muhurat = trainable['hindu_muhurat_count'].corr(trainable['demand_index'])
    corr_favorable = trainable['combined_favorable_score'].corr(trainable['demand_index'])
    corr_lag12 = trainable['demand_lag_12m'].corr(trainable['demand_index'])
    
    print(f"\n🔗 Key Correlations with Demand:")
    print(f"   hindu_muhurat_count:      r = {corr_muhurat:.4f}")
    print(f"   combined_favorable_score: r = {corr_favorable:.4f}")
    print(f"   demand_lag_12m:           r = {corr_lag12:.4f}")
