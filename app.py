"""
app.py — Streamlit Interactive Dashboard
==========================================
The Missing Variable: Wedding Apparel Demand Forecasting
using Hindu Muhurat & Islamic Calendar Features

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import sys, os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Page config
st.set_page_config(
    page_title="The Missing Variable — Wedding Demand Forecasting",
    page_icon="🕉️",
    layout="wide"
)

st.title("🕉️ ☪️ The Missing Variable")
st.markdown("### How Religious Calendar Systems Predict Retail Demand in India")
st.markdown("---")


@st.cache_data
def load_data():
    """Load and prepare all data."""
    from muhurat_data import build_master_dataset
    from feature_engineering import engineer_features, get_feature_sets
    
    raw = build_master_dataset()
    featured = engineer_features(raw)
    return raw, featured


@st.cache_data
def run_model(featured_df, feature_cols, alpha=1.0):
    """Train Ridge regression and return results."""
    trainable = featured_df[featured_df['is_trainable'] == 1].copy()
    years = sorted([y for y in trainable['year'].unique() if y >= 2012])
    
    all_preds = []
    fold_rmses = []
    
    for test_year in years:
        tr = trainable[trainable['year'] != test_year]
        te = trainable[trainable['year'] == test_year]
        
        sc = StandardScaler()
        Xtr = sc.fit_transform(tr[feature_cols].fillna(0))
        Xte = sc.transform(te[feature_cols].fillna(0))
        
        model = Ridge(alpha=alpha)
        model.fit(Xtr, tr['demand_index'].values)
        yp = np.clip(model.predict(Xte), 0, 100)
        yt = te['demand_index'].values
        
        fold_rmses.append(np.sqrt(mean_squared_error(yt, yp)))
        
        for i, (_, row) in enumerate(te.iterrows()):
            all_preds.append({
                'year': row['year'], 'month': row['month'],
                'actual': yt[i], 'predicted': yp[i]
            })
    
    # Train final model on all data for coefficients
    X_all = trainable[feature_cols].fillna(0).values
    sc_all = StandardScaler()
    Xs_all = sc_all.fit_transform(X_all)
    final_model = Ridge(alpha=alpha)
    final_model.fit(Xs_all, trainable['demand_index'].values)
    
    coefs = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': final_model.coef_,
        'abs_coef': np.abs(final_model.coef_)
    }).sort_values('abs_coef', ascending=False)
    
    return pd.DataFrame(all_preds), np.array(fold_rmses), coefs


# Sidebar
st.sidebar.header("⚙️ Model Controls")
alpha = st.sidebar.slider("Ridge Regularization (α)", 0.1, 50.0, 1.0, 0.5)
show_cultural = st.sidebar.checkbox("Include Cultural Calendar Features", True)
show_economic = st.sidebar.checkbox("Include Economic Features", False)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown(
    "This dashboard demonstrates that Hindu Muhurat dates and "
    "Islamic Hijri calendar periods are statistically significant "
    "predictors of wedding apparel demand in India."
)

# Load data
try:
    raw, featured = load_data()
except Exception:
    st.error("Run `python src/muhurat_data.py` first to generate data.")
    st.stop()

# Define feature sets
base_features = [
    'month_sin', 'month_cos', 'quarter', 'year_trend',
    'months_to_diwali', 'is_diwali_month',
]
cultural_features = [
    'hindu_muhurat_count', 'muhurat_density_vs_avg',
    'is_top3_muhurat_month', 'muhurat_rolling_3m',
    'is_pitru_paksha_month', 'is_kharmas_month', 'is_mal_maas_year',
    'hijri_total_restricted_days', 'hijri_availability_ratio',
    'combined_favorable_score',
]
economic_features = [
    'gold_price_normalized', 'gold_price_mom_3m', 'cpi_yoy_change',
]

active_features = base_features.copy()
if show_cultural:
    active_features += cultural_features
if show_economic:
    active_features += economic_features

# Run model
preds_df, fold_rmses, coefs_df = run_model(featured, active_features, alpha)

# ====== MAIN DASHBOARD ======

# Row 1: Key Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Mean RMSE", f"{np.mean(fold_rmses):.2f}")
with col2:
    st.metric("Std RMSE", f"{np.std(fold_rmses):.2f}")
with col3:
    r2 = r2_score(preds_df['actual'], preds_df['predicted'])
    st.metric("R² Score", f"{r2:.3f}")
with col4:
    st.metric("Features Used", len(active_features))

st.markdown("---")

# Row 2: Time Series
st.subheader("📈 Demand vs. Muhurat Overlay")

fig, ax1 = plt.subplots(figsize=(14, 5))
dates = pd.to_datetime(raw[['year', 'month']].assign(day=15))
ax1.plot(dates, raw['demand_index'], color='#0D3B66', linewidth=1.5, label='Demand Index')
ax1.set_ylabel('Demand Index', color='#0D3B66')

ax2 = ax1.twinx()
ax2.bar(dates, raw['hindu_muhurat_count'], width=25, alpha=0.4,
        color='#F4A261', label='Muhurat Days')
ax2.set_ylabel('Muhurat Days/Month', color='#F4A261')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
ax1.set_title('Wedding Apparel Demand Follows Muhurat Date Density')
st.pyplot(fig)
plt.close()

# Row 3: Predictions + Feature Importance
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🎯 Actual vs Predicted (LOYO-CV)")
    preds_df['date'] = pd.to_datetime(preds_df[['year', 'month']].assign(day=15))
    preds_df = preds_df.sort_values('date')
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(preds_df['date'], preds_df['actual'], label='Actual', color='#0D3B66')
    ax.plot(preds_df['date'], preds_df['predicted'], label='Predicted',
            color='#F4A261', linestyle='--')
    ax.legend()
    ax.set_title(f'RMSE: {np.mean(fold_rmses):.2f}')
    st.pyplot(fig)
    plt.close()

with col_right:
    st.subheader("🏆 Top 10 Feature Importance")
    top10 = coefs_df.head(10)
    
    cultural_kw = ['muhurat', 'favorable', 'hijri', 'pitru', 'kharmas', 'mal_maas']
    colors = ['#F4A261' if any(kw in f for kw in cultural_kw) else '#8D99AE'
              for f in top10['feature']]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(top10)), top10['abs_coef'].values, color=colors)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('|Coefficient|')
    ax.set_title('Orange = Cultural Feature ★')
    st.pyplot(fig)
    plt.close()

# Row 4: Heatmap
st.subheader("🗓️ Muhurat Density Heatmap (Year × Month)")
pivot = raw.pivot_table(values='hindu_muhurat_count', index='year', columns='month')
month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd',
            xticklabels=month_labels, linewidths=0.5, ax=ax)
ax.set_title('Shubh Vivah Muhurat Days per Month (Dark = Peak Wedding Season)')
st.pyplot(fig)
plt.close()

# Footer
st.markdown("---")
st.markdown(
    "**The Missing Variable** | "
    "Hindu Muhurat × Hijri Calendar → Apparel Demand Forecasting | "
    "Built with Ridge Regression + LOYO Cross-Validation"
)
