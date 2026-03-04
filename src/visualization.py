"""
visualization.py — Publication-Quality Visualizations
======================================================
Generates all figures for the paper/portfolio.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import engineer_features

# Style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

COLORS = {
    'primary': '#0D3B66',
    'accent': '#F4A261',
    'green': '#2A9D8F',
    'red': '#E76F51',
    'purple': '#6A4C93',
    'light': '#EDF6FC',
    'gray': '#8D99AE',
}


def plot_demand_vs_muhurat(df, save_path):
    """Figure 1: Demand index overlaid with muhurat count — the key visual."""
    fig, ax1 = plt.subplots(figsize=(16, 6))
    
    dates = pd.to_datetime(df[['year', 'month']].assign(day=15))
    
    # Demand line
    ax1.plot(dates, df['demand_index'], color=COLORS['primary'], linewidth=1.8, 
             label='Apparel Demand Index', zorder=3)
    ax1.fill_between(dates, 0, df['demand_index'], alpha=0.08, color=COLORS['primary'])
    ax1.set_ylabel('Demand Index (Google Trends Scale)', color=COLORS['primary'], fontsize=12)
    ax1.tick_params(axis='y', labelcolor=COLORS['primary'])
    ax1.set_ylim(0, 110)
    
    # Muhurat bars on secondary axis
    ax2 = ax1.twinx()
    ax2.bar(dates, df['hindu_muhurat_count'], width=25, alpha=0.45, 
            color=COLORS['accent'], label='Hindu Muhurat Days', zorder=2)
    ax2.set_ylabel('Shubh Vivah Muhurat Days per Month', color=COLORS['accent'], fontsize=12)
    ax2.tick_params(axis='y', labelcolor=COLORS['accent'])
    ax2.set_ylim(0, 20)
    
    # COVID annotation
    covid_start = pd.Timestamp('2020-04-01')
    covid_end = pd.Timestamp('2020-09-01')
    ax1.axvspan(covid_start, covid_end, alpha=0.15, color=COLORS['red'], zorder=1)
    ax1.annotate('COVID\nLockdown', xy=(pd.Timestamp('2020-06-15'), 8), fontsize=9,
                color=COLORS['red'], ha='center', fontweight='bold')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    ax1.set_title('The Missing Variable: Wedding Apparel Demand Follows Muhurat Date Density',
                  fontsize=14, pad=15)
    ax1.set_xlabel('Date', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {save_path}")


def plot_correlation_scatter(df, save_path):
    """Figure 2: Scatter plot of muhurat count vs demand with regression line."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Muhurat count vs demand
    ax = axes[0]
    trainable = df[df['is_trainable'] == 1] if 'is_trainable' in df.columns else df
    x = trainable['hindu_muhurat_count']
    y = trainable['demand_index']
    
    ax.scatter(x, y, c=COLORS['primary'], alpha=0.5, s=50, edgecolors='white', linewidth=0.5)
    
    # Regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), color=COLORS['red'], linewidth=2, linestyle='--',
            label=f'r = {x.corr(y):.3f}')
    
    ax.set_xlabel('Hindu Muhurat Days per Month', fontsize=12)
    ax.set_ylabel('Demand Index', fontsize=12)
    ax.set_title('Muhurat Count vs. Apparel Demand', fontsize=13)
    ax.legend(fontsize=12, loc='upper left')
    
    # Right: Combined favorable score vs demand
    ax = axes[1]
    x2 = trainable['combined_favorable_score']
    ax.scatter(x2, y, c=COLORS['green'], alpha=0.5, s=50, edgecolors='white', linewidth=0.5)
    
    z2 = np.polyfit(x2, y, 1)
    p2 = np.poly1d(z2)
    x2_line = np.linspace(x2.min(), x2.max(), 100)
    ax.plot(x2_line, p2(x2_line), color=COLORS['red'], linewidth=2, linestyle='--',
            label=f'r = {x2.corr(y):.3f}')
    
    ax.set_xlabel('Combined Favorable Score (Hindu + Hijri)', fontsize=12)
    ax.set_ylabel('Demand Index', fontsize=12)
    ax.set_title('Combined Cultural Score vs. Apparel Demand', fontsize=13)
    ax.legend(fontsize=12, loc='upper left')
    
    plt.suptitle('Cultural Calendar Features Show Strong Linear Relationship with Demand',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {save_path}")


def plot_feature_importance(save_path):
    """Figure 3: Feature importance bar chart highlighting cultural features."""
    imp_df = pd.read_csv("/home/claude/wedding-demand-forecast/results/feature_importance_v2.csv")
    imp_df = imp_df.head(16)  # Top 16
    
    cultural_kw = ['muhurat', 'favorable', 'hijri', 'pitru', 'kharmas', 'mal_maas']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(imp_df))
    colors = [COLORS['accent'] if any(kw in f for kw in cultural_kw) else COLORS['gray'] 
              for f in imp_df['feature']]
    
    bars = ax.barh(y_pos, imp_df['avg_pct'], color=colors, edgecolor='white', linewidth=0.5, height=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(imp_df['feature'], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Average Importance (%)', fontsize=12)
    ax.set_title('Feature Importance: Cultural Calendar Features (★) Explain 49% of Signal',
                 fontsize=13, pad=15)
    
    # Add percentage labels
    for i, (bar, val) in enumerate(zip(bars, imp_df['avg_pct'])):
        is_cult = any(kw in imp_df.iloc[i]['feature'] for kw in cultural_kw)
        label = f'{val:.1f}% ★' if is_cult else f'{val:.1f}%'
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=9, 
                fontweight='bold' if is_cult else 'normal',
                color=COLORS['accent'] if is_cult else COLORS['gray'])
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['accent'], label='Cultural Calendar Feature (Novel)'),
        Patch(facecolor=COLORS['gray'], label='Standard Temporal Feature'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {save_path}")


def plot_model_comparison(save_path):
    """Figure 4: Bar chart comparing Baseline vs Enhanced RMSE."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Track A
    ax = axes[0]
    models = ['Ridge\nBaseline', 'Ridge\nEnhanced', 'Ridge\nFull']
    rmses = [11.38, 7.54, 7.71]
    colors = [COLORS['gray'], COLORS['green'], COLORS['purple']]
    
    bars = ax.bar(models, rmses, color=colors, edgecolor='white', width=0.6)
    ax.set_ylabel('RMSE (lower = better)', fontsize=12)
    ax.set_title('Track A: No-Lag (Cold-Start)\n+33.8% improvement ✅ p < 0.001', fontsize=12)
    
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.2f}', ha='center', fontsize=11, fontweight='bold')
    
    # Improvement arrow
    ax.annotate('', xy=(1, 7.54), xytext=(0, 11.38),
                arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=2.5))
    ax.text(0.5, 9.5, '-33.8%', ha='center', fontsize=14, fontweight='bold',
            color=COLORS['green'])
    
    ax.set_ylim(0, 14)
    
    # Track B
    ax = axes[1]
    models = ['Ridge\nBaseline+Lag', 'Ridge\nEnhanced+Lag']
    rmses = [5.97, 5.46]
    colors = [COLORS['gray'], COLORS['green']]
    
    bars = ax.bar(models, rmses, color=colors, edgecolor='white', width=0.5)
    ax.set_ylabel('RMSE (lower = better)', fontsize=12)
    ax.set_title('Track B: With-Lag (Practical)\n+8.5% improvement (p = 0.14)', fontsize=12)
    
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_ylim(0, 8)
    
    plt.suptitle('Model Comparison: Cultural Features Add Significant Value',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {save_path}")


def plot_seasonal_muhurat_heatmap(df, save_path):
    """Figure 5: Year × Month heatmap of muhurat counts."""
    pivot = df.pivot_table(values='hindu_muhurat_count', index='year', columns='month')
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd',
                xticklabels=month_labels, yticklabels=True,
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Shubh Vivah Muhurat Days'},
                ax=ax)
    
    ax.set_title('Hindu Shubh Vivah Muhurat Density: Year × Month Heatmap\n'
                 '(Dark = Peak Wedding Season, Light = Restricted Periods)',
                 fontsize=13, pad=15)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Year', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {save_path}")


def plot_predictions_vs_actual(save_path):
    """Figure 6: Time series of actual vs predicted for best model."""
    pred_df = pd.read_csv("/home/claude/wedding-demand-forecast/results/track_a_predictions.csv")
    
    pred_df['date'] = pd.to_datetime(pred_df[['year', 'month']].assign(day=15))
    pred_df = pred_df.sort_values('date')
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Top: Actual vs Predicted
    ax = axes[0]
    ax.plot(pred_df['date'], pred_df['actual'], color=COLORS['primary'], linewidth=1.5,
            label='Actual Demand', marker='o', markersize=3)
    ax.plot(pred_df['date'], pred_df['predicted'], color=COLORS['accent'], linewidth=1.5,
            label='Predicted (Enhanced Model)', marker='s', markersize=3, linestyle='--')
    ax.fill_between(pred_df['date'], pred_df['actual'], pred_df['predicted'],
                    alpha=0.15, color=COLORS['red'])
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylabel('Demand Index', fontsize=12)
    ax.set_title('Actual vs. Predicted Demand (Leave-One-Year-Out CV)',
                 fontsize=13, pad=10)
    
    # Bottom: Residuals
    ax = axes[1]
    residuals = pred_df['predicted'] - pred_df['actual']
    ax.bar(pred_df['date'], residuals, width=25, 
           color=[COLORS['green'] if r >= 0 else COLORS['red'] for r in residuals],
           alpha=0.6)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('Error (Pred - Actual)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title('Prediction Residuals', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {save_path}")


def generate_all_figures():
    """Generate all publication figures."""
    print("=" * 70)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 70)
    
    raw = pd.read_csv("/home/claude/wedding-demand-forecast/data/raw/master_raw.csv")
    df = engineer_features(raw)
    
    base = "/home/claude/wedding-demand-forecast/results"
    
    print()
    plot_demand_vs_muhurat(raw, f"{base}/fig1_demand_vs_muhurat.png")
    plot_correlation_scatter(df, f"{base}/fig2_correlation_scatter.png")
    plot_feature_importance(f"{base}/fig3_feature_importance.png")
    plot_model_comparison(f"{base}/fig4_model_comparison.png")
    plot_seasonal_muhurat_heatmap(raw, f"{base}/fig5_muhurat_heatmap.png")
    plot_predictions_vs_actual(f"{base}/fig6_predictions_vs_actual.png")
    
    print(f"\n✅ All 6 figures generated in results/")


if __name__ == "__main__":
    generate_all_figures()
