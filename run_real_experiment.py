"""
run_real_experiment.py — Complete Pipeline with REAL Google Trends Data
========================================================================
This is the FINAL production script that:
1. Loads real Google Trends data (wedding lehenga + sherwani + bridal saree)
2. Merges with muhurat, hijri, gold, CPI features
3. Engineers all three feature layers
4. Runs two-track LOYO cross-validation
5. Performs statistical significance tests
6. Generates all publication figures
7. Saves all results
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
from real_trends_loader import build_master_with_real_trends
from feature_engineering import engineer_features

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
COLORS = {'primary':'#0D3B66','accent':'#F4A261','green':'#2A9D8F',
          'red':'#E76F51','purple':'#6A4C93','gray':'#8D99AE'}

plt.rcParams.update({'font.size':11,'axes.titleweight':'bold',
                     'figure.facecolor':'white','axes.facecolor':'#FAFAFA',
                     'axes.grid':True,'grid.alpha':0.3,'grid.linestyle':'--'})


# ============================================================
# FEATURE SETS
# ============================================================
def get_feature_sets():
    track_a = {
        'baseline_no_lag': [
            'month_sin','month_cos','quarter','year_trend',
            'months_to_diwali','is_diwali_month',
        ],
        'enhanced_no_lag': [
            'month_sin','month_cos','quarter','year_trend',
            'months_to_diwali','is_diwali_month',
            'hindu_muhurat_count','muhurat_density_vs_avg',
            'is_top3_muhurat_month','muhurat_rolling_3m',
            'is_pitru_paksha_month','is_kharmas_month','is_mal_maas_year',
            'hijri_total_restricted_days','hijri_availability_ratio',
            'combined_favorable_score',
        ],
        'full_no_lag': [
            'month_sin','month_cos','quarter','year_trend',
            'months_to_diwali','is_diwali_month',
            'hindu_muhurat_count','muhurat_density_vs_avg',
            'is_top3_muhurat_month','muhurat_rolling_3m',
            'is_pitru_paksha_month','is_kharmas_month','is_mal_maas_year',
            'hijri_total_restricted_days','hijri_availability_ratio',
            'combined_favorable_score',
            'gold_price_normalized','gold_price_mom_3m','cpi_yoy_change',
        ],
    }
    track_b = {
        'baseline_with_lag': [
            'month_sin','month_cos','quarter','year_trend',
            'months_to_diwali','is_diwali_month',
            'demand_lag_1m','demand_lag_12m','demand_rolling_mean_3m',
        ],
        'enhanced_with_lag': [
            'month_sin','month_cos','quarter','year_trend',
            'months_to_diwali','is_diwali_month',
            'demand_lag_1m','demand_lag_12m','demand_rolling_mean_3m',
            'hindu_muhurat_count','muhurat_density_vs_avg',
            'combined_favorable_score','muhurat_yoy_change',
            'is_pitru_paksha_month','is_kharmas_month',
        ],
    }
    return track_a, track_b


# ============================================================
# CROSS-VALIDATION & SIGNIFICANCE TESTING
# ============================================================
def loyo_cv(df, features, model_cls, model_params):
    trainable = df[df['is_trainable']==1]
    cv_years = sorted([y for y in trainable['year'].unique() if y >= 2012])
    fold_rmses,fold_maes,fold_r2s = [],[],[]
    all_preds = []
    
    for ty in cv_years:
        tr = trainable[trainable['year']!=ty]
        te = trainable[trainable['year']==ty]
        sc = StandardScaler()
        Xtr = sc.fit_transform(tr[features].fillna(0))
        Xte = sc.transform(te[features].fillna(0))
        ytr,yte = tr['demand_index'].values, te['demand_index'].values
        m = model_cls(**model_params); m.fit(Xtr,ytr)
        yp = np.clip(m.predict(Xte),0,100)
        fold_rmses.append(np.sqrt(mean_squared_error(yte,yp)))
        fold_maes.append(mean_absolute_error(yte,yp))
        fold_r2s.append(r2_score(yte,yp))
        for i,(_, row) in enumerate(te.iterrows()):
            all_preds.append({'year':row['year'],'month':row['month'],
                              'actual':yte[i],'predicted':yp[i]})
    
    return {'rmses':np.array(fold_rmses),'mean_rmse':np.mean(fold_rmses),
            'std_rmse':np.std(fold_rmses),'mean_mae':np.mean(fold_maes),
            'mean_r2':np.mean(fold_r2s),'predictions':pd.DataFrame(all_preds)}


def paired_test(ra,rb,na,nb):
    t,p2 = stats.ttest_rel(ra,rb)
    p1 = p2/2 if t>0 else 1-p2/2
    imp = (np.mean(ra)-np.mean(rb))/np.mean(ra)*100
    return {'comparison':f"{nb} vs {na}",'rmse_a':np.mean(ra),'rmse_b':np.mean(rb),
            'improvement_pct':imp,'t_stat':t,'p_one_sided':p1,
            'significant':p1<0.05,'marginal':p1<0.10,
            'folds_b_wins':int(np.sum(ra>rb)),'n_folds':len(ra)}


# ============================================================
# VISUALIZATION (with real data)
# ============================================================
def plot_all(raw, df, imp_df, results_a, results_b, track_a):
    
    # FIG 1: Demand vs Muhurat Overlay
    fig, ax1 = plt.subplots(figsize=(16,6))
    dates = pd.to_datetime(raw[['year','month']].assign(day=15))
    ax1.plot(dates,raw['demand_index'],color=COLORS['primary'],lw=1.8,label='Composite Demand Index (Real)')
    ax1.fill_between(dates,0,raw['demand_index'],alpha=0.08,color=COLORS['primary'])
    ax1.set_ylabel('Demand Index (Google Trends)',color=COLORS['primary'],fontsize=12)
    ax1.set_ylim(0,110)
    ax2 = ax1.twinx()
    ax2.bar(dates,raw['hindu_muhurat_count'],width=25,alpha=0.45,color=COLORS['accent'],label='Muhurat Days')
    ax2.set_ylabel('Muhurat Days/Month',color=COLORS['accent'],fontsize=12)
    ax2.set_ylim(0,20)
    # COVID
    ax1.axvspan(pd.Timestamp('2020-04-01'),pd.Timestamp('2020-09-01'),alpha=0.15,color=COLORS['red'])
    ax1.annotate('COVID',xy=(pd.Timestamp('2020-06-15'),15),fontsize=9,color=COLORS['red'],ha='center',fontweight='bold')
    l1,la1=ax1.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels()
    ax1.legend(l1+l2,la1+la2,loc='upper left',fontsize=10)
    ax1.set_title('REAL DATA: Wedding Apparel Demand vs. Muhurat Date Density (India, 2010–2025)',fontsize=14,pad=15)
    plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/fig1_real_demand_vs_muhurat.png",dpi=150,bbox_inches='tight'); plt.close()
    print("  ✅ fig1_real_demand_vs_muhurat.png")
    
    # FIG 2: Correlation Scatter
    fig, axes = plt.subplots(1,2,figsize=(14,6))
    trainable = df[df['is_trainable']==1]
    for ax,col,clr,title in [
        (axes[0],'hindu_muhurat_count',COLORS['primary'],'Muhurat Count vs. Demand'),
        (axes[1],'combined_favorable_score',COLORS['green'],'Combined Cultural Score vs. Demand')]:
        x,y = trainable[col],trainable['demand_index']
        ax.scatter(x,y,c=clr,alpha=0.5,s=50,edgecolors='white',lw=0.5)
        z=np.polyfit(x,y,1); p=np.poly1d(z); xl=np.linspace(x.min(),x.max(),100)
        ax.plot(xl,p(xl),color=COLORS['red'],lw=2,ls='--',label=f'r = {x.corr(y):.3f}')
        ax.set_xlabel(col,fontsize=11); ax.set_ylabel('Demand Index',fontsize=11)
        ax.set_title(title,fontsize=13); ax.legend(fontsize=12)
    plt.suptitle('REAL DATA: Cultural Calendar Features vs. Apparel Demand',fontsize=14,fontweight='bold',y=1.02)
    plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/fig2_real_correlation.png",dpi=150,bbox_inches='tight'); plt.close()
    print("  ✅ fig2_real_correlation.png")
    
    # FIG 3: Feature Importance
    cultural_kw = ['muhurat','favorable','hijri','pitru','kharmas','mal_maas']
    fig, ax = plt.subplots(figsize=(12,8))
    top = imp_df.head(16)
    colors = [COLORS['accent'] if any(kw in f for kw in cultural_kw) else COLORS['gray'] for f in top['feature']]
    bars = ax.barh(range(len(top)),top['avg_pct'],color=colors,edgecolor='white',height=0.7)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top['feature'],fontsize=10); ax.invert_yaxis()
    ax.set_xlabel('Average Importance (%)',fontsize=12)
    cult_total = imp_df[imp_df['feature'].apply(lambda f:any(kw in f for kw in cultural_kw))]['avg_pct'].sum()
    ax.set_title(f'REAL DATA: Feature Importance — Cultural Features (★) = {cult_total:.0f}% of Signal',fontsize=13,pad=15)
    for i,(bar,val) in enumerate(zip(bars,top['avg_pct'])):
        ic = any(kw in top.iloc[i]['feature'] for kw in cultural_kw)
        ax.text(bar.get_width()+0.3,bar.get_y()+bar.get_height()/2,
                f'{val:.1f}% ★' if ic else f'{val:.1f}%',va='center',fontsize=9,
                fontweight='bold' if ic else 'normal',color=COLORS['accent'] if ic else COLORS['gray'])
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=COLORS['accent'],label='Cultural (Novel)'),
                        Patch(facecolor=COLORS['gray'],label='Standard Temporal')],loc='lower right',fontsize=10)
    plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/fig3_real_feature_importance.png",dpi=150,bbox_inches='tight'); plt.close()
    print("  ✅ fig3_real_feature_importance.png")
    
    # FIG 4: Model Comparison Bars
    ra_base = results_a['Ridge|baseline_no_lag']['mean_rmse']
    ra_enh = results_a['Ridge|enhanced_no_lag']['mean_rmse']
    ra_full = results_a['Ridge|full_no_lag']['mean_rmse']
    rb_base = results_b['Ridge|baseline_with_lag']['mean_rmse']
    rb_enh = results_b['Ridge|enhanced_with_lag']['mean_rmse']
    imp_a = (ra_base-ra_enh)/ra_base*100
    imp_b = (rb_base-rb_enh)/rb_base*100
    ta = paired_test(results_a['Ridge|baseline_no_lag']['rmses'],results_a['Ridge|enhanced_no_lag']['rmses'],'B','E')
    tb = paired_test(results_b['Ridge|baseline_with_lag']['rmses'],results_b['Ridge|enhanced_with_lag']['rmses'],'B','E')
    
    fig, axes = plt.subplots(1,2,figsize=(14,6))
    ax = axes[0]
    bars=ax.bar(['Ridge\nBaseline','Ridge\nEnhanced','Ridge\nFull'],[ra_base,ra_enh,ra_full],
                 color=[COLORS['gray'],COLORS['green'],COLORS['purple']],edgecolor='white',width=0.6)
    for b,v in zip(bars,[ra_base,ra_enh,ra_full]):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.2,f'{v:.2f}',ha='center',fontsize=11,fontweight='bold')
    sig_a = "✅ p<0.05" if ta['significant'] else ("⚠️ p<0.10" if ta['marginal'] else f"p={ta['p_one_sided']:.3f}")
    ax.set_title(f'Track A: No-Lag (Cold-Start)\n{imp_a:+.1f}% improvement | {sig_a}',fontsize=12)
    ax.set_ylabel('RMSE (lower = better)'); ax.set_ylim(0,max(ra_base,ra_enh,ra_full)*1.25)
    
    ax = axes[1]
    bars=ax.bar(['Ridge\nBaseline+Lag','Ridge\nEnhanced+Lag'],[rb_base,rb_enh],
                 color=[COLORS['gray'],COLORS['green']],edgecolor='white',width=0.5)
    for b,v in zip(bars,[rb_base,rb_enh]):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.1,f'{v:.2f}',ha='center',fontsize=11,fontweight='bold')
    sig_b = "✅ p<0.05" if tb['significant'] else ("⚠️ p<0.10" if tb['marginal'] else f"p={tb['p_one_sided']:.3f}")
    ax.set_title(f'Track B: With-Lag (Practical)\n{imp_b:+.1f}% improvement | {sig_b}',fontsize=12)
    ax.set_ylabel('RMSE (lower = better)'); ax.set_ylim(0,max(rb_base,rb_enh)*1.25)
    
    plt.suptitle('REAL DATA: Model Comparison — Cultural Features Impact',fontsize=14,fontweight='bold',y=1.02)
    plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/fig4_real_model_comparison.png",dpi=150,bbox_inches='tight'); plt.close()
    print("  ✅ fig4_real_model_comparison.png")
    
    # FIG 5: Heatmap
    fig, ax = plt.subplots(figsize=(14,8))
    pivot = raw.pivot_table(values='hindu_muhurat_count',index='year',columns='month')
    mnames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    sns.heatmap(pivot,annot=True,fmt='.0f',cmap='YlOrRd',xticklabels=mnames,
                linewidths=0.5,linecolor='white',cbar_kws={'label':'Muhurat Days'},ax=ax)
    ax.set_title('Shubh Vivah Muhurat Density: Year × Month',fontsize=13,pad=15)
    plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/fig5_real_muhurat_heatmap.png",dpi=150,bbox_inches='tight'); plt.close()
    print("  ✅ fig5_real_muhurat_heatmap.png")
    
    # FIG 6: Predictions vs Actual
    best_key = min(results_a.keys(),key=lambda k:results_a[k]['mean_rmse'])
    pred_df = results_a[best_key]['predictions'].copy()
    pred_df['date'] = pd.to_datetime(pred_df[['year','month']].assign(day=15))
    pred_df = pred_df.sort_values('date')
    
    fig, axes = plt.subplots(2,1,figsize=(16,10),gridspec_kw={'height_ratios':[3,1]})
    ax = axes[0]
    ax.plot(pred_df['date'],pred_df['actual'],color=COLORS['primary'],lw=1.5,label='Actual (Real Google Trends)',marker='o',ms=3)
    ax.plot(pred_df['date'],pred_df['predicted'],color=COLORS['accent'],lw=1.5,label='Predicted (Enhanced)',marker='s',ms=3,ls='--')
    ax.fill_between(pred_df['date'],pred_df['actual'],pred_df['predicted'],alpha=0.15,color=COLORS['red'])
    ax.legend(fontsize=11); ax.set_ylabel('Demand Index',fontsize=12)
    ax.set_title(f'REAL DATA: Actual vs. Predicted (LOYO-CV) | Best Model: {best_key}',fontsize=13)
    
    ax = axes[1]
    res = pred_df['predicted']-pred_df['actual']
    ax.bar(pred_df['date'],res,width=25,color=[COLORS['green'] if r>=0 else COLORS['red'] for r in res],alpha=0.6)
    ax.axhline(0,color='black',lw=0.5); ax.set_ylabel('Error'); ax.set_xlabel('Date')
    
    plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/fig6_real_predictions.png",dpi=150,bbox_inches='tight'); plt.close()
    print("  ✅ fig6_real_predictions.png")
    
    # FIG 7 (NEW): Three Search Terms Decomposition
    fig, ax = plt.subplots(figsize=(16,5))
    dates = pd.to_datetime(raw[['year','month']].assign(day=15))
    ax.plot(dates,raw['wedding_lehenga'],label='wedding lehenga',color=COLORS['red'],alpha=0.7,lw=1.2)
    ax.plot(dates,raw['sherwani'],label='sherwani',color=COLORS['green'],alpha=0.7,lw=1.2)
    ax.plot(dates,raw['bridal_saree'],label='bridal saree',color=COLORS['purple'],alpha=0.7,lw=1.2)
    ax.plot(dates,raw['demand_index'],label='Composite Index',color=COLORS['primary'],lw=2.5)
    ax.legend(fontsize=10); ax.set_ylabel('Google Trends Index (0-100)'); ax.set_xlabel('Date')
    ax.set_title('REAL DATA: Three Wedding Search Terms + Composite Demand Index (India, 2010–2025)',fontsize=13)
    plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/fig7_real_search_terms.png",dpi=150,bbox_inches='tight'); plt.close()
    print("  ✅ fig7_real_search_terms.png")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("THE MISSING VARIABLE — REAL DATA EXPERIMENT")
    print("Using ACTUAL Google Trends Data (2010–2025)")
    print("=" * 70)
    
    # 1. Build master dataset with real trends
    raw = build_master_with_real_trends()
    raw.to_csv(os.path.join(PROJECT_ROOT, "data", "raw", "master.csv"), index=False)
    
    # 2. Engineer features
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)
    df = engineer_features(raw)
    
    # 3. Key correlations with REAL data
    tr = df[df['is_trainable']==1]
    corr_muh = tr['hindu_muhurat_count'].corr(tr['demand_index'])
    corr_fav = tr['combined_favorable_score'].corr(tr['demand_index'])
    corr_lag = tr['demand_lag_12m'].corr(tr['demand_index'])
    print(f"\n🔗 Key Correlations with REAL Demand:")
    print(f"   hindu_muhurat_count:      r = {corr_muh:.4f}")
    print(f"   combined_favorable_score: r = {corr_fav:.4f}")
    print(f"   demand_lag_12m:           r = {corr_lag:.4f}")
    
    # 4. Run experiments
    track_a, track_b = get_feature_sets()
    ridge_p = {'alpha':1.0}
    gbm_p = {'n_estimators':100,'max_depth':3,'learning_rate':0.05,
             'subsample':0.8,'random_state':42,'min_samples_leaf':4}
    
    print("\n" + "=" * 70)
    print("TRACK A: NO-LAG EXPERIMENT (REAL DATA)")
    print("=" * 70)
    results_a = {}
    for mn,(cls,par) in [('Ridge',(Ridge,ridge_p)),('GBM',(GradientBoostingRegressor,gbm_p))]:
        print(f"\n  Model: {mn}")
        for fn,fc in track_a.items():
            r = loyo_cv(df,fc,cls,par); results_a[f"{mn}|{fn}"] = r
            print(f"    {fn:20s} ({len(fc):2d} feats) → RMSE: {r['mean_rmse']:6.2f} ± {r['std_rmse']:.2f} | R²: {r['mean_r2']:.3f}")
    
    print("\n  Significance Tests (Track A):")
    for mn in ['Ridge','GBM']:
        t = paired_test(results_a[f'{mn}|baseline_no_lag']['rmses'],
                        results_a[f'{mn}|enhanced_no_lag']['rmses'],'Baseline','Enhanced')
        flag = "✅ SIG" if t['significant'] else ("⚠️ MARG" if t['marginal'] else "❌ NS")
        print(f"    {mn:5s} | {t['comparison']:25s} | RMSE {t['rmse_a']:.2f}→{t['rmse_b']:.2f} "
              f"({t['improvement_pct']:+.1f}%) | p={t['p_one_sided']:.4f} | Won {t['folds_b_wins']}/{t['n_folds']} | {flag}")
    
    print("\n" + "=" * 70)
    print("TRACK B: WITH-LAG EXPERIMENT (REAL DATA)")
    print("=" * 70)
    results_b = {}
    for mn,(cls,par) in [('Ridge',(Ridge,ridge_p)),('GBM',(GradientBoostingRegressor,gbm_p))]:
        print(f"\n  Model: {mn}")
        for fn,fc in track_b.items():
            r = loyo_cv(df,fc,cls,par); results_b[f"{mn}|{fn}"] = r
            print(f"    {fn:20s} ({len(fc):2d} feats) → RMSE: {r['mean_rmse']:6.2f} ± {r['std_rmse']:.2f} | R²: {r['mean_r2']:.3f}")
    
    print("\n  Significance Tests (Track B):")
    for mn in ['Ridge','GBM']:
        t = paired_test(results_b[f'{mn}|baseline_with_lag']['rmses'],
                        results_b[f'{mn}|enhanced_with_lag']['rmses'],'Baseline+Lag','Enhanced+Lag')
        flag = "✅ SIG" if t['significant'] else ("⚠️ MARG" if t['marginal'] else "❌ NS")
        print(f"    {mn:5s} | {t['comparison']:30s} | RMSE {t['rmse_a']:.2f}→{t['rmse_b']:.2f} "
              f"({t['improvement_pct']:+.1f}%) | p={t['p_one_sided']:.4f} | Won {t['folds_b_wins']}/{t['n_folds']} | {flag}")
    
    # 5. Feature Importance (Track A Enhanced)
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE — REAL DATA")
    print("=" * 70)
    feats = track_a['enhanced_no_lag']
    X = tr[feats].fillna(0).values; y = tr['demand_index'].values
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    ridge = Ridge(alpha=1.0); ridge.fit(Xs,y)
    gbm = GradientBoostingRegressor(**gbm_p); gbm.fit(Xs,y)
    imp_df = pd.DataFrame({'feature':feats,
        'ridge_pct':np.abs(ridge.coef_)/np.sum(np.abs(ridge.coef_))*100,
        'gbm_pct':gbm.feature_importances_/np.sum(gbm.feature_importances_)*100})
    imp_df['avg_pct'] = (imp_df['ridge_pct']+imp_df['gbm_pct'])/2
    imp_df = imp_df.sort_values('avg_pct',ascending=False)
    
    cultural_kw = ['muhurat','favorable','hijri','pitru','kharmas','mal_maas']
    print(f"\n  {'Rank':>4}  {'Feature':<30}  {'Ridge%':>7}  {'GBM%':>7}  {'Avg%':>6}")
    print(f"  {'─'*4}  {'─'*30}  {'─'*7}  {'─'*7}  {'─'*6}")
    for i,(_,row) in enumerate(imp_df.iterrows()):
        ic = any(kw in row['feature'] for kw in cultural_kw)
        print(f"  {i+1:4d}  {row['feature']:<30}  {row['ridge_pct']:6.1f}%  {row['gbm_pct']:6.1f}%  {row['avg_pct']:5.1f}%{'  ★' if ic else ''}")
    
    cult_total = imp_df[imp_df['feature'].apply(lambda f:any(kw in f for kw in cultural_kw))]['avg_pct'].sum()
    print(f"\n  ★ Cultural features total importance: {cult_total:.1f}%")
    
    imp_df.to_csv(f"{RESULTS_DIR}/real_feature_importance.csv",index=False)
    
    # 6. Generate all figures
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    plot_all(raw, df, imp_df, results_a, results_b, track_a)
    
    # 7. Save predictions
    best_a = min(results_a.keys(),key=lambda k:results_a[k]['mean_rmse'])
    results_a[best_a]['predictions'].to_csv(f"{RESULTS_DIR}/real_track_a_predictions.csv",index=False)
    best_b = min(results_b.keys(),key=lambda k:results_b[k]['mean_rmse'])
    results_b[best_b]['predictions'].to_csv(f"{RESULTS_DIR}/real_track_b_predictions.csv",index=False)
    
    # 8. Final Verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT — REAL GOOGLE TRENDS DATA")
    print("=" * 70)
    
    ra_b = results_a['Ridge|baseline_no_lag']; ra_e = results_a['Ridge|enhanced_no_lag']
    ta = paired_test(ra_b['rmses'],ra_e['rmses'],'Baseline','Enhanced')
    rb_b = results_b['Ridge|baseline_with_lag']; rb_e = results_b['Ridge|enhanced_with_lag']
    tb = paired_test(rb_b['rmses'],rb_e['rmses'],'Baseline','Enhanced')
    
    print(f"\n  TRACK A (Cold-Start):")
    print(f"    Baseline RMSE: {ta['rmse_a']:.2f}  →  Enhanced: {ta['rmse_b']:.2f}")
    print(f"    Improvement: {ta['improvement_pct']:+.1f}%  |  p = {ta['p_one_sided']:.4f}  |  Won {ta['folds_b_wins']}/{ta['n_folds']} folds")
    if ta['significant']: print(f"    ✅ SIGNIFICANT at p < 0.05")
    elif ta['marginal']: print(f"    ⚠️  MARGINAL at p < 0.10")
    else: print(f"    Result: {'Positive trend' if ta['improvement_pct']>0 else 'Not significant'}")
    
    print(f"\n  TRACK B (Practical):")
    print(f"    Baseline RMSE: {tb['rmse_a']:.2f}  →  Enhanced: {tb['rmse_b']:.2f}")
    print(f"    Improvement: {tb['improvement_pct']:+.1f}%  |  p = {tb['p_one_sided']:.4f}  |  Won {tb['folds_b_wins']}/{tb['n_folds']} folds")
    if tb['significant']: print(f"    ✅ SIGNIFICANT at p < 0.05")
    elif tb['marginal']: print(f"    ⚠️  MARGINAL at p < 0.10")
    else: print(f"    Result: {'Positive trend' if tb['improvement_pct']>0 else 'Not significant'}")
    
    print(f"\n  KEY NUMBERS:")
    print(f"    muhurat_count × demand correlation: r = {corr_muh:.4f}")
    print(f"    Cultural feature total importance:   {cult_total:.1f}%")
    print(f"    Data source: REAL Google Trends (wedding lehenga + sherwani + bridal saree)")
    
    print(f"\n  💾 All results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
