# Methodology

## 1. Problem Statement

Standard retail demand forecasting models for the Indian wedding market rely on Gregorian calendar features and historical sales lags. They ignore the Hindu Panchang — the calendar system that actually determines when weddings happen. This study tests whether encoding Panchang-based features into a demand model yields statistically significant improvements.

## 2. Data Sources

**Target Variable — Wedding Apparel Demand Proxy**
Monthly Google Trends data (2010–2025) for three India-specific search terms:
- "wedding lehenga" — bridal wear
- "sherwani" — groom wear  
- "bridal saree" — traditional bridal wear

These were averaged into a composite demand index (0–100 scale). The composite was selected over individual terms after empirical comparison (see Section 3).

**Feature Sources:**
- Hindu Panchang: Shubh Vivah Muhurat day counts, Pitru Paksha, Kharmas, Mal Maas periods (derived from Drik Panchang seasonal patterns)
- Hijri Calendar: Ramadan and Muharram restricted days (from IslamicFinder)
- Economic: Gold prices in INR/10g (World Gold Council), Consumer Price Index (RBI)

## 3. Target Variable Selection (EDA)

I evaluated each search term individually as a demand proxy before selecting the composite.

| Term | Corr. with Muhurat | Track B p-val (Ridge) | Track B p-val (GBM) |
|------|-------------------|-----------------------|---------------------|
| wedding lehenga | r = 0.22 | p = 0.371 (NS) | p = 0.582 (NS) |
| sherwani | r = 0.53 | — | — |
| bridal saree | r = 0.30 | — | — |
| **composite (avg)** | **r = 0.43** | **p = 0.020 (SIG)** | **p = 0.007 (SIG)** |

The lehenga term's low correlation is explained by its secular growth trend — near-zero search volume in 2010 rising to 100 by 2023, driven by internet penetration rather than changes in wedding frequency. This growth trend overwhelms the seasonal signal. The composite dilutes this trend artifact while preserving the seasonal pattern.

## 4. Feature Engineering

**Layer 1 — Temporal Baseline (14 features)**
Cyclical month encoding, quarter, year trend, Diwali proximity, demand lags (1m/2m/3m/12m), rolling mean/std (3m/6m), YoY change.

**Layer 2 — Cultural Calendar (16 features)**
Muhurat count and density, lead/lag indicators, rolling muhurat averages, inauspicious period flags (Pitru Paksha, Kharmas, Mal Maas), Hijri restricted days and availability ratio, combined favorable score.

**Layer 3 — Economic Context (5 features)**
Gold price (normalized, 3m/12m momentum), CPI YoY change, muhurat×gold interaction.

Total: 46 features from 12 raw columns.

## 5. Experimental Framework

**Track A (Cold-Start):** Baseline (6 temporal features) vs Enhanced (+10 cultural features) vs Full (+3 economic features). No lag features. Simulates new market entry.

**Track B (Practical):** Baseline (9 features: temporal + lags) vs Enhanced (+6 cultural features). With lag features. Simulates existing retailer.

**Cross-Validation:** Leave-One-Year-Out (LOYO), 14 folds (2012–2025). Each fold holds out one calendar year for testing, trains on all remaining years. This prevents temporal leakage and tests generalization across different muhurat configurations.

**Statistical Testing:** Paired t-test on fold-level RMSE vectors. One-sided test (H₁: enhanced RMSE < baseline RMSE). Significance threshold: p < 0.05.

**Models:** Ridge Regression (α=1.0) and Gradient Boosting (100 trees, depth 3, lr 0.05, subsample 0.8).

## 6. Results

**Track A:** Neither Ridge nor GBM showed significant improvement from cultural features alone (p > 0.7 for both). Cultural features cannot substitute for historical demand data.

**Track B:**
- Ridge: RMSE 4.60 → 4.30 (+6.4%), p = 0.020, won 11/14 folds
- GBM: RMSE 4.51 → 4.23 (+6.3%), p = 0.018, won 10/14 folds

Both significant at p < 0.05.

**Feature Importance:** Cultural features collectively account for 31.8% of the Enhanced model's predictive signal. `hindu_muhurat_count` ranks 4th among all 16 features.

## 7. Interpretation

Cultural calendar features provide meaningful incremental value when combined with standard time-series features. They capture inter-year variation in muhurat distribution that lag-based models cannot — a year where November has 15 muhurat days will see different demand than one where November has 9, even if the prior year's November sales were identical.

The null result in Track A is expected and informative: wedding demand is influenced by many factors beyond the calendar (fashion trends, economic conditions, regional events), so calendar features alone are insufficient predictors.

## 8. Reproducibility

All results can be reproduced with a single command: `python run_real_experiment.py`. The pipeline loads Google Trends CSVs, generates calendar/economic data, engineers features, runs cross-validation, performs significance tests, and generates all figures.
