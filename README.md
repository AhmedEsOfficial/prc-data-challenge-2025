# PRC Data Challenge 2025 — Aircraft Fuel Consumption Prediction

My entry to the [PRC Data Challenge 2025](https://ansperformance.eu/study/data-challenge/) (EUROCONTROL Performance Review Commission × OpenSky Network). The task is to predict the fuel burnt (`fuel_kg`) by a flight during given time intervals, using its ADS-B trajectory, flight list and airport data. Submissions are scored by RMSE.

The whole pipeline was built and trained on a laptop: Polars for memory-efficient feature extraction, then gradient-boosted tree ensembles (LightGBM, CatBoost, XGBoost) stratified by fuel range.

## Results

Validation RMSE, by phase:

| Phase | Approach | Validation RMSE |
|-------|----------|----------------:|
| 4 | Baseline models — Ridge, Random Forest, LightGBM, **CatBoost** | 196.20 kg |
| 5 | Optuna tuning + weighted ensemble | 189.29 kg |
| 6 | Error analysis, physics features, two-stage model | 163.04 kg |
| 7 | Fuel-range stratification, log targets, ensembles | 144.19 kg |
| 8 | 11-stage stratification, multi-model blend per stage | **84.30 kg** |

Most of the remaining error comes from the rare, very high fuel intervals (above 2,500 kg, about 2% of the samples). [`documentation/PHASE8_COMPREHENSIVE_SUMMARY.md`](documentation/PHASE8_COMPREHENSIVE_SUMMARY.md) breaks the error down by stage and explains why.

## Approach

1. **Data understanding:** profiling the fuel intervals, the trajectories and the gaps in their data ([phase 1](notebooks/phase1_data_understanding.py)).
2. **Feature engineering:** for each interval, aggregates of the trajectory within it (altitude, speed, vertical rate), the flight phase, the interval's position within the flight, and features about the aircraft type and route ([phase 2](notebooks/phase2_feature_engineering.py), [`src/features.py`](src/features.py)).
3. **Data preparation:** a temporal train/validation split (April–July for training, August for validation), target and frequency encoding of the categorical features, and feature scaling ([phase 3](notebooks/phase3_data_preparation.py), [`src/preprocessing.py`](src/preprocessing.py)).
4. **Modelling:**
   - Baselines, followed by Optuna hyperparameter search and ensembling (phases 4–5).
   - Error analysis led to *stratified* modelling: separate models per fuel range, each with its own hyperparameters and blend weights (phases 6–8).

The full plan is in [`documentation/PLAN.md`](documentation/PLAN.md), and each phase has a write-up in [`documentation/`](documentation/).

## Repository layout

```
├── pull.py                     # download the challenge data (needs team credentials)
├── analyze.py                  # parquet schema / profile dump
├── notebooks/                  # phases 1–7, one script per experiment
├── src/                        # shared feature engineering & preprocessing
├── prepare_phase8_data.py      # phase 8 feature table with flight ids
├── train_phase8_*.py           # phase 8 training variants (ultimate_v2 = best)
├── predict_phase8_ranking.py   # build a ranking submission from trained models
├── models/                     # trained LightGBM / CatBoost models and ensemble weights
├── reports/                    # metrics and logs from each phase
├── documentation/              # plan and per-phase write-ups
├── context/                    # challenge description: data, submission, eligibility
├── external/                   # weather-data exploration (Meteostat)
└── plots/
```

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Get the data. Credentials are issued per team by the challenge organisers.
export OPENSKY_ACCESS_KEY=...
export OPENSKY_SECRET_KEY=...
python pull.py                     # downloads into the repo root (git-ignored)

# 2. Feature pipeline
python notebooks/phase1_data_understanding.py
python notebooks/phase2_feature_engineering.py
python notebooks/phase3_data_preparation.py
python notebooks/phase6_advanced_features.py   # writes cache/X_*_enhanced.parquet
python notebooks/phase6_fix_and_retrain.py     # writes cache/X_*_clean.parquet

# 3. Train the best model (phase 8 ultimate v2)
python train_phase8_ultimate_v2.py
```

Run every script from the repository root, because paths resolve relative to it. The challenge data is not redistributed here: register for the challenge to get access to it.

## Status

This is competition research code: one script per experiment, kept as they were run, so the history of what worked and what didn't stays readable. The scored results above come from the validation split. [`predict_phase8_ranking.py`](predict_phase8_ranking.py) expects ranking-set features at `cache/ranking_features.parquet`, and no script here produces that file. You would have to run the phase 2 feature extraction against `flightlist_rank.parquet` and `fuel_rank_submission.parquet` first.
