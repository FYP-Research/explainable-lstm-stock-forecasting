# explainable-lstm-stock-forecasting
BEng FYP: Explainable LSTM stock forecasting using SHAP and LIME

# Explainable LSTM-based Stock Price Forecasting Using SHAP and LIME

BEng Final Year Project submission. Implementation of a nine-experiment 
pipeline for one-step-ahead daily closing price forecasting on Apple 
Inc. (AAPL), combined with SHAP and LIME explainability analysis.

## Notebooks

- **`experiments_1_to_3_data_pipeline.ipynb`** — Experiments 1 to 3: 
  baseline model comparison, returns-based reformulation, and 13-year 
  dataset extension with directional significance testing. Also 
  contains the exploratory data analysis on both the earlier 3-year 
  and final 13-year datasets.

- **`experiments_4_to_9_main_pipeline.ipynb`** — Experiments 4 to 9: 
  regularisation, model compression, architectural robustness, 
  cross-architecture error overlap analysis, augmentation and 
  classification reformulation, SHAP-guided feature reduction, and the 
  full SHAP + LIME explainability analysis including intervention-based 
  validation.

- **`sendi_replication.ipynb`** — Replication of Sendi et al. (2025) 
  methodology on AAPL data for benchmark comparison.

## Dataset

AAPL daily OHLCV data (December 2012 to December 2025, 3,248 trading 
days) is downloaded from Yahoo Finance using the `yfinance` library at 
runtime. No data files are stored in this repository.

## Environment

All experiments were executed on Google Colaboratory with an NVIDIA T4 
GPU. Full library versions are documented in `requirements.txt`.

## Reproduction

All notebooks use fixed random seeds where appropriate. Each experiment 
reports results as mean plus or minus standard deviation across 10 
independent runs with full hyperparameter disclosure.

## Key Results

- RMSE $3.77 with standard deviation 0.00, MAE $2.54 with standard deviation 0.00
- Directional accuracy 56.6% with standard deviation 0.1% (p = 1.25 x 10^-19)
- SHAP-LIME rank agreement: Spearman rho = 0.939 (64/32 model), rho = 1.000 (32/16 compressed)
- 73% parameter reduction with zero performance loss
- Approximately 280 g CO2 total across the nine-experiment pipeline
