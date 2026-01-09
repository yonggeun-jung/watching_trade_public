import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from pathlib import Path
import optuna
import joblib

SEED = 42
np.random.seed(SEED)

df = pd.read_csv("watching_trade/data/cleaned/main.csv")
df = df.sort_values(['port_name', 'year_month']).reset_index(drop=True)

satellite_cols = ['sar_diff_median', 'vh_median_mean','ntl_mean', 'ntl_std', 'lit_area_ratio']

feature_cols = satellite_cols                  # without Port features
target_col = 'log_total_ves_val'               # total trade (vessel) value 

df_clean = df[feature_cols + [target_col, 'harbor_size', 'region', 'port_name', 'year_month']].dropna()

print(f"Features: {len(feature_cols)}")
print(f"Sample size: {len(df_clean)}")

# Categorical encoding
cat_cols = df_clean[feature_cols].select_dtypes(include=['object']).columns.tolist()
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    label_encoders[col] = le


target_size = 'Medium'
df_size = df_clean[df_clean['harbor_size'] == target_size].copy()

print(f"\n{target_size} harbors: {len(df_size)} rows, {df_size['port_name'].nunique()} ports")

holdout_region = 'Hawaiian Islands -- 56050'
df_train = df_size[df_size['region'] != holdout_region].copy()
df_test = df_size[df_size['region'] == holdout_region].copy()

print(f"Train: {len(df_train)} rows ({df_train['port_name'].nunique()} ports)")
print(f"Test ({holdout_region}): {len(df_test)} rows ({df_test['port_name'].nunique()} ports)")
print(f"Test ports: {df_test['port_name'].unique()}")

X_train = df_train[feature_cols].values
y_train = df_train[target_col].values
X_test = df_test[feature_cols].values
y_test = df_test[target_col].values


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 2, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'random_state': SEED,
        'n_jobs': -1,
    }
    
    model = xgb.XGBRegressor(**params)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
    return -scores.mean()


print("\nHyperparameter Search")
sampler = optuna.samplers.TPESampler(seed=SEED)
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"\nBest CV RMSE: {study.best_trial.value:.4f}")
print(f"Best params: {study.best_trial.params}")

best = study.best_trial.params
best['random_state'] = SEED
best['n_jobs'] = -1

model = xgb.XGBRegressor(**best)
model.fit(X_train, y_train, verbose=False)

# Predictions
preds_raw = model.predict(X_test)
actuals = y_test

# Bias correction: per-port first month basis
df_test = df_test.copy()
df_test['pred_raw'] = preds_raw
df_test['actual'] = actuals

preds_adjusted = []
for port in df_test['port_name'].unique():
    port_df = df_test[df_test['port_name'] == port].copy()
    first_idx = port_df.index[0]
    bias = port_df.loc[first_idx, 'actual'] - port_df.loc[first_idx, 'pred_raw']
    port_df['pred_adjusted'] = port_df['pred_raw'] + bias
    preds_adjusted.append(port_df)

df_result = pd.concat(preds_adjusted)
df_result = df_result.sort_values(['port_name', 'year_month'])

# Metrics 
metrics_raw = {
    'model': 'XGBoost_LOO_Raw',
    'holdout_region': holdout_region,
    'harbor_size': target_size,
    'n_train': len(y_train),
    'n_test': len(y_test),
    'r2': r2_score(df_result['actual'], df_result['pred_raw']),
    'mae': mean_absolute_error(df_result['actual'], df_result['pred_raw']),
    'rmse': np.sqrt(mean_squared_error(df_result['actual'], df_result['pred_raw'])),
    'mape': mean_absolute_percentage_error(df_result['actual'], df_result['pred_raw']) * 100,
    'corr': np.corrcoef(df_result['actual'], df_result['pred_raw'])[0, 1],
    'best_params': str(study.best_trial.params),
}

metrics_adj = {
    'model': 'XGBoost_LOO_Adj',
    'holdout_region': holdout_region,
    'harbor_size': target_size,
    'n_train': len(y_train),
    'n_test': len(y_test),
    'r2': r2_score(df_result['actual'], df_result['pred_adjusted']),
    'mae': mean_absolute_error(df_result['actual'], df_result['pred_adjusted']),
    'rmse': np.sqrt(mean_squared_error(df_result['actual'], df_result['pred_adjusted'])),
    'mape': mean_absolute_percentage_error(df_result['actual'], df_result['pred_adjusted']) * 100,
    'corr': np.corrcoef(df_result['actual'], df_result['pred_adjusted'])[0, 1],
    'best_params': str(study.best_trial.params),
}

print(f"\nRaw Prediction")
for k, v in metrics_raw.items():
    if isinstance(v, float):
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")

print(f"\nBias Adjusted Prediction")
for k, v in metrics_adj.items():
    if isinstance(v, float):
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")

# save outputs

# Metrics
metrics_df = pd.DataFrame([metrics_raw, metrics_adj])
metrics_df.to_csv("watching_trade/output_tables/09_01_xgb_metrics_LOO_val_noports.csv", index=False)

# Predictions
pred_df = df_result[['port_name', 'year_month', 'actual', 'pred_raw', 'pred_adjusted']].copy()
pred_df['error_raw'] = pred_df['pred_raw'] - pred_df['actual']
pred_df['error_adj'] = pred_df['pred_adjusted'] - pred_df['actual']
pred_df.to_csv("watching_trade/output_tables/09_02_xgb_predictions_LOO_val_noports.csv", index=False)

# Feature importance
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
importance_df.to_csv("watching_trade/output_tables/09_03_xgb_importance_LOO_val_noports.csv", index=False)

print(f"\nFeature Importance:")
print(importance_df)

# Model save
joblib.dump(model, "watching_trade/output_models/08_xgb_model_LOO_val_noports.joblib")

print(f"\nPer-port results (Raw):")
print(pred_df.groupby('port_name')[['actual', 'pred_raw', 'error_raw']].mean().round(2))

print(f"\nPer-port results (Bias Adjusted):")
print(pred_df.groupby('port_name')[['actual', 'pred_adjusted', 'error_adj']].mean().round(2))


print(f"\nCompleted.")