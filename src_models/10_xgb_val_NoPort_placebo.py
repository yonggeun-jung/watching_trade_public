import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from pathlib import Path
import optuna
import joblib

SEED = 42
np.random.seed(SEED)

df = pd.read_csv("watching_trade/data/cleaned/main.csv")

satellite_cols = ['sar_diff_median', 'vh_median_mean','ntl_mean', 'ntl_std', 'lit_area_ratio']

feature_cols = satellite_cols
target_col = 'log_total_ves_val'

df_clean = df[feature_cols + [target_col, 'year_month', 'port_name']].dropna()

# Categorical encoding
cat_cols = df_clean[feature_cols].select_dtypes(include=['object']).columns.tolist()
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    label_encoders[col] = le

# === PLACEBO: Shuffle X (with fixed seed) ===
rng = np.random.RandomState(SEED)  
df_placebo = df_clean.copy()
for col in feature_cols:
    df_placebo[col] = rng.permutation(df_placebo[col].values)

# Time-based split
df_placebo = df_placebo.sort_values('year_month').reset_index(drop=True)
split_idx = int(len(df_placebo) * 0.7)

df_train_full = df_placebo.iloc[:split_idx].copy()
df_test = df_placebo.iloc[split_idx:].copy()

val_split_idx = int(len(df_train_full) * 0.8)
df_train = df_train_full.iloc[:val_split_idx].copy()
df_val = df_train_full.iloc[val_split_idx:].copy()

X_train = df_train[feature_cols].values
y_train = df_train[target_col].values
X_val = df_val[feature_cols].values
y_val = df_val[target_col].values
X_test = df_test[feature_cols].values
y_test = df_test[target_col].values

print(f"PLACEBO TEST (Shuffled X)")
print(f"Train: {len(df_train)} rows ({df_train['year_month'].min()} ~ {df_train['year_month'].max()})")
print(f"Val: {len(df_val)} rows ({df_val['year_month'].min()} ~ {df_val['year_month'].max()})")
print(f"Test: {len(df_test)} rows ({df_test['year_month'].min()} ~ {df_test['year_month'].max()})")


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
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
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, preds))


print("\nHyperparameter Search")
sampler = optuna.samplers.TPESampler(seed=SEED)  # Optuna sampler with fixed seed
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"\nBest val_rmse: {study.best_trial.value:.4f}")
print(f"Best params: {study.best_trial.params}")

best = study.best_trial.params
best['random_state'] = SEED
best['n_jobs'] = -1

X_train_full = df_train_full[feature_cols].values
y_train_full = df_train_full[target_col].values

model = xgb.XGBRegressor(**best)
model.fit(X_train_full, y_train_full, verbose=False)

# 전체 예측
X_all = df_placebo[feature_cols].values
preds_all = model.predict(X_all)

preds_test = model.predict(X_test)
actuals_test = y_test

metrics = {
    'model': 'XGBoost_Placebo',
    'n_train': len(y_train_full),
    'n_test': len(y_test),
    'test_period': f"{df_test['year_month'].min()} ~ {df_test['year_month'].max()}",
    'r2': r2_score(actuals_test, preds_test),
    'mae': mean_absolute_error(actuals_test, preds_test),
    'rmse': np.sqrt(mean_squared_error(actuals_test, preds_test)),
    'mape': mean_absolute_percentage_error(actuals_test, preds_test) * 100,
    'corr': np.corrcoef(actuals_test, preds_test)[0, 1],
    'best_params': str(study.best_trial.params),
}

print(f"\nPlacebo Test Results")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")

pd.DataFrame([metrics]).to_csv("watching_trade/output_tables/10_01_xgb_metrics_placebo.csv", index=False)

# Save predictions with test flag
result_df = df_placebo[['port_name', 'year_month']].copy()
result_df['actual'] = df_placebo[target_col].values
result_df['predicted'] = preds_all
result_df['error'] = result_df['predicted'] - result_df['actual']
result_df['is_test'] = 0
result_df.loc[split_idx:, 'is_test'] = 1
result_df.to_csv("watching_trade/output_tables/10_02_xgb_predictions_placebo.csv", index=False)

# Feature importance
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
importance_df.to_csv("watching_trade/output_tables/10_03_xgb_importance_placebo.csv", index=False)

print(f"\nFeature Importance:")
print(importance_df)

print(f"\nCompleted.")