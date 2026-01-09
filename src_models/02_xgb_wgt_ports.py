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

ports_cols = [
    'harbor_size', 'harbor_type', 'harbor_use', 'shelter_afforded',
    'ent_restr_tide', 'ent_restr_swell', 'ent_restr_ice', 'ent_restr_other',
    'overhead_limits', 'underkeel_mgmt', 'good_holding_ground', 'turning_area',
    'traffic_sep_scheme', 'vessel_traffic_svc', 'search_rescue',
    'port_security', 'eta_message',
    'quarantine_pratique', 'quarantine_sanitation', 'quarantine_other',
    'first_port_of_entry', 'us_representative',
    'pilotage_compulsory', 'pilotage_available', 'pilotage_local_assist', 'pilotage_advisable',
    'tugs_salvage', 'tugs_assistance',
    'comm_telephone', 'comm_telefax', 'comm_radio', 'comm_radiotelephone',
    'comm_airport', 'comm_rail',
    'fac_wharves', 'fac_anchorage', 'fac_dangerous_cargo',
    'fac_med_mooring', 'fac_beach_mooring', 'fac_ice_mooring',
    'fac_roro', 'fac_solid_bulk', 'fac_liquid_bulk', 'fac_container',
    'fac_breakbulk', 'fac_oil_terminal', 'fac_lng_terminal', 'fac_other',
    'medical_facilities', 'garbage_disposal', 'chemical_tank_disposal',
    'dirty_ballast_disposal', 'degaussing',
    'cranes_fixed', 'cranes_mobile', 'cranes_floating', 'cranes_container',
    'lifts_100_tons', 'lifts_50_100_tons', 'lifts_25_49_tons', 'lifts_0_24_tons',
    'svc_longshoremen', 'svc_electricity', 'svc_steam', 'svc_nav_equip',
    'svc_electrical_repair', 'svc_ice_breaking', 'svc_diving',
    'sup_provisions', 'sup_potable_water', 'sup_fuel_oil', 'sup_diesel_oil',
    'sup_aviation_fuel', 'sup_deck', 'sup_engine',
    'repairs', 'dry_dock',
    'tidal_range_m', 'entrance_width_m', 'channel_depth_m', 'anchorage_depth_m',
    'cargo_pier_depth_m', 'oil_terminal_depth_m', 'lng_terminal_depth_m',
    'max_vessel_length_m', 'max_vessel_beam_m', 'max_vessel_draft_m',
    'offshore_max_vessel_length_m', 'offshore_max_vessel_beam_m', 'offshore_max_vessel_draft_m'
]

feature_cols = satellite_cols + ports_cols     # with Port features
target_col = 'log_total_ves_wgt'               # total trade (vessel) weight 

df_clean = df[feature_cols + [target_col, 'year_month', 'port_name']].dropna()

# Categorical encoding
cat_cols = df_clean[feature_cols].select_dtypes(include=['object']).columns.tolist()
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    label_encoders[col] = le

# Time-based split (70% train_full, 30% test)
df_clean = df_clean.sort_values('year_month').reset_index(drop=True)
split_idx = int(len(df_clean) * 0.7)

df_train_full = df_clean.iloc[:split_idx].copy()
df_test = df_clean.iloc[split_idx:].copy()

# Within train_full, further split into train (80%) and val (20%)
val_split_idx = int(len(df_train_full) * 0.8)
df_train = df_train_full.iloc[:val_split_idx].copy()
df_val = df_train_full.iloc[val_split_idx:].copy()

X_train = df_train[feature_cols].values
y_train = df_train[target_col].values
X_val = df_val[feature_cols].values
y_val = df_val[target_col].values
X_test = df_test[feature_cols].values
y_test = df_test[target_col].values

print(f"Train: {len(df_train)} rows ({df_train['year_month'].min()} ~ {df_train['year_month'].max()})")
print(f"Val: {len(df_val)} rows ({df_val['year_month'].min()} ~ {df_val['year_month'].max()})")
print(f"Test: {len(df_test)} rows ({df_test['year_month'].min()} ~ {df_test['year_month'].max()})")

# For Optuna: Hyperparameter optimization
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
sampler = optuna.samplers.TPESampler(seed=SEED)
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"\nBest val_rmse: {study.best_trial.value:.4f}")
print(f"Best params: {study.best_trial.params}")

# Retrain on train_full with best parameters
best = study.best_trial.params
best['random_state'] = SEED
best['n_jobs'] = -1

X_train_full = df_train_full[feature_cols].values
y_train_full = df_train_full[target_col].values

model = xgb.XGBRegressor(**best)
model.fit(X_train_full, y_train_full, verbose=False)

# Predict on the entire dataset (fitted values)
X_all = df_clean[feature_cols].values
preds_all = model.predict(X_all)

# Evaluate performance on the test set
preds_test = model.predict(X_test)
actuals_test = y_test

metrics = {
    'model': 'XGBoost',
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

print(f"\nTest Results")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")


# Save metrics
pd.DataFrame([metrics]).to_csv("watching_trade/output_tables/02_01_xgb_metrics_wgt_ports.csv", index=False)

# Save predictions with test flag
result_df = df_clean[['port_name', 'year_month']].copy()
result_df['actual'] = df_clean[target_col].values
result_df['predicted'] = preds_all
result_df['error'] = result_df['predicted'] - result_df['actual']
result_df['is_test'] = 0
result_df.loc[split_idx:, 'is_test'] = 1
result_df.to_csv("watching_trade/output_tables/02_02_xgb_predictions_wgt_ports.csv", index=False)

# Feature importance
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
importance_df.to_csv("watching_trade/output_tables/02_03_xgb_importance_wgt_ports.csv", index=False)

print(f"\nFeature Importance (Top 15):")
print(importance_df.head(15))

# Save model and encoders
joblib.dump(model, "watching_trade/output_models/01_xgb_model_wgt_ports.joblib")
joblib.dump(label_encoders, "watching_trade/output_models/01_xgb_encoders_wgt_ports.joblib")

print(f"\nCompleted.")