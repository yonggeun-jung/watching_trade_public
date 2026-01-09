import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import optuna
import joblib

SEED = 42
np.random.seed(SEED)

# Load US data for training
df = pd.read_csv("watching_trade/data/cleaned/main.csv")
df = df.sort_values(['port_name', 'year_month']).reset_index(drop=True)

# Load Russia data for prediction
df_rus = pd.read_csv("watching_trade/data/cleaned/main_rus.csv")
df_rus = df_rus[df_rus['year_month'] >= '2017-01'].copy()

satellite_cols = ['sar_diff_median', 'vh_median_mean', 'ntl_mean', 'ntl_std', 'lit_area_ratio']

ports_cols = [
    'harbor_type', 'harbor_use', 'shelter_afforded',
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

feature_cols = satellite_cols + ports_cols
target_col = 'log_total_ves_val'

# Russian ports by size (where 80%+ of data is available)
RUS_PORTS = {
    'Large': ['Novorossiysk', 'Sankt-Peterburg'],
    'Medium': ['De Kastri', 'Kaliningrad', 'Vyborg'],
    'Small': [
        'Aleksandrovsk -Sakhalinskiy', 'Baltiysk', 'Bukhta Nagayeva (Magadan)',
        'Gavan Vysotsk', 'Kholmsk', 'Korsakov', 'Kronshtadt',
        'Rostov-Na-Donu', 'Sovetskaya Gavan', 'Tuapse'
    ]
}

# Prepare US data
df_clean = df[feature_cols + [target_col, 'harbor_size', 'port_name', 'year_month']].dropna()

# Categorical encoding (fit on US data)
cat_cols = df_clean[feature_cols].select_dtypes(include=['object']).columns.tolist()
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    label_encoders[col] = le

print(f"US data: {len(df_clean)} rows, {df_clean['port_name'].nunique()} ports")


def objective(trial, X_train, y_train):
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


all_results = []

for size in ['Large', 'Medium', 'Small']:
    print(f"Processing {size} harbors")
    
    rus_ports = RUS_PORTS[size]
    
    # Filter Russian data for this size
    df_rus_size = df_rus[df_rus['port_name'].isin(rus_ports)].copy()
    df_rus_size = df_rus_size.dropna(subset=satellite_cols)
    
    if len(df_rus_size) == 0:
        print(f"No Russian data for {size}")
        continue
    
    print(f"Russian ports: {rus_ports}")
    print(f"Russian data: {len(df_rus_size)} rows")
    
    # Train on US data of same size
    df_us_size = df_clean[df_clean['harbor_size'] == size].copy()
    print(f"US training data: {len(df_us_size)} rows, {df_us_size['port_name'].nunique()} ports")
    
    if len(df_us_size) < 100:
        print(f"Insufficient US training data for {size}, skipping")
        continue
    
    X_train = df_us_size[feature_cols].values
    y_train = df_us_size[target_col].values
    
    # Hyperparameter search
    print(f"\nHyperparameter search...")
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50, show_progress_bar=True)
    
    print(f"Best CV RMSE: {study.best_trial.value:.4f}")
    
    # Train final model
    best = study.best_trial.params
    best['random_state'] = SEED
    best['n_jobs'] = -1
    
    model = xgb.XGBRegressor(**best)
    model.fit(X_train, y_train, verbose=False)
    
    # Encode Russian data
    df_rus_encoded = df_rus_size.copy()
    for col in label_encoders.keys():
        if col in df_rus_encoded.columns:
            le = label_encoders[col]
            df_rus_encoded[col] = df_rus_encoded[col].astype(str)
            known_classes = set(le.classes_)
            df_rus_encoded[col] = df_rus_encoded[col].apply(lambda x: x if x in known_classes else le.classes_[0])
            df_rus_encoded[col] = le.transform(df_rus_encoded[col])
    
    # Predict
    X_rus = df_rus_encoded[feature_cols].values
    predictions = model.predict(X_rus)
    
    # Store results
    result = df_rus_size[['port_name', 'year_month']].copy()
    result['harbor_size'] = size
    result['predicted_log_trade_val'] = predictions
    all_results.append(result)
    
    # Summary
    print(f"\n{size} Results:")
    print(result.groupby('port_name')['predicted_log_trade_val'].agg(['mean', 'std', 'count']).round(3))
    
    # Save model
    joblib.dump(model, f"watching_trade/output_models/09_russia_xgb_{size.lower()}.joblib")

# Combine all results
result_df = pd.concat(all_results, ignore_index=True)
result_df.to_csv("watching_trade/output_tables/12_russia_predictions_by_size.csv", index=False)

print("Final Results Summary:")

# Pre/post sanctions
result_df['post_sanctions'] = result_df['year_month'] >= '2022-02'

print("\nPre/Post Sanctions by Port")
comparison = result_df.groupby(['harbor_size', 'port_name', 'post_sanctions'])['predicted_log_trade_val'].mean().unstack()
comparison.columns = ['pre', 'post']
comparison['change_pct'] = ((comparison['post'] - comparison['pre']) / comparison['pre'] * 100).round(2)
print(comparison.sort_values('change_pct'))

print("\nAggregate by Size")
agg = result_df.groupby(['harbor_size', 'post_sanctions'])['predicted_log_trade_val'].mean().unstack()
agg.columns = ['pre', 'post']
agg['change_pct'] = ((agg['post'] - agg['pre']) / agg['pre'] * 100).round(2)
print(agg)

print("\nCompleted.")