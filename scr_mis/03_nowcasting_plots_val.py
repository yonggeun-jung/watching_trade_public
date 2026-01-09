
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# File paths
PRED_PORT_PATH = "watching_trade/output_tables/04_02_xgb_predictions_val_ports.csv"
PRED_NOPORT_PATH = "watching_trade/output_tables/05_02_xgb_predictions_val_noports.csv"
MAIN_DATA_PATH = "watching_trade/data/cleaned/main.csv"
OUTPUT_DIR = "watching_trade/output_figures"

# Load data
df_port = pd.read_csv(PRED_PORT_PATH)
df_noport = pd.read_csv(PRED_NOPORT_PATH)
df_main = pd.read_csv(MAIN_DATA_PATH)

# Merge predictions
df = df_port[['port_name', 'year_month', 'actual', 'predicted', 'is_test']].copy()
df = df.rename(columns={'predicted': 'predicted_port'})
df = df.merge(
    df_noport[['port_name', 'year_month', 'predicted']].rename(columns={'predicted': 'predicted_noport'}),
    on=['port_name', 'year_month'],
    how='left'
)

# Merge harbor_size from main
df = df.merge(
    df_main[['port_name', 'year_month', 'harbor_size']].drop_duplicates(),
    on=['port_name', 'year_month'],
    how='left'
)

# Time series (monthly aggregate)
monthly = df.groupby('year_month').agg({
    'actual': 'sum',
    'predicted_port': 'sum',
    'predicted_noport': 'sum',
    'is_test': 'max'
}).reset_index()

monthly = monthly.sort_values('year_month')
monthly = monthly[monthly['year_month'] >= '2017-01'].reset_index(drop=True) # from 2017-01 due to missing values issues in 2016

# Find test set start
test_start = monthly[monthly['is_test'] == 1]['year_month'].min()
test_start_idx = monthly[monthly['year_month'] == test_start].index[0]

fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(
    range(len(monthly)), monthly['actual'],
    color='blue', linewidth=2, label='Actual'
)
ax.plot(
    range(len(monthly)), monthly['predicted_port'],
    color='tab:blue', linewidth=2, linestyle='--',
    label='Port'
)
ax.plot(
    range(len(monthly)), monthly['predicted_noport'],
    color='tab:orange', linewidth=2, linestyle=':',
    label='No port'
)

# Test set start
ax.axvline(
    x=test_start_idx,
    color='red', linestyle='--', linewidth=1.5, alpha=1
)

# X ticks (yearly)
tick_positions = range(0, len(monthly), 12)
tick_labels = [monthly.iloc[i]['year_month'] for i in tick_positions]
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)

ax.set_ylabel('Log Trade Value')

# Legend without box
ax.legend(frameon=False, loc='upper left')

# Axis styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)
ax.tick_params(axis='both', width=1)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/timeseries_agg_val.pdf', bbox_inches='tight')
plt.close()


# Percent residuals
monthly['resid_pct_port'] = (
    (monthly['actual'] - monthly['predicted_port']) / monthly['actual'] * 100
)
monthly['resid_pct_noport'] = (
    (monthly['actual'] - monthly['predicted_noport']) / monthly['actual'] * 100
)

fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(
    range(len(monthly)), monthly['resid_pct_port'],
    color='tab:blue', linewidth=2,
    label='Residual % (with Port Features)'
)
ax.plot(
    range(len(monthly)), monthly['resid_pct_noport'],
    color='tab:orange', linewidth=2, linestyle='--',
    label='Residual % (without Port Features)'
)

# Zero line
ax.axhline(0, color='black', linewidth=1)

# Test split
ax.axvline(
    x=test_start_idx,
    color='red', linestyle='--', linewidth=1.5, alpha=1
)

# X ticks
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)

ax.set_ylabel('Prediction Error (%)')

# Legend without box
ax.legend(frameon=False, loc='upper left')

# Axis styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)
ax.tick_params(axis='both', width=1)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/timeseries_resid_agg_val.pdf', bbox_inches='tight')
plt.close()


# Scatter plot (Actual vs Predicted with Port)
fig, ax = plt.subplots(figsize=(5, 5))

ax.scatter(df['actual'], df['predicted_port'], alpha=0.5, s=20, c='black')

# 45-degree line
lims = [min(df['actual'].min(), df['predicted_port'].min()),
        max(df['actual'].max(), df['predicted_port'].max())]
ax.plot(lims, lims, 'r--', linewidth=2, alpha=0.9)


ax.set_xlabel('actual trade value (log)')
ax.set_ylabel('predicted trade value (log)')
ax.set_aspect('equal', adjustable='box')
ax.tick_params(axis='both', width=0)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/pred_scatter_port_val.pdf', bbox_inches='tight')


# Scatter plot (Actual vs Predicted without Port)
fig, ax = plt.subplots(figsize=(5, 5))

ax.scatter(df['actual'], df['predicted_noport'], alpha=0.5, s=20, c='black')

# 45-degree line
lims = [min(df['actual'].min(), df['predicted_noport'].min()),
        max(df['actual'].max(), df['predicted_noport'].max())]
ax.plot(lims, lims, 'r--', linewidth=2, alpha=0.9)


ax.set_xlabel('actual trade value (log)')
ax.set_ylabel('predicted trade value (log)')
ax.set_aspect('equal', adjustable='box')
ax.tick_params(axis='both', width=0)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/pred_scatter_noport_val.pdf', bbox_inches='tight')

# Time series by Harbor Size
harbor_sizes = ['Large', 'Medium', 'Small']

fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)

metrics_by_size = []

for i, size in enumerate(harbor_sizes):
    ax = axes[i]
    df_size = df[df['harbor_size'] == size].copy()
    
    if len(df_size) == 0:
        ax.set_title(f'{size} Harbors (No Data)')
        continue
    
    monthly_size = df_size.groupby('year_month').agg({
        'actual': 'sum',
        'predicted_port': 'sum',
        'predicted_noport': 'sum',
        'is_test': 'max'
    }).reset_index().sort_values('year_month')
    monthly_size = monthly_size[monthly_size['year_month'] >= '2017-01'].reset_index(drop=True)  # from 2017-01 due to missing values issues in 2016
    
    ax.plot(range(len(monthly_size)), monthly_size['actual'], color='blue', linewidth=1.5, label='actual')
    ax.plot(range(len(monthly_size)), monthly_size['predicted_port'], color='tab:blue', linestyle='--', linewidth=1.5, label='predicted (w/ port features)')
    ax.plot(range(len(monthly_size)), monthly_size['predicted_noport'], color='tab:orange', linestyle=':', linewidth=1.5, label='predicted (w/o port features)')
    
    # Vertical line at test set start
    test_start_size = monthly_size[monthly_size['is_test'] == 1]['year_month'].min()
    if pd.notna(test_start_size):
        test_idx = monthly_size[monthly_size['year_month'] == test_start_size].index[0]
        test_pos = list(monthly_size.index).index(test_idx)
        ax.axvline(x=test_pos, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # X-axis labels: show only YYYY, no rotation
    tick_positions = range(0, len(monthly_size), 12)
    tick_labels = [monthly_size.iloc[j]['year_month'][:4] for j in tick_positions]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0)

    ax.set_ylabel('trade value (log)')

    # Legend without box
    ax.legend(loc='lower right', fontsize=6, frameon=False)

    ax.tick_params(axis='both', width=0)
    
    ax.text(
    0.02, 0.95,
    f'{size} ports',
    transform=ax.transAxes,
    fontsize=8,
    va='top',
    ha='left'
)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/timeseries_by_size_val.pdf', bbox_inches='tight')


print("\nCompleted.")