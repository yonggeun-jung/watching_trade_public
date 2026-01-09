import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# File paths
PRED_PORT_PATH = "watching_trade/output_tables/06_02_xgb_predictions_LOO_wgt_ports.csv"
PRED_NOPORT_PATH = "watching_trade/output_tables/07_02_xgb_predictions_LOO_wgt_noports.csv"
OUTPUT_DIR = "watching_trade/output_figures"


# Load data
df_port = pd.read_csv(PRED_PORT_PATH)
df_noport = pd.read_csv(PRED_NOPORT_PATH)

# Process each port
ports = df_port['port_name'].unique()

for port in ports:
    port_data_port = df_port[df_port['port_name'] == port].sort_values('year_month').reset_index(drop=True)
    port_data_noport = df_noport[df_noport['port_name'] == port].sort_values('year_month').reset_index(drop=True)
    
    # Safe filename
    port_filename = port.replace(' ', '_').replace('/', '_')
    
    # Time series (actual + adjusted from port & noport)
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.plot(range(len(port_data_port)), port_data_port['actual'], 
            color='blue', linewidth=2, label='actual')
    ax.plot(range(len(port_data_port)), port_data_port['pred_adjusted'], 
            color='tab:blue', linewidth=2, linestyle='--', label='predicted (w/ port features)')
    ax.plot(range(len(port_data_noport)), port_data_noport['pred_adjusted'], 
            color='tab:orange', linewidth=2, linestyle=':', label='predicted (w/o port features)')
    
    # X ticks (yearly)
    years = port_data_port['year_month'].str[:4].unique()
    tick_positions = [port_data_port[port_data_port['year_month'].str.startswith(y)].index[0] for y in years]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(years)
    
    ax.set_ylabel('trade value (log)')
    ax.legend(frameon=False, loc='best')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(axis='both', width=0)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/LOO_timeseries_wgt_{port_filename}.pdf', bbox_inches='tight')
    plt.close()
    
    # Error time series (with port features)
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.plot(range(len(port_data_port)), port_data_port['error_raw'], 
            color='tab:blue', linewidth=2, label='error (raw)')
    ax.plot(range(len(port_data_port)), port_data_port['error_adj'], 
            color='tab:orange', linewidth=2, linestyle='--', label='error (adjusted)')
    ax.axhline(0, color='black', linewidth=1)
    
    # X ticks (yearly)
    years = port_data_port['year_month'].str[:4].unique()
    tick_positions = [port_data_port[port_data_port['year_month'].str.startswith(y)].index[0] for y in years]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(years)
    
    ax.set_ylabel('prediction error (%)')
    ax.legend(frameon=False, loc='best')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(axis='both', width=0)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/LOO_error_port_wgt_{port_filename}.pdf', bbox_inches='tight')
    plt.close()
    

    # Error time series (without port features)
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.plot(range(len(port_data_noport)), port_data_noport['error_raw'], 
            color='tab:blue', linewidth=2, label='error (raw)')
    ax.plot(range(len(port_data_noport)), port_data_noport['error_adj'], 
            color='tab:orange', linewidth=2, linestyle='--', label='error (adjusted)')
    ax.axhline(0, color='black', linewidth=1)
    
    # X ticks (yearly)
    years = port_data_noport['year_month'].str[:4].unique()
    tick_positions = [port_data_noport[port_data_noport['year_month'].str.startswith(y)].index[0] for y in years]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(years)
    
    ax.set_ylabel('prediction error (%)')
    ax.legend(frameon=False, loc='best')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(axis='both', width=0)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/LOO_error_noport_wgt_{port_filename}.pdf', bbox_inches='tight')
    plt.close()
    
   
print("\nCompleted.")