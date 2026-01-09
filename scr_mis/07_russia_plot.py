import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Load prediction results
result_df = pd.read_csv("watching_trade/output_tables/12_russia_predictions_by_size.csv")

# Calculate pre/post sanctions
result_df['post_sanctions'] = result_df['year_month'] >= '2022-02'
comparison = result_df.groupby(['port_name', 'post_sanctions'])['predicted_log_trade_val'].mean().unstack()
comparison.columns = ['pre', 'post']
comparison['change_pct'] = ((comparison['post'] - comparison['pre']) / comparison['pre'] * 100)

# Port coordinates + label offsets
port_info = {
    'Novorossiysk': {'coords': (44.7167, 37.7833), 'offset': (-4, -5), 'ha': 'left'},
    'Sankt-Peterburg': {'coords': (59.9333, 30.3), 'offset': (8, -4), 'ha': 'left'},
    'Kaliningrad': {'coords': (54.7, 20.5), 'offset': (2, 2), 'ha': 'left'},
    'Vyborg': {'coords': (60.7, 28.75), 'offset': (8, 4), 'ha': 'left'},
    'De Kastri': {'coords': (51.4667, 140.7833), 'offset': (-8, 4), 'ha': 'right'},
    'Aleksandrovsk -Sakhalinskiy': {'coords': (50.9, 142.15), 'offset': (-4, -1), 'ha': 'right'},
    'Baltiysk': {'coords': (54.65, 19.8833), 'offset': (2, -4), 'ha': 'left'},
    'Bukhta Nagayeva (Magadan)': {'coords': (59.5667, 150.7333), 'offset': (-5, -4), 'ha': 'right'},
    'Gavan Vysotsk': {'coords': (60.6167, 28.5667), 'offset': (2, 2), 'ha': 'left'},
    'Kholmsk': {'coords': (47.05, 142.05), 'offset': (-8, 0), 'ha': 'right'},
    'Korsakov': {'coords': (46.6333, 142.7667), 'offset': (-8, -6), 'ha': 'right'},
    'Kronshtadt': {'coords': (59.9833, 29.7667), 'offset': (10, -8), 'ha': 'left'},
    'Rostov-Na-Donu': {'coords': (47.2333, 39.7167), 'offset': (4, -3), 'ha': 'left'},
    'Sovetskaya Gavan': {'coords': (48.9667, 140.2833), 'offset': (-15, 6), 'ha': 'right'},
    'Tuapse': {'coords': (44.1, 39.0667), 'offset': (8, -4), 'ha': 'left'},
}

plot_data = []
for port, info in port_info.items():
    if port in comparison.index:
        lat, lon = info['coords']
        change = comparison.loc[port, 'change_pct']
        plot_data.append({
            'port': port,
            'lat': lat,
            'lon': lon,
            'change_pct': change,
            'offset': info['offset'],
            'ha': info['ha']
        })

plot_df = pd.DataFrame(plot_data)

lon_min, lon_max = 10, 160  
lat_min, lat_max = 15, 90   

fig = plt.figure(figsize=(12, 6))
ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())

ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND, facecolor='#e8e8e8', edgecolor='none')
ax.add_feature(cfeature.OCEAN, facecolor='#d4e5f7')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray')
ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':', color='gray')

sizes = np.abs(plot_df['change_pct']) * 70 + 70

scatter = ax.scatter(
    plot_df['lon'], plot_df['lat'],
    c=plot_df['change_pct'],
    cmap='RdBu',
    s=sizes,
    edgecolors='black',
    linewidths=0.8,
    transform=ccrs.PlateCarree(),
    vmin=-2, vmax=2,
    zorder=5
)

for _, row in plot_df.iterrows():
    short_name = row['port'].replace('Bukhta Nagayeva (Magadan)', 'Magadan') \
                            .replace('Aleksandrovsk -Sakhalinskiy', 'Aleksandrovsk') \
                            .replace('Sovetskaya Gavan', 'Sov. Gavan')
    
    ax.annotate(
        short_name,
        xy=(row['lon'], row['lat']),
        xytext=(row['lon'] + row['offset'][0], row['lat'] + row['offset'][1]),
        fontsize=9,
        ha=row['ha'],
        va='center',
        transform=ccrs.PlateCarree(),
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
        zorder=10
    )

# Colorbar
cax = inset_axes(ax, width="30%", height="2%", loc='lower center',
                 bbox_to_anchor=(0, 0.1, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
cbar = plt.colorbar(scatter, cax=cax, orientation='horizontal')
cbar.set_label('Δ Predicted Trade (%)', fontsize=8)
cbar.ax.tick_params(labelsize=9)

plt.savefig('watching_trade/output_figures/russia_sanctions_map.pdf', pad_inches=0)
plt.close()

print("Saved: russia_sanctions_map.pdf")


# Figure 2: Time Series by Port (All ports in one figure)

all_months = pd.date_range('2017-01', '2024-12', freq='MS').strftime('%Y-%m').tolist()
n_months = len(all_months)


year_ticks = {year: all_months.index(f'{year}-01') for year in range(2017, 2025)}


sanction_idx = all_months.index('2022-02')

port_order = plot_df.sort_values('change_pct')['port'].tolist()

fig, axes = plt.subplots(5, 3, figsize=(14, 16))
axes = axes.flatten()

for i, port in enumerate(port_order):
    ax = axes[i]
    
    port_data = result_df[result_df['port_name'] == port].sort_values('year_month')
    
    if len(port_data) == 0:
        ax.set_visible(False)
        continue
    
    x_vals = []
    y_vals = []
    for ym in all_months:
        row = port_data[port_data['year_month'] == ym]
        if len(row) > 0:
            x_vals.append(all_months.index(ym))
            y_vals.append(row['predicted_log_trade_val'].values[0])
    
    ax.plot(x_vals, y_vals, color='black', linewidth=1.2)
    
    # vertical line
    ax.axvline(x=sanction_idx, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
    
    change = comparison.loc[port, 'change_pct']
    color = 'red' if change < 0 else 'blue'
    ax.text(0.95, 0.95, f'{change:+.2f}%', transform=ax.transAxes, 
            fontsize=10, ha='right', va='top', color=color, fontweight='bold')
    
    ax.set_title(port, fontsize=10, fontweight='bold')
    
    if i % 3 == 0:
        ax.set_ylabel('Predicted Trade (log)', fontsize=9)
    
    ax.set_xlim(0, n_months - 1)
    ax.set_xticks(list(year_ticks.values()))
    
    if i >= 12:
        ax.set_xticklabels(list(year_ticks.keys()), fontsize=8)
    else:
        ax.set_xticklabels([])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', width=0)

for j in range(len(port_order), len(axes)):
    axes[j].set_visible(False)


plt.tight_layout()
plt.savefig('watching_trade/output_figures/russia_timeseries_all.pdf', bbox_inches='tight')
plt.close()

print("Saved: russia_timeseries_all.pdf")

print("\n=== Port Changes (sorted) ===")
print(plot_df.sort_values('change_pct')[['port', 'change_pct']].to_string(index=False))