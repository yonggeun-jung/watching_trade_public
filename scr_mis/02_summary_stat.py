import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

df = pd.read_csv("watching_trade/data/cleaned/main.csv")
df_clean = df.dropna()

print(f"Sample size: {len(df_clean)}")
print(f"Ports: {df_clean['port_name'].nunique()}")

# Summary statistics (Omitted ports features)
summary_vars = [
    'log_total_ves_val', 'log_total_ves_wgt',
    'sar_diff_median', 'vh_median_mean', 'ntl_mean', 'ntl_std', 'lit_area_ratio'
]

summary_df = df_clean[summary_vars].describe().T[['count', 'mean', 'std', 'min', 'max']]
summary_df.columns = ['N', 'Mean', 'Std', 'Min', 'Max']
summary_df['N'] = summary_df['N'].astype(int)

print("\nSummary Statistics")
print(summary_df.round(3))

# Sace
summary_df.to_csv("watching_trade/output_tables/11_01_summary_stats.csv")
print("\nSaved: 11_01_summary_stats.csv")

# Latex table
latex = []
latex.append(r"\begin{table}[ht]")
latex.append(r"\centering")
latex.append(r"\caption{Summary Statistics}")
latex.append(r"\begin{tabular}{lccccc}")
latex.append(r"\toprule")
latex.append(r"Variable & N & Mean & Std & Min & Max \\")
latex.append(r"\midrule")

var_labels = {
    'log_total_ves_val': 'Trade Value (log)',
    'log_total_ves_wgt': 'Trade Weight (log)',
    'sar_diff_median': 'SAR VV Difference',
    'vh_median_mean': 'SAR VH Backscatter',
    'ntl_mean': 'NTL Mean',
    'ntl_std': 'NTL Std',
    'lit_area_ratio': 'Light Area Ratio'
}

for var in summary_vars:
    row = summary_df.loc[var]
    label = var_labels.get(var, var)
    latex.append(f"{label} & {int(row['N'])} & {row['Mean']:.3f} & {row['Std']:.3f} & {row['Min']:.3f} & {row['Max']:.3f} \\\\")

latex.append(r"\bottomrule")
latex.append(r"\end{tabular}")
latex.append(r"\end{table}")

with open("watching_trade/output_tables/11_02_summary_stats.tex", "w") as f:
    f.write("\n".join(latex))
print("Saved: 11_02_summary_stats.tex")

# Scatter plots
color_map = {'Large': 'red', 'Medium': 'blue', 'Small': 'green'}
point_colors = df_clean['harbor_size'].map(color_map)

satellite_vars = ['sar_diff_median', 'vh_median_mean', 'ntl_mean', 'ntl_std', 'lit_area_ratio']
y_var = 'log_total_ves_val'

fig = plt.figure(figsize=(10, 8)) 
gs = fig.add_gridspec(2, 6, wspace=0.5, hspace=0.1)

axes = []
# 3 figures in the first row
axes.append(fig.add_subplot(gs[0, 0:2]))
axes.append(fig.add_subplot(gs[0, 2:4]))
axes.append(fig.add_subplot(gs[0, 4:6]))

# Second row (2 figures, centered)
axes.append(fig.add_subplot(gs[1, 1:3]))
axes.append(fig.add_subplot(gs[1, 3:5]))


for i, x_var in enumerate(satellite_vars):
    ax = axes[i]
    ax.scatter(df_clean[x_var], df_clean[y_var], alpha=0.8, s=15, c=point_colors, edgecolors='none')

    ax.set_box_aspect(1)

    ax.set_xlabel(x_var, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    if i == 0 or i == 3:
        ax.set_ylabel(y_var, fontsize=10)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Large', markerfacecolor='red', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Medium', markerfacecolor='blue', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Small', markerfacecolor='green', markersize=8)
]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.82, 0.25), ncol=1, frameon=False)

plt.savefig('watching_trade/output_figures/raw_scatter_sat.pdf', bbox_inches='tight')

print("\nCorrelations with log_total_ves_val")
for x_var in satellite_vars:
    corr = df_clean[[x_var, y_var]].corr().iloc[0, 1]
    print(f"{x_var}: {corr:.4f}")
print("\nCompleted.")
