"""
Monte Carlo Simulation: Level vs. Change in Spatial Extrapolation

Y_it = α_i + β·X_it + ε_it
ΔY = β·ΔX (α cancels) → justifies percentage-change approach
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)


def generate_data(n_ports, n_months, alpha_range, beta=1.0, sigma_eps=0.5):
    """Y_it = α_i + β*X_it + ε_it with AR(1) + seasonality in X"""
    alpha = np.random.uniform(*alpha_range, size=n_ports)
    X = np.zeros((n_ports, n_months))
    Y = np.zeros((n_ports, n_months))

    for i in range(n_ports):
        # AR(1) + trend + seasonality
        for t in range(1, n_months):
            X[i, t] = 0.7 * X[i, t-1] + np.random.normal()
        trend = np.linspace(0, np.random.uniform(-0.5, 0.5), n_months)
        season = 0.3 * np.sin(2 * np.pi * np.arange(n_months) / 12)
        X[i] = (X[i] - X[i].mean()) / X[i].std() * 2 + 5 + trend + season
        Y[i] = alpha[i] + beta * X[i] + np.random.normal(0, sigma_eps, n_months)

    return X, Y, alpha


def to_panel(X, Y, prefix="p"):
    """Arrays → long-format DataFrame"""
    rows = []
    for i in range(X.shape[0]):
        for t in range(X.shape[1]):
            rows.append({'port': f"{prefix}_{i}", 'month': t, 'X': X[i,t], 'Y': Y[i,t]})
    return pd.DataFrame(rows)


def run_level_simulation(n_sim=100):
    """Test level prediction under spatial extrapolation"""
    r2_raw, r2_anch, r2_chg = [], [], []

    for s in range(n_sim):
        X_tr, Y_tr, _ = generate_data(30, 100, (12, 18))
        X_te, Y_te, _ = generate_data(20, 100, (6, 10))
        df_tr, df_te = to_panel(X_tr, Y_tr, "tr"), to_panel(X_te, Y_te, "te")

        mdl = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=s)
        mdl.fit(df_tr[['X']], df_tr['Y'])
        df_te['Yp'] = mdl.predict(df_te[['X']])

        # Raw level
        r2_raw.append(r2_score(df_te['Y'], df_te['Yp']))

        # Anchored level
        df_a = df_te.copy()
        for p in df_a['port'].unique():
            m = df_a['port'] == p
            offset = df_a.loc[m & (df_a['month']==0), 'Y'].values[0] - \
                     df_a.loc[m & (df_a['month']==0), 'Yp'].values[0]
            df_a.loc[m, 'Yp'] += offset
        r2_anch.append(r2_score(df_a['Y'], df_a['Yp']))

        # ΔY (month-over-month)
        df_s = df_te.sort_values(['port','month'])
        dy = df_s.groupby('port')['Y'].diff().dropna()
        dyp = df_s.groupby('port')['Yp'].diff().dropna()
        r2_chg.append(r2_score(dy, dyp))

    return {'raw': r2_raw, 'anchored': r2_anch, 'change': r2_chg}


def run_prepost_simulation(n_sim=100):
    """Test pre/post level change recovery (ΔY cancels α)"""
    true_chg, pred_chg = [], []

    for s in range(n_sim):
        X_tr, Y_tr, _ = generate_data(30, 100, (12, 18))
        X_te, Y_te, alpha_te = generate_data(20, 100, (6, 10))

        # Heterogeneous shock at t=50
        shocks = np.random.uniform(-2, 2, 20)
        for i in range(20):
            X_te[i, 50:] += shocks[i]
            Y_te[i, 50:] += shocks[i]

        df_tr, df_te = to_panel(X_tr, Y_tr, "tr"), to_panel(X_te, Y_te, "te")

        mdl = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=s)
        mdl.fit(df_tr[['X']], df_tr['Y'])
        df_te['Yp'] = mdl.predict(df_te[['X']])

        for i in range(20):
            d = df_te[df_te['port'] == f"te_{i}"]
            true_chg.append(d[d['month']>=50]['Y'].mean() - d[d['month']<50]['Y'].mean())
            pred_chg.append(d[d['month']>=50]['Yp'].mean() - d[d['month']<50]['Yp'].mean())

    return np.array(true_chg), np.array(pred_chg)


def plot_two_panels(level_results, true_chg, pred_chg):
    """Two-panel figure for appendix"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel (a): R² bar chart
    labels = ['Raw Level', 'Anchored Level', 'Changes ($\\Delta Y$)']
    means = [np.mean(level_results[k]) for k in ['raw','anchored','change']]
    stds = [np.std(level_results[k]) for k in ['raw','anchored','change']]
    colors = ['#d62728', '#2ca02c', '#1f77b4']

    bars = ax1.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_ylabel('$R^2$', fontsize=12)
    ax1.set_title('(a) Prediction Performance', fontsize=12)
    ax1.set_ylim(-1.5, 1.1)
    for b, m in zip(bars, means):
        ax1.annotate(f'{m:.2f}', xy=(b.get_x()+b.get_width()/2, m),
                     xytext=(0,5), textcoords='offset points', ha='center', fontsize=10)

    # Panel (b): Pre/post scatter
    slope, intercept, r_val, _, se = stats.linregress(true_chg, pred_chg)
    lims = [min(true_chg.min(), pred_chg.min())-0.5, max(true_chg.max(), pred_chg.max())+0.5]

    ax2.scatter(true_chg, pred_chg, alpha=0.3, s=20, color='#1f77b4')
    ax2.plot(lims, lims, 'k--', alpha=0.75, lw=1.5, label='45° line')
    ax2.plot(lims, [slope*x+intercept for x in lims], 'r-', alpha=0.75, lw=1.5,
             label=f'Fitted (slope={slope:.3f})')
    ax2.set_xlim(lims); ax2.set_ylim(lims)
    ax2.set_xlabel('True Level Change ($\\Delta Y$)', fontsize=11)
    ax2.set_ylabel('Predicted Level Change', fontsize=11)
    ax2.set_title('(b) Pre/Post Change Recovery', fontsize=12)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.annotate(f'Corr: {r_val:.3f}\nSlope: {slope:.3f} (SE={se:.3f})',
                 xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10,
                 va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('watching_trade/output_figures/mc_simulation.pdf', bbox_inches='tight')
    return fig


if __name__ == "__main__":
    print("Running simulations...")
    level_results = run_level_simulation(100)
    true_chg, pred_chg = run_prepost_simulation(100)

    # Summary
    slope, _, r_val, _, se = stats.linregress(true_chg, pred_chg)
    print(f"\nLevel R²:     raw={np.mean(level_results['raw']):.3f}, "
          f"anchored={np.mean(level_results['anchored']):.3f}, "
          f"ΔY={np.mean(level_results['change']):.3f}")
    print(f"Pre/post:     corr={r_val:.3f}, slope={slope:.3f} (SE={se:.3f})")

    plot_two_panels(level_results, true_chg, pred_chg)