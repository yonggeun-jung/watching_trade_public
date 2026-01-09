import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

df = pd.read_csv("watching_trade/data/cleaned/main.csv")

# Main satellite variables
rhs_main = "sar_diff_median + vh_median_mean + ntl_mean + ntl_std + lit_area_ratio"

# Port continuous controls
rhs_port_cont = (
    "tidal_range_m + entrance_width_m + channel_depth_m + anchorage_depth_m + "
    "cargo_pier_depth_m + oil_terminal_depth_m + lng_terminal_depth_m + "
    "max_vessel_length_m + max_vessel_beam_m + max_vessel_draft_m + "
    "offshore_max_vessel_length_m + offshore_max_vessel_beam_m + offshore_max_vessel_draft_m"
)

# Port categorical controls
rhs_port_fe = (
    "C(harbor_size) + C(harbor_type) + C(harbor_use) + C(shelter_afforded) + "
    "C(ent_restr_tide) + C(ent_restr_swell) + C(ent_restr_ice) + C(ent_restr_other) + "
    "C(overhead_limits) + C(underkeel_mgmt) + C(good_holding_ground) + C(turning_area) + "
    "C(traffic_sep_scheme) + C(vessel_traffic_svc) + C(search_rescue) + "
    "C(port_security) + C(eta_message) + "
    "C(quarantine_pratique) + C(quarantine_sanitation) + C(quarantine_other) + "
    "C(first_port_of_entry) + C(us_representative) + "
    "C(pilotage_compulsory) + C(pilotage_available) + C(pilotage_local_assist) + C(pilotage_advisable) + "
    "C(tugs_salvage) + C(tugs_assistance) + "
    "C(comm_telephone) + C(comm_telefax) + C(comm_radio) + C(comm_radiotelephone) + "
    "C(comm_airport) + C(comm_rail) + "
    "C(fac_wharves) + C(fac_anchorage) + C(fac_dangerous_cargo) + "
    "C(fac_med_mooring) + C(fac_beach_mooring) + C(fac_ice_mooring) + "
    "C(fac_roro) + C(fac_solid_bulk) + C(fac_liquid_bulk) + C(fac_container) + "
    "C(fac_breakbulk) + C(fac_oil_terminal) + C(fac_lng_terminal) + C(fac_other) + "
    "C(medical_facilities) + C(garbage_disposal) + C(chemical_tank_disposal) + "
    "C(dirty_ballast_disposal) + C(degaussing) + "
    "C(cranes_fixed) + C(cranes_mobile) + C(cranes_floating) + C(cranes_container) + "
    "C(lifts_100_tons) + C(lifts_50_100_tons) + C(lifts_25_49_tons) + C(lifts_0_24_tons) + "
    "C(svc_longshoremen) + C(svc_electricity) + C(svc_steam) + C(svc_nav_equip) + "
    "C(svc_electrical_repair) + C(svc_ice_breaking) + C(svc_diving) + "
    "C(sup_provisions) + C(sup_potable_water) + C(sup_fuel_oil) + C(sup_diesel_oil) + "
    "C(sup_aviation_fuel) + C(sup_deck) + C(sup_engine) + "
    "C(repairs) + C(dry_dock)"
)

formulas = {
    'Model1_NoFE_log_val': f"log_total_ves_val ~ {rhs_main}",
    'Model2_TimeFE_log_val': f"log_total_ves_val ~ {rhs_main} + C(year_month)",
    'Model3_AllFE_log_val': f"log_total_ves_val ~ {rhs_main} + {rhs_port_cont} + {rhs_port_fe} + C(year_month)",
    'Model4_NoFE_log_wgt': f"log_total_ves_wgt ~ {rhs_main}",
    'Model5_TimeFE_log_wgt': f"log_total_ves_wgt ~ {rhs_main} + C(year_month)",
    'Model6_AllFE_log_wgt': f"log_total_ves_wgt ~ {rhs_main} + {rhs_port_cont} + {rhs_port_fe} + C(year_month)",
}

# Drop NA
df_clean = df.dropna()

print(f"Original sample: {len(df)}")
print(f"After dropna: {len(df_clean)}")
print(f"Ports: {df_clean['port_name'].nunique()}")

results = []
for name, formula in formulas.items():
    print(f"Running {name}...")
    try:
        model = smf.ols(formula, data=df_clean).fit(
            cov_type='cluster', cov_kwds={'groups': df_clean['port_name']}
        )
        
        coef_df = model.summary2().tables[1].reset_index()
        coef_df.columns = ['variable', 'coef', 'std_err', 'z', 'p_value', 'ci_low', 'ci_high']
        coef_df['model'] = name
        coef_df['nobs'] = int(model.nobs)
        coef_df['r2'] = model.rsquared
        coef_df['r2_adj'] = model.rsquared_adj
        
        results.append(coef_df)
        print(f"  Done. R2={model.rsquared:.4f}, Adj.R2={model.rsquared_adj:.4f}")
    except Exception as e:
        print(f"  Error: {e}")

results_df = pd.concat(results, ignore_index=True)

results_df.to_csv("watching_trade/output_tables/01_01_ols_results.csv", index=False)
print(f"\nSaved: ols_results.csv")


# Latex table generation
def generate_latex_table(formulas, target_models, df_clean, title, output_filename):
    main_vars = ['sar_diff_median', 'vh_median_mean', 'ntl_mean', 'ntl_std', 'lit_area_ratio', 'Intercept']
    
    est_results = {}
    for name in target_models:
        est_results[name] = smf.ols(formulas[name], data=df_clean).fit(
            cov_type='cluster', cov_kwds={'groups': df_clean['port_name']}
        )

    num_cols = len(target_models)
    col_spec = "l" + "c" * num_cols 
    
    latex = []
    latex.append(r"\begin{table}[ht]")
    latex.append(r"\centering")
    latex.append(f"\\caption{{{title}}}")
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append(r"\toprule")
    
    # Header
    header = " & (1) & (2) & (3) & (4) & (5) & (6)" + r" \\"
    latex.append(header)
    dep_var = " & \\multicolumn{3}{c}{Log Value} & \\multicolumn{3}{c}{Log Weight}" + r" \\"
    latex.append(dep_var)
    latex.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")

    # Coefficients
    for var in main_vars:
        coef_row = [var.replace('_', r'\_')]
        exists_in_any = False
        for name in target_models:
            res = est_results[name]
            if var in res.params.index:
                coef = res.params[var]
                p_val = res.pvalues[var]
                stars = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
                coef_row.append(f"{coef:.4f}{stars}")
                exists_in_any = True
            else:
                coef_row.append("")
        
        if not exists_in_any:
            continue
            
        latex.append(" & ".join(coef_row) + r" \\")
        
        se_row = [""]
        for name in target_models:
            res = est_results[name]
            if var in res.params.index:
                se = res.bse[var]
                se_row.append(f"({se:.4f})")
            else:
                se_row.append("")
        latex.append(" & ".join(se_row) + r" \\")

    latex.append(r"\midrule")
    
    # Summary stats
    obs_row = ["Observations"]
    r2_row = ["Adj. R-squared"]
    time_row = ["Time FE"]
    port_row = ["Port Controls"]
    
    for name in target_models:
        res = est_results[name]
        obs_row.append(f"{int(res.nobs)}")
        r2_row.append(f"{res.rsquared_adj:.3f}")
        time_row.append("Yes" if "year_month" in formulas[name] else "No")
        port_row.append("Yes" if "harbor_size" in formulas[name] else "No")
    
    latex.append(" & ".join(obs_row) + r" \\")
    latex.append(" & ".join(r2_row) + r" \\")
    latex.append(" & ".join(time_row) + r" \\")
    latex.append(" & ".join(port_row) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\begin{tablenotes}")
    latex.append(r"\footnotesize")
    latex.append(r"\item Standard errors clustered by port in parentheses. *** p<0.01, ** p<0.05, * p<0.1")
    latex.append(r"\end{tablenotes}")
    latex.append(r"\end{table}")

    with open(f"watching_trade/output_tables/{output_filename}.tex", "w") as f:
        f.write("\n".join(latex))
    print(f"Saved: {output_filename}.tex")


# Generate table
models = ['Model1_NoFE_log_val', 'Model2_TimeFE_log_val', 'Model3_AllFE_log_val', 
          'Model4_NoFE_log_wgt', 'Model5_TimeFE_log_wgt', 'Model6_AllFE_log_wgt']
generate_latex_table(formulas, models, df_clean, "OLS Results: Satellite Features and Trade", "01_02_ols_table_log")

print("\nCompleted.")