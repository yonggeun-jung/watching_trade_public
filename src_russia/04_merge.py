"""
Merge SAR, NTL, and WPI features by year_month and wpi_id
"""
import numpy as np
import pandas as pd
from pathlib import Path

# WPI features
WPI_FEATURES = [
    'World Port Index Number',
    'Region Name',
    'Tidal Range (m)',
    'Entrance Width (m)',
    'Channel Depth (m)',
    'Anchorage Depth (m)',
    'Cargo Pier Depth (m)',
    'Oil Terminal Depth (m)',
    'Liquified Natural Gas Terminal Depth (m)',
    'Maximum Vessel Length (m)',
    'Maximum Vessel Beam (m)',
    'Maximum Vessel Draft (m)',
    'Offshore Maximum Vessel Length (m)',
    'Offshore Maximum Vessel Beam (m)',
    'Offshore Maximum Vessel Draft (m)',
    'Harbor Size',
    'Harbor Type',
    'Harbor Use',
    'Shelter Afforded',
    'Entrance Restriction - Tide',
    'Entrance Restriction - Heavy Swell',
    'Entrance Restriction - Ice',
    'Entrance Restriction - Other',
    'Overhead Limits',
    'Underkeel Clearance Management System',
    'Good Holding Ground',
    'Turning Area',
    'Traffic Separation Scheme',
    'Vessel Traffic Service',
    'NAVAREA',
    'Search and Rescue',
    'Port Security',
    'Estimated Time of Arrival Message',
    'Quarantine - Pratique',
    'Quarantine - Sanitation',
    'Quarantine - Other',
    'First Port of Entry',
    'US Representative',
    'Pilotage - Compulsory',
    'Pilotage - Available',
    'Pilotage - Local Assistance',
    'Pilotage - Advisable',
    'Tugs - Salvage',
    'Tugs - Assistance',
    'Communications - Telephone',
    'Communications - Telefax',
    'Communications - Radio',
    'Communications - Radiotelephone',
    'Communications - Airport',
    'Communications - Rail',
    'Facilities - Wharves',
    'Facilities - Anchorage',
    'Facilities - Dangerous Cargo Anchorage',
    'Facilities - Med Mooring',
    'Facilities - Beach Mooring',
    'Facilities - Ice Mooring',
    'Facilities - Ro-Ro',
    'Facilities - Solid Bulk',
    'Facilities - Liquid Bulk',
    'Facilities - Container',
    'Facilities - Breakbulk',
    'Facilities - Oil Terminal',
    'Facilities - LNG Terminal',
    'Facilities - Other',
    'Medical Facilities',
    'Garbage Disposal',
    'Chemical Holding Tank Disposal',
    'Dirty Ballast Disposal',
    'Degaussing',
    'Cranes - Fixed',
    'Cranes - Mobile',
    'Cranes - Floating',
    'Cranes Container',
    'Lifts - 100+ Tons',
    'Lifts - 50-100 Tons',
    'Lifts - 25-49 Tons',
    'Lifts - 0-24 Tons',
    'Services - Longshoremen',
    'Services - Electricity',
    'Services -Steam',
    'Services - Navigation Equipment',
    'Services - Electrical Repair',
    'Services - Ice Breaking',
    'Services -Diving',
    'Supplies - Provisions',
    'Supplies - Potable Water',
    'Supplies - Fuel Oil',
    'Supplies - Diesel Oil',
    'Supplies - Aviation Fuel',
    'Supplies - Deck',
    'Supplies - Engine',
    'Repairs',
    'Dry Dock',
]

RENAME_MAP = {
    'Region Name': 'region',
    'Tidal Range (m)': 'tidal_range_m',
    'Entrance Width (m)': 'entrance_width_m',
    'Channel Depth (m)': 'channel_depth_m',
    'Anchorage Depth (m)': 'anchorage_depth_m',
    'Cargo Pier Depth (m)': 'cargo_pier_depth_m',
    'Oil Terminal Depth (m)': 'oil_terminal_depth_m',
    'Liquified Natural Gas Terminal Depth (m)': 'lng_terminal_depth_m',
    'Maximum Vessel Length (m)': 'max_vessel_length_m',
    'Maximum Vessel Beam (m)': 'max_vessel_beam_m',
    'Maximum Vessel Draft (m)': 'max_vessel_draft_m',
    'Offshore Maximum Vessel Length (m)': 'offshore_max_vessel_length_m',
    'Offshore Maximum Vessel Beam (m)': 'offshore_max_vessel_beam_m',
    'Offshore Maximum Vessel Draft (m)': 'offshore_max_vessel_draft_m',
    'Harbor Size': 'harbor_size',
    'Harbor Type': 'harbor_type',
    'Harbor Use': 'harbor_use',
    'Shelter Afforded': 'shelter_afforded',
    'Entrance Restriction - Tide': 'ent_restr_tide',
    'Entrance Restriction - Heavy Swell': 'ent_restr_swell',
    'Entrance Restriction - Ice': 'ent_restr_ice',
    'Entrance Restriction - Other': 'ent_restr_other',
    'Overhead Limits': 'overhead_limits',
    'Underkeel Clearance Management System': 'underkeel_mgmt',
    'Good Holding Ground': 'good_holding_ground',
    'Turning Area': 'turning_area',
    'Traffic Separation Scheme': 'traffic_sep_scheme',
    'Vessel Traffic Service': 'vessel_traffic_svc',
    'NAVAREA': 'navarea',
    'Search and Rescue': 'search_rescue',
    'Port Security': 'port_security',
    'Estimated Time of Arrival Message': 'eta_message',
    'Quarantine - Pratique': 'quarantine_pratique',
    'Quarantine - Sanitation': 'quarantine_sanitation',
    'Quarantine - Other': 'quarantine_other',
    'First Port of Entry': 'first_port_of_entry',
    'US Representative': 'us_representative',
    'Pilotage - Compulsory': 'pilotage_compulsory',
    'Pilotage - Available': 'pilotage_available',
    'Pilotage - Local Assistance': 'pilotage_local_assist',
    'Pilotage - Advisable': 'pilotage_advisable',
    'Tugs - Salvage': 'tugs_salvage',
    'Tugs - Assistance': 'tugs_assistance',
    'Communications - Telephone': 'comm_telephone',
    'Communications - Telefax': 'comm_telefax',
    'Communications - Radio': 'comm_radio',
    'Communications - Radiotelephone': 'comm_radiotelephone',
    'Communications - Airport': 'comm_airport',
    'Communications - Rail': 'comm_rail',
    'Facilities - Wharves': 'fac_wharves',
    'Facilities - Anchorage': 'fac_anchorage',
    'Facilities - Dangerous Cargo Anchorage': 'fac_dangerous_cargo',
    'Facilities - Med Mooring': 'fac_med_mooring',
    'Facilities - Beach Mooring': 'fac_beach_mooring',
    'Facilities - Ice Mooring': 'fac_ice_mooring',
    'Facilities - Ro-Ro': 'fac_roro',
    'Facilities - Solid Bulk': 'fac_solid_bulk',
    'Facilities - Liquid Bulk': 'fac_liquid_bulk',
    'Facilities - Container': 'fac_container',
    'Facilities - Breakbulk': 'fac_breakbulk',
    'Facilities - Oil Terminal': 'fac_oil_terminal',
    'Facilities - LNG Terminal': 'fac_lng_terminal',
    'Facilities - Other': 'fac_other',
    'Medical Facilities': 'medical_facilities',
    'Garbage Disposal': 'garbage_disposal',
    'Chemical Holding Tank Disposal': 'chemical_tank_disposal',
    'Dirty Ballast Disposal': 'dirty_ballast_disposal',
    'Degaussing': 'degaussing',
    'Cranes - Fixed': 'cranes_fixed',
    'Cranes - Mobile': 'cranes_mobile',
    'Cranes - Floating': 'cranes_floating',
    'Cranes Container': 'cranes_container',
    'Lifts - 100+ Tons': 'lifts_100_tons',
    'Lifts - 50-100 Tons': 'lifts_50_100_tons',
    'Lifts - 25-49 Tons': 'lifts_25_49_tons',
    'Lifts - 0-24 Tons': 'lifts_0_24_tons',
    'Services - Longshoremen': 'svc_longshoremen',
    'Services - Electricity': 'svc_electricity',
    'Services -Steam': 'svc_steam',
    'Services - Navigation Equipment': 'svc_nav_equip',
    'Services - Electrical Repair': 'svc_electrical_repair',
    'Services - Ice Breaking': 'svc_ice_breaking',
    'Services -Diving': 'svc_diving',
    'Supplies - Provisions': 'sup_provisions',
    'Supplies - Potable Water': 'sup_potable_water',
    'Supplies - Fuel Oil': 'sup_fuel_oil',
    'Supplies - Diesel Oil': 'sup_diesel_oil',
    'Supplies - Aviation Fuel': 'sup_aviation_fuel',
    'Supplies - Deck': 'sup_deck',
    'Supplies - Engine': 'sup_engine',
    'Repairs': 'repairs',
    'Dry Dock': 'dry_dock',
}


def load_sar_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def load_ntl_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def load_wpi_features(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df_subset = df[WPI_FEATURES].copy()
    df_subset = df_subset.rename(columns={'World Port Index Number': 'wpi_id'})
    return df_subset

def merge_all_data(
    sar_path: str,
    ntl_path: str,
    wpi_path: str,
    output_path: str,
):
    print("Loading SAR data...")
    sar_df = load_sar_data(sar_path)
    print(f"  {len(sar_df)} rows, {sar_df['port_name'].nunique()} ports")
    
    print("Loading NTL data...")
    ntl_df = load_ntl_data(ntl_path)
    print(f"  {len(ntl_df)} rows, {ntl_df['port_name'].nunique()} ports")

    
    print("Loading WPI features...")
    wpi_df = load_wpi_features(wpi_path)
    print(f"  {len(wpi_df)} ports, {len(WPI_FEATURES)-1} features")
    
    
    # Merge SAR and NTL
    print("\nMerging SAR and NTL...")
    satellite_df = pd.merge(
        sar_df,
        ntl_df[["wpi_id", "year_month", "n_ntl_images", "ntl_sum", "ntl_mean", "ntl_max", "ntl_std", "lit_area_ratio"]],
        on=["wpi_id", "year_month"],
        how="outer",
    )
    print(f"  {len(satellite_df)} rows")
    
    # Merge with WPI features
    print("Merging with WPI features...")
    final_df = pd.merge(satellite_df, wpi_df, on="wpi_id", how="left")
    print(f"  {len(final_df)} rows, {final_df.shape[1]} columns")
    
    # Rename columns
    print("\nRenaming columns...")
    final_df = final_df.rename(columns=RENAME_MAP)
        
    # Summary
    print(f"\nFinal dataset:")
    print(f"  Ports: {final_df['port_name'].nunique()}")
    print(f"  Date range: {final_df['year_month'].min()} to {final_df['year_month'].max()}")
    print(f"  Columns ({final_df.shape[1]}): {list(final_df.columns)}")
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    
    return final_df


def main():
    sar_path = "watching_trade/data/raw/sar_ports_rus.csv"
    ntl_path = "watching_trade/data/raw/viirs_ports_rus.csv"
    wpi_path = "watching_trade/data/raw/target_ports_rus.csv"
    output_path = "watching_trade/data/cleaned/main_rus.csv"
    
    for p in [sar_path, ntl_path, wpi_path]:
        if not Path(p).exists():
            print(f"Missing: {p}")
            return
    
    df = merge_all_data(sar_path, ntl_path, wpi_path, output_path)
    
    print("\nSample rows:")
    print(df.head(5).to_string())


if __name__ == "__main__":
    main()