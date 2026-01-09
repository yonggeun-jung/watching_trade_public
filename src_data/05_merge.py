"""
Merge SAR, NTL, Trade, and WPI features by year_month and wpi_id
"""
import numpy as np
import pandas as pd
from pathlib import Path

# Census PORT -> WPI wpi_id mapping
PORT_TO_WPI_ID = {
    # California
    "2704": 16080.0,  # LOS ANGELES -> Los Angeles
    "2709": 16070.0,  # LONG BEACH -> Long Beach
    "2811": 16340.0,  # OAKLAND -> Oakland
    "2809": 16300.0,  # SAN FRANCISCO -> San Francisco
    "2501": 16010.0,  # SAN DIEGO -> San Diego
    "2810": 16520.0,  # STOCKTON -> Stockton
    "2812": 16370.0,  # RICHMOND -> Point Richmond

    # Texas
    "5301": 9240.0,   # HOUSTON -> Houston
    "5310": 9150.0,   # GALVESTON -> Galveston
    "5312": 9300.0,   # CORPUS CHRISTI -> Corpus Christi
    "2104": 9140.0,   # BEAUMONT -> Beaumont
    "2101": 9080.0,   # PORT ARTHUR -> Port Arthur
    "2301": 9340.0,   # BROWNSVILLE -> Brownsville

    # NY / NJ / PA / DE / CT / RI / MA / ME
    "1001": 7640.0,   # NEW YORK -> New York City
    "1003": 7810.0,   # NEWARK -> Newark
    "1002": 7720.0,   # ALBANY -> Albany
    "1102": 8080.0,   # CHESTER -> Chester
    "1101": 8110.0,   # PHILADELPHIA -> Philadelphia
    "1103": 8050.0,   # WILMINGTON, DE -> Wilmington
    "0502": 7420.0,   # PROVIDENCE -> Providence
    "0401": 7250.0,   # BOSTON -> Boston
    "0101": 7150.0,   # PORTLAND, ME -> Portland
    "0131": 7180.0,   # PORTSMOUTH -> Portsmouth
    "0152": 6910.0,   # SEARSPORT -> Searsport

    # Maryland / Virginia / Carolinas / Georgia
    "1303": 8210.0,   # BALTIMORE -> Baltimore
    "1401": 8280.0,   # NORFOLK -> Norfolk
    "1703": 8530.0,   # SAVANNAH -> Savannah
    "1601": 8500.0,   # CHARLESTON, SC -> Charleston
    "1501": 8470.0,   # WILMINGTON, NC -> Wilmington
    "1701": 8550.0,   # BRUNSWICK -> Brunswick
    "1511": 8460.0,   # MOREHEAD CITY -> Wrightsville

    # Florida (Go Gators!) / Gulf Coast
    "1801": 8670.0,   # TAMPA -> Tampa
    "1803": 8580.0,   # JACKSONVILLE -> Jacksonville
    "5203": 8630.0,   # PORT EVERGLADES -> Port Everglades
    "5201": 8640.0,   # MIAMI -> Miami
    "1818": 8750.0,   # PANAMA CITY -> Panama City
    "1819": 8760.0,   # PENSACOLA -> Pensacola
    "1821": 8667.0,   # PORT MANATEE -> Port Manatee
    "5202": 8660.0,   # KEY WEST -> Key West
    "1901": 8770.0,   # MOBILE -> Mobile
    "1902": 8800.0,   # GULFPORT -> Gulfport
    "1903": 8780.0,   # PASCAGOULA -> Pascagoula
    "2002": 8860.0,   # NEW ORLEANS -> New Orleans
    "2004": 8970.0,   # BATON ROUGE -> Baton Rouge
    "2010": 8895.0,   # GRAMERCY -> Convent
    "2001": 8980.0,   # MORGAN CITY -> Loop Terminal

    # Pacific Northwest
    "3001": 17730.0,  # SEATTLE -> Seattle
    "3002": 17700.0,  # TACOMA -> Tacoma
    "2904": 16940.0,  # PORTLAND, OR -> Portland
    "2908": 16950.0,  # VANCOUVER -> Vancouver
    "2905": 16900.0,  # LONGVIEW -> Longview
    "2903": 16770.0,  # COOS BAY -> Coos Bay
    "3003": 17060.0,  # ABERDEEN -> Aberdeen
    "3005": 18050.0,  # BELLINGHAM -> Bellingham
    "3006": 17790.0,  # EVERETT -> Everett
    "3010": 18040.0,  # ANACORTES -> Anacortes

    # Great Lakes
    "3901": 4800.0,   # CHICAGO -> Chicago
    "3801": 3620.0,   # DETROIT -> Detroit
    "4101": 3490.0,   # CLEVELAND -> Cleveland
    "4105": 3560.0,   # TOLEDO -> Toledo
    "0901": 3430.0,   # BUFFALO -> Buffalo
    "3510": 5460.0,   # DULUTH -> Duluth
    "3701": 4860.0,   # MILWAUKEE -> Milwaukee
    "4106": 3450.0,   # ERIE -> Erie

    # Hawaii / Alaska 
    "3201": 56280.0,  # HONOLULU -> Honolulu
    "3202": 56090.0,  # HILO -> Hilo
    "3102": 19130.0,  # KETCHIKAN -> Ketchikan
    "3126": 19800.0,  # ANCHORAGE -> Anchorage
}

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


def load_trade_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["PORT"] = df["PORT"].astype(str).str.strip()
    df = df[~df["PORT_NAME"].str.contains("TOTAL|DISTRICT|LOW VALUE|MAIL|VESSELS UNDER", case=False, na=False)]
    df = df[df["PORT"] != "-"]
    df = df[df["PORT"] != "nan"]
    df["wpi_id"] = df["PORT"].map(PORT_TO_WPI_ID)
    return df


def load_wpi_features(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df_subset = df[WPI_FEATURES].copy()
    df_subset = df_subset.rename(columns={'World Port Index Number': 'wpi_id'})
    return df_subset


def check_mapping_coverage(trade_df: pd.DataFrame, sar_df: pd.DataFrame):
    sar_wpi_ids = set(sar_df["wpi_id"].unique())
    trade_wpi_ids = set(trade_df["wpi_id"].dropna().unique())
    missing_trade = sar_wpi_ids - trade_wpi_ids
    
    print(f"\nMapping summary:")
    print(f"  SAR ports: {len(sar_wpi_ids)}")
    print(f"  Trade ports mapped: {len(trade_wpi_ids)}")
    print(f"  SAR ports without trade data: {len(missing_trade)}")
    
    if missing_trade:
        missing_names = sar_df[sar_df["wpi_id"].isin(missing_trade)][["port_name", "wpi_id"]].drop_duplicates()
        print(f"\n  Missing trade data for:")
        for _, row in missing_names.iterrows():
            print(f"    {row['port_name']} (wpi_id: {row['wpi_id']})")


def merge_all_data(
    sar_path: str,
    ntl_path: str,
    trade_path: str,
    wpi_path: str,
    output_path: str,
):
    print("Loading SAR data...")
    sar_df = load_sar_data(sar_path)
    print(f"  {len(sar_df)} rows, {sar_df['port_name'].nunique()} ports")
    
    print("Loading NTL data...")
    ntl_df = load_ntl_data(ntl_path)
    print(f"  {len(ntl_df)} rows, {ntl_df['port_name'].nunique()} ports")
    
    print("Loading Trade data...")
    trade_df = load_trade_data(trade_path)
    trade_mapped = trade_df[trade_df["wpi_id"].notna()].copy()
    print(f"  {len(trade_mapped)} rows mapped, {trade_mapped['wpi_id'].nunique()} ports")
    
    print("Loading WPI features...")
    wpi_df = load_wpi_features(wpi_path)
    print(f"  {len(wpi_df)} ports, {len(WPI_FEATURES)-1} features")
    
    check_mapping_coverage(trade_df, sar_df)
    
    # Merge SAR and NTL
    print("\nMerging SAR and NTL...")
    satellite_df = pd.merge(
        sar_df,
        ntl_df[["wpi_id", "year_month", "n_ntl_images", "ntl_sum", "ntl_mean", "ntl_max", "ntl_std", "lit_area_ratio"]],
        on=["wpi_id", "year_month"],
        how="outer",
    )
    print(f"  {len(satellite_df)} rows")
    
    # Merge with Trade
    print("Merging with Trade data...")
    merged_df = pd.merge(
        satellite_df,
        trade_mapped[["wpi_id", "year_month", "import_ves_val", "import_ves_wgt", 
                      "export_ves_val", "export_ves_wgt", "total_ves_val", "total_ves_wgt"]],
        on=["wpi_id", "year_month"],
        how="inner",
    )
    print(f"  {len(merged_df)} rows after merge")
    
    # Merge with WPI features
    print("Merging with WPI features...")
    final_df = pd.merge(merged_df, wpi_df, on="wpi_id", how="left")
    print(f"  {len(final_df)} rows, {final_df.shape[1]} columns")
    
    # Rename columns
    print("\nRenaming columns...")
    final_df = final_df.rename(columns=RENAME_MAP)
    
    # Log transformations
    print("Adding log transformations...")
    final_df['log_total_ves_val'] = np.log1p(final_df['total_ves_val'])
    final_df['log_total_ves_wgt'] = np.log1p(final_df['total_ves_wgt'])
    final_df['log_import_ves_val'] = np.log1p(final_df['import_ves_val'])
    final_df['log_import_ves_wgt'] = np.log1p(final_df['import_ves_wgt'])
    final_df['log_export_ves_val'] = np.log1p(final_df['export_ves_val'])
    final_df['log_export_ves_wgt'] = np.log1p(final_df['export_ves_wgt'])
    
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
    sar_path = "watching_trade/data/raw/sar_ports.csv"
    ntl_path = "watching_trade/data/raw/viirs_ports.csv"
    trade_path = "watching_trade/data/raw/us_trade_ports.csv"
    wpi_path = "watching_trade/data/raw/target_ports.csv"
    output_path = "watching_trade/data/cleaned/main.csv"
    
    for p in [sar_path, ntl_path, trade_path, wpi_path]:
        if not Path(p).exists():
            print(f"Missing: {p}")
            return
    
    df = merge_all_data(sar_path, ntl_path, trade_path, wpi_path, output_path)
    
    print("\nSample rows:")
    print(df.head(5).to_string())


if __name__ == "__main__":
    main()