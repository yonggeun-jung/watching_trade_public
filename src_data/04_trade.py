"""
Download US port trade data from Census API
"""
import requests
import pandas as pd
from pathlib import Path
import time

BASE_URL_IMPORTS = "https://api.census.gov/data/timeseries/intltrade/imports/porths"
BASE_URL_EXPORTS = "https://api.census.gov/data/timeseries/intltrade/exports/porths"
API_KEY = None


def fetch_port_trade(year: int, month: int, trade_type: str = "imports") -> pd.DataFrame:
    base_url = BASE_URL_IMPORTS if trade_type == "imports" else BASE_URL_EXPORTS
    
    if trade_type == "imports":
        variables = "PORT,PORT_NAME,GEN_VAL_MO,VES_VAL_MO,VES_WGT_MO,AIR_VAL_MO,AIR_WGT_MO,CNT_VAL_MO,CNT_WGT_MO"
    else:
        variables = "PORT,PORT_NAME,ALL_VAL_MO,VES_VAL_MO,VES_WGT_MO,AIR_VAL_MO,AIR_WGT_MO,CNT_VAL_MO,CNT_WGT_MO"
    
    params = {
        "get": variables,
        "YEAR": year,
        "MONTH": f"{month:02d}",
    }
    
    if API_KEY:
        params["key"] = API_KEY
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 204 or response.status_code != 200:
        return pd.DataFrame()
    
    data = response.json()
    
    if len(data) < 2:
        return pd.DataFrame()
    
    df = pd.DataFrame(data[1:], columns=data[0])
    df["year"] = year
    df["month"] = month
    df["year_month"] = f"{year}-{month:02d}"
    df["trade_type"] = trade_type
    
    return df


def download_trade_data(start_year: int, end_year: int, output_path: Path) -> pd.DataFrame:
    all_imports = []
    all_exports = []
    
    # Download imports
    print("Downloading imports...")
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            print(f"  {year}-{month:02d}")
            df = fetch_port_trade(year, month, "imports")
            if len(df) > 0:
                all_imports.append(df)
            time.sleep(0.2)
    
    # Download exports
    print("Downloading exports...")
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            print(f"  {year}-{month:02d}")
            df = fetch_port_trade(year, month, "exports")
            if len(df) > 0:
                all_exports.append(df)
            time.sleep(0.2)
    
    # Combine
    imports_df = pd.concat(all_imports, ignore_index=True) if all_imports else pd.DataFrame()
    exports_df = pd.concat(all_exports, ignore_index=True) if all_exports else pd.DataFrame()
    
    # Convert numeric columns BEFORE aggregation
    for df in [imports_df, exports_df]:
        if len(df) > 0:
            numeric_cols = [c for c in df.columns if "VAL" in c or "WGT" in c]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Aggregate imports
    imports_agg = imports_df.groupby(["PORT", "PORT_NAME", "year_month"]).agg({
        "VES_VAL_MO": "sum",
        "VES_WGT_MO": "sum",
    }).reset_index()
    imports_agg.columns = ["PORT", "PORT_NAME", "year_month", "import_ves_val", "import_ves_wgt"]
    
    # Aggregate exports
    exports_agg = exports_df.groupby(["PORT", "PORT_NAME", "year_month"]).agg({
        "VES_VAL_MO": "sum",
        "VES_WGT_MO": "sum",
    }).reset_index()
    exports_agg.columns = ["PORT", "PORT_NAME", "year_month", "export_ves_val", "export_ves_wgt"]
    
    # Merge imports and exports
    merged = pd.merge(imports_agg, exports_agg, on=["PORT", "PORT_NAME", "year_month"], how="outer")
    merged = merged.fillna(0)
    
    # Calculate totals
    merged["total_ves_val"] = merged["import_ves_val"] + merged["export_ves_val"]
    merged["total_ves_wgt"] = merged["import_ves_wgt"] + merged["export_ves_wgt"]
    
    # Save
    merged.to_csv(output_path, index=False)
    print(f"Saved: {output_path} ({len(merged)} rows)")
    
    return merged


def main():
    output_dir = Path("watching_trade/data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = download_trade_data(
        start_year=2016,
        end_year=2024,
        output_path=output_dir / "us_trade_ports.csv",
    )
    
    print(f"\nTotal records: {len(df)}")
    print(f"\nTop 10 ports by total vessel value:")
    top_ports = df.groupby("PORT_NAME")["total_ves_val"].sum().sort_values(ascending=False).head(10)
    print(top_ports)


if __name__ == "__main__":
    main()