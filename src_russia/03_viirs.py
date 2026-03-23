"""
Extract Nighttime Light (VIIRS) features
"""

import ee
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
import argparse
import time


GEE_PROJECT = "GET-YOUR-GEE-PROJECT-ID"       # Your GEE project ID (https://developers.google.com/earth-engine/guides/quickstart_python)
BUFFER_RADIUS_M = 3000                        # AOI buffer radius in meters
VIIRS_SCALE = 500                             # VIIRS spatial resolution in meters
VIIRS_COLLECTION = "NASA/VIIRS/002/VNP46A2"
VIIRS_BAND = "DNB_BRDF_Corrected_NTL"


# Helper functions
def init_gee():
    try:
        ee.Initialize(project=GEE_PROJECT)
    except:
        print("Run ee.Authenticate() first")
        raise


def create_aoi(lat, lon, buffer_m=BUFFER_RADIUS_M):
    return ee.Geometry.Point([lon, lat]).buffer(buffer_m).bounds()


def get_viirs_collection(aoi, start_date, end_date):
    return (
        ee.ImageCollection(VIIRS_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .select(VIIRS_BAND)
    )


def compute_monthly_ntl_features(aoi, year, month, lit_threshold=0.5):
    start = f"{year}-{month:02d}-01"
    end = (datetime(year, month, 1) + relativedelta(months=1)).strftime("%Y-%m-%d")

    collection = get_viirs_collection(aoi, start, end)
    n_images = collection.size().getInfo()

    result = {
        "n_ntl_images": n_images,
        "ntl_sum": None,
        "ntl_mean": None,
        "ntl_max": None,
        "ntl_std": None,
        "lit_area_ratio": None,
    }

    if n_images == 0:
        return result

    composite = collection.median().max(ee.Image.constant(0))

    stats = composite.reduceRegion(
        reducer=(
            ee.Reducer.sum()
            .combine(ee.Reducer.mean(), sharedInputs=True)
            .combine(ee.Reducer.max(), sharedInputs=True)
            .combine(ee.Reducer.stdDev(), sharedInputs=True)
        ),
        geometry=aoi,
        scale=VIIRS_SCALE,
        maxPixels=1e10,
    ).getInfo()


    result["ntl_sum"] = stats.get(f"{VIIRS_BAND}_sum")
    result["ntl_mean"] = stats.get(f"{VIIRS_BAND}_mean")
    result["ntl_max"] = stats.get(f"{VIIRS_BAND}_max")
    result["ntl_std"] = stats.get(f"{VIIRS_BAND}_stdDev")

    lit_mask = composite.gt(lit_threshold)
    counts = lit_mask.reduceRegion(
        reducer=ee.Reducer.sum().combine(ee.Reducer.count(), sharedInputs=True),
        geometry=aoi,
        scale=VIIRS_SCALE,
        maxPixels=1e10,
    ).getInfo()

    lit = counts.get(f"{VIIRS_BAND}_sum", 0)
    total = counts.get(f"{VIIRS_BAND}_count", 1)

    if total > 0:
        result["lit_area_ratio"] = lit / total

    return result


def extract_all_ports_ntl(ports_df, start_month, end_month, output_path):
    start_dt = datetime.strptime(start_month, "%Y-%m")
    end_dt = datetime.strptime(end_month, "%Y-%m")

    all_records = []

    for idx, port in ports_df.iterrows():
        print(f"[{idx+1}/{len(ports_df)}] {port['Country Code']} - {port['Main Port Name']}")

        aoi = create_aoi(port["Latitude"], port["Longitude"])
        current = start_dt

        while current <= end_dt:
            year, month = current.year, current.month
            ym = current.strftime("%Y-%m")

            try:
                feats = compute_monthly_ntl_features(aoi, year, month)

                record = {
                    "country": port["Country Code"],
                    "port_name": port["Main Port Name"],
                    "wpi_id": port["World Port Index Number"],
                    "lat": port["Latitude"],
                    "lon": port["Longitude"],
                    "year_month": ym,
                    **feats,
                }
                all_records.append(record)

            except Exception as e:
                print(f"  Error {ym}: {e}")

            current += relativedelta(months=1)
            time.sleep(0.05)

        if (idx + 1) % 10 == 0:
            pd.DataFrame(all_records).to_csv(output_path, index=False)
            print(f"  Saved checkpoint: {len(all_records)} rows")

    result = pd.DataFrame(all_records)
    result.to_csv(output_path, index=False)
    print(f"Saved: {output_path} ({len(result)} rows)")
    return result



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2016-01")
    parser.add_argument("--end", type=str, default="2024-12")
    parser.add_argument(
        "--ports-file",
        type=str,
        default="watching_trade/data/raw/target_ports_rus.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="watching_trade/data/raw/viirs_ports_rus.csv",
    )
    args = parser.parse_args()

    init_gee()

    ports_df = pd.read_csv(args.ports_file)
    print(f"Loaded {len(ports_df)} ports")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extract_all_ports_ntl(ports_df, args.start, args.end, output_path)


if __name__ == "__main__":
    main()
