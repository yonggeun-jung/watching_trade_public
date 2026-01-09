"""
Extract Sentinel-1 SAR features
"""
import ee                    # Earth Engine API (https://pypi.org/project/earthengine-api/)
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
import argparse
import time

GEE_PROJECT = "ygjung91"     # Your GEE project ID (https://developers.google.com/earth-engine/guides/quickstart_python)
BUFFER_RADIUS_M = 3000       # AOI buffer radius in meters
S1_SCALE = 10                # Sentinel-1 spatial resolution in meters


def init_gee():
    try:
        ee.Initialize(project=GEE_PROJECT)
    except:
        print("Run ee.Authenticate() first")
        raise


def create_aoi(lat: float, lon: float, buffer_m: int = BUFFER_RADIUS_M) -> ee.Geometry:
    point = ee.Geometry.Point([lon, lat])
    return point.buffer(buffer_m).bounds()


def get_s1_collection(aoi: ee.Geometry, start_date: str, end_date: str) -> ee.ImageCollection:
    return (
        ee.ImageCollection('COPERNICUS/S1_GRD')    # https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .sort('system:time_start')
    )


def to_db(image: ee.Image) -> ee.Image:
    return ee.Image(10).multiply(image.add(1e-8).log10())


def compute_monthly_sar_features(aoi: ee.Geometry, year: int, month: int) -> dict:
    start = f"{year}-{month:02d}-01"
    end_dt = datetime(year, month, 1) + relativedelta(months=1)
    end = end_dt.strftime("%Y-%m-%d")
    
    collection = get_s1_collection(aoi, start, end)
    
    result = {
        'n_images': 0,
        'sar_diff_median': None,
        'sar_diff_mean': None,
        'vh_median_mean': None,
        'vh_max_mean': None,
    }
    
    n_images = collection.size().getInfo()
    result['n_images'] = n_images
    
    if n_images == 0:
        return result
    
    # SAR Difference (server-side loop)
    if n_images >= 2:
        image_list = collection.toList(n_images)
        indices = ee.List.sequence(0, ee.Number(n_images).subtract(2))
        
        def compute_diff(i):
            idx = ee.Number(i)
            img1 = ee.Image(image_list.get(idx)).select('VV')
            img2 = ee.Image(image_list.get(idx.add(1))).select('VV')
            # change to dB and compute absolute difference
            img1_db = img1.add(1e-8).log10().multiply(10)
            img2_db = img2.add(1e-8).log10().multiply(10)
            diff = img2_db.subtract(img1_db).abs()
            return diff.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=aoi,
                scale=S1_SCALE,
                maxPixels=1e10,
                bestEffort=True
            ).values().get(0)
        
        diff_values = indices.map(compute_diff)
        
        diff_stats = ee.Dictionary({
            'mean': diff_values.reduce(ee.Reducer.mean()),
            'median': diff_values.reduce(ee.Reducer.median())
        }).getInfo()
        
        result['sar_diff_median'] = diff_stats.get('median')
        result['sar_diff_mean'] = diff_stats.get('mean')
    
    # VH backscatter (mainly for stocks)
    comp_median_db = to_db(collection.median())
    comp_max_db = to_db(collection.max())
    
    vh_stats = ee.Dictionary({
        'median_mean': comp_median_db.select('VH').reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi, scale=S1_SCALE, maxPixels=1e10
        ).get('VH'),
        'max_mean': comp_max_db.select('VH').reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi, scale=S1_SCALE, maxPixels=1e10
        ).get('VH')
    }).getInfo()
    
    result['vh_median_mean'] = vh_stats.get('median_mean')
    result['vh_max_mean'] = vh_stats.get('max_mean')
    
    return result


def extract_all_ports(ports_df: pd.DataFrame, start_month: str, end_month: str, output_path: Path):
    """Extract SAR for all ports, save to single file"""
    
    start_dt = datetime.strptime(start_month, "%Y-%m")
    end_dt = datetime.strptime(end_month, "%Y-%m")
    
    all_records = []
    
    for idx, port in ports_df.iterrows():
        port_name = port['Main Port Name']
        country = port['Country Code']
        lat, lon = port['Latitude'], port['Longitude']
        wpi_id = port['World Port Index Number']
        
        print(f"[{idx+1}/{len(ports_df)}] {country} - {port_name}")
        
        aoi = create_aoi(lat, lon)
        current = start_dt
        
        while current <= end_dt:
            year, month = current.year, current.month
            month_str = current.strftime("%Y-%m")
            
            try:
                features = compute_monthly_sar_features(aoi, year, month)
                
                record = {
                    'country': country,
                    'port_name': port_name,
                    'wpi_id': wpi_id,
                    'lat': lat,
                    'lon': lon,
                    'year_month': month_str,
                    **features,
                }
                all_records.append(record)
                
            except Exception as e:
                print(f"  Error {month_str}: {e}")
            
            current += relativedelta(months=1)
            time.sleep(0.05)
        
        # Save periodically
        if (idx + 1) % 10 == 0:
            pd.DataFrame(all_records).to_csv(output_path, index=False)
            print(f"  Saved checkpoint: {len(all_records)} records")
    
    # Final save
    result = pd.DataFrame(all_records)
    result.to_csv(output_path, index=False)
    print(f"Saved: {output_path} ({len(result)} records)")
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str, default='2016-01')
    parser.add_argument('--end', type=str, default='2024-12')
    parser.add_argument('--ports-file', type=str, default='watching_trade/data/raw/target_ports_rus.csv')
    parser.add_argument('--output', type=str, default='watching_trade/data/raw/sar_ports_rus.csv')
    args = parser.parse_args()
    
    init_gee()
    
    ports_df = pd.read_csv(args.ports_file)
    print(f"Loaded {len(ports_df)} ports")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    extract_all_ports(ports_df, args.start, args.end, output_path)


if __name__ == '__main__':
    main()
