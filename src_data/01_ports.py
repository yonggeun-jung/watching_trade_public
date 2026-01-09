"""
Process WPI data and filter target ports
"""
import pandas as pd
from pathlib import Path


def load_wpi(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath, encoding='utf-8-sig')


def filter_ports(df: pd.DataFrame, sizes: list = None, countries: list = None) -> pd.DataFrame:
    result = df.copy()

    if sizes:
        result = result[result['Harbor Size'].isin(sizes)]

    if countries:
        result = result[result['Country Code'].isin(countries)]
    
    return result


def main():
    wpi_path = "watching_trade/data/raw/UpdatedPub150.csv"
    output_dir = Path("watching_trade/data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Target countries (edit this list manually)
    # For this project, we focus on US ports
    TARGET_COUNTRIES = [
        'United States'
    ]
    
    df = load_wpi(wpi_path)
    print(f"Total ports: {len(df)}")
    
    # Filter by country/size
    target_ports = filter_ports(df, sizes=['Large', 'Medium', 'Small'], countries=TARGET_COUNTRIES)
    print(f"Target ports (in selected countries and size): {len(target_ports)}")
    
    # Summary
    print(target_ports.groupby('Country Code').size())
    
    # Save
    target_ports.to_csv(output_dir / 'target_ports.csv', index=False)
    print(f"Saved to {output_dir / 'target_ports.csv'}")
    
    return target_ports


if __name__ == '__main__':
    main()