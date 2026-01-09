# Watching Trade from Space

This repository contains the code and data used to construct a satellite-based framework for measuring port-level maritime trade activity. The project combines synthetic aperture radar (SAR), nighttime lights (NTL), port characteristics, and machine learning models to nowcast and extrapolate trade flows.

All scripts are written in Python and use relative paths by default. Trained models, figures, and tables are generated automatically from the pipeline and saved to the corresponding output folders.

---

## Requirements

- Python version: 3.9 or later
  - The code was developed and tested using Python 3.12.10.
- Main packages:
  - numpy
  - pandas
  - scikit-learn
  - xgboost
  - earthengine-api

## Note
### Google Earth Engine (GEE) Access
- Satellite data processing requires access to Google Earth Engine.
- To run SAR and NTL data extraction scripts, users must obtain:
  - Earth Engine API: https://pypi.org/project/earthengine-api/
  - A personal GEE project ID: https://developers.google.com/earth-engine/guides/quickstart_python
  - The project ID must be set in the environment before running the data scripts.
### Data
- All datasets required to reproduce the main results are already stored in the data/ directory.
- If data processing is not required, users can skip all data construction steps.
### Running the Pipeline
- The main entry point is ``main.py``.
- To skip data processing and use existing datasets, comment out ``run_data.py`` inside ``main.py`` and run the remaining steps.
- All paths are specified as relative paths. If execution issues arise due to the local environment, switching to absolute paths is recommended.

## Project structure
```
watching_trade/
├── data/
│   ├── raw/
│   │   ├── UpdatedPub150.csv          # world port index (port information, available at https://msi.nga.mil/Publications/WPI)
│   │   ├── target_ports.csv           # target ports extracted from the world port index
│   │   ├── sar_ports.csv              # SAR data 
│   │   ├── viirs_ports.csv            # NTL data
│   │   ├── trade_ports.csv            # US trade data by ports
│   │   ├── sar_ports_rus.csv          # SAR data for Russia
│   │   ├── viirs_ports_rus.csv        # NTL data for Russia
│   │   └── trade_ports_rus.csv        # US trade data by ports for Russia
│   └── cleaned/
│       ├── main.csv                   # merged dataset for main analysis
│       └── main_rus.csv               # merged dataset for Russia analysis
├── src_data/
│   ├── 01_ports.py                    # extracting target ports
│   ├── 02_sar.py                      # SAR data extraction 
│   ├── 03_viirs.py                    # NTL data extraction
│   ├── 04_trade.py                    # US port trade data extraction
│   ├── 05_merge.py                    # Merge all data to make main dataset
│   └── run_data.py                    # run data source codes
├── src_models/
│   ├── 01_ols.py                      # OLS estimation
│   ├── 02_xgb_wgt_ports.py            # XGBoost model (weight, w/ ports features)
│   ├── 03_xgb_wgt_NoPorts.py          # XGBoost model (weight, w/o ports features)
│   ├── 04_xgb_val_ports.py            # XGBoost model (value, w/ ports features)
│   ├── 05_xgb_val_NoPorts.py          # XGBoost model (value, w/o ports features)
│   ├── 06_xgb_wgt_ports_LOO.py        # XGBoost model (weight, w/ ports features, Leave-one-out, Hawaii)
│   ├── 07_xgb_wgt_NoPorts_LOO.py      # XGBoost model (weight, w/o ports features, Leave-one-out, Hawaii)
│   ├── 08_xgb_val_ports_LOO.py        # XGBoost model (value, w/ ports features, Leave-one-out, Hawaii)
│   ├── 09_xgb_val_NoPorts_LOO.py      # XGBoost model (value, w/o ports features, Leave-one-out, Hawaii)
│   ├── 10_xgb_val_NoPorts_placebo.py  # XGBoost model - placebo test (value, w/o ports features)
│   └── run_models.py                  # run models source codes
├── src_russia/
│   ├── 01_ports.py                    # extracting target ports for US/Russia
│   ├── 02_sar.py                      # SAR data extraction for US/Russia
│   ├── 03_viirs.py                    # NTL data extraction for US/Russia
│   ├── 04_merge.py                    # Merge all data to make dataset
│   ├── 05_predict.py                  # XGBoost model for Russian trade prediction
│   └── run_russia.py                  # run russia source codes
├── src_mis/                           # Generate figures and tables
│   ├── 01_ols.py                      # OLS estimation
│   ├── 02_xgb_wgt_ports.py            # XGBoost model (weight, w/ ports features)
│   ├── 03_xgb_wgt_NoPorts.py          # XGBoost model (weight, w/o ports features)
│   ├── 04_xgb_val_ports.py            # XGBoost model (value, w/ ports features)
│   ├── 05_xgb_val_NoPorts.py          # XGBoost model (value, w/o ports features)
│   ├── 06_xgb_wgt_ports_LOO.py        # XGBoost model (weight, w/ ports features, Leave-one-out, Hawaii)
│   ├── 07_xgb_wgt_NoPorts_LOO.py      # XGBoost model (weight, w/o ports features, Leave-one-out, Hawaii)
│   └── run_mis.py                     # run miscellaneous source codes
├── output_figures/
├── output_models/
├── output_tables/
└── main.py                             # main pipeline
```