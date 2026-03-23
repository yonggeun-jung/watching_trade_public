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
  - matplotlib
  - scipy

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
- The main entry point is `main.py`. Run it from the parent directory of `watching_trade` (i.e., the project root).
- To skip data processing and use existing datasets, comment out the `run_data.py` step inside `main.py` and run the remaining steps.
- All paths are specified as relative paths. If execution issues arise due to the local environment, switching to absolute paths is recommended.

## Project structure

```
watching_trade/
├── data/
│   ├── raw/
│   │   ├── UpdatedPub150.csv          # World Port Index (port information, https://msi.nga.mil/Publications/WPI)
│   │   ├── target_ports.csv           # target ports extracted from the World Port Index (US)
│   │   ├── sar_ports.csv              # SAR data (US ports)
│   │   ├── viirs_ports.csv            # NTL data (US ports)
│   │   ├── us_trade_ports.csv         # US trade data by ports
│   │   ├── target_ports_rus.csv       # target ports for Russia (from World Port Index)
│   │   ├── sar_ports_rus.csv          # SAR data for Russian ports
│   │   └── viirs_ports_rus.csv        # NTL data for Russian ports
│   └── cleaned/
│       ├── main.csv                   # merged dataset for main analysis
│       └── main_rus.csv               # merged dataset for Russia application
├── src_data/
│   ├── 01_ports.py                    # extracting target ports from World Port Index
│   ├── 02_sar.py                      # SAR data extraction (Google Earth Engine)
│   ├── 03_viirs.py                    # NTL data extraction (Google Earth Engine)
│   ├── 04_trade.py                    # US port trade data extraction (Census API)
│   ├── 05_merge.py                    # merge all data to create main dataset
│   └── run_data.py                    # runs all data construction scripts
├── src_models/
│   ├── 01_ols.py                      # OLS baseline estimation
│   ├── 02_xgb_wgt_ports.py            # XGBoost (weight, w/ port features)
│   ├── 03_xgb_wgt_NoPorts.py          # XGBoost (weight, w/o port features)
│   ├── 04_xgb_val_ports.py            # XGBoost (value, w/ port features)
│   ├── 05_xgb_val_NoPorts.py          # XGBoost (value, w/o port features)
│   ├── 06_xgb_wgt_ports_LOO.py        # XGBoost (weight, w/ ports, Leave-one-out Hawaii)
│   ├── 07_xgb_wgt_NoPorts_LOO.py      # XGBoost (weight, w/o ports, Leave-one-out Hawaii)
│   ├── 08_xgb_val_ports_LOO.py        # XGBoost (value, w/ ports, Leave-one-out Hawaii)
│   ├── 09_xgb_val_NoPorts_LOO.py      # XGBoost (value, w/o ports, Leave-one-out Hawaii)
│   ├── 10_xgb_val_NoPort_placebo.py   # XGBoost placebo test (value, w/o ports)
│   ├── 11_xgb_wgt_OnlyPorts.py        # XGBoost (weight, port characteristics only)
│   ├── 12_xgb_val_OnlyPorts.py        # XGBoost (value, port characteristics only)
│   └── run_models.py                  # runs all model estimation scripts
├── src_russia/
│   ├── 01_ports.py                    # extracting target ports for Russia
│   ├── 02_sar.py                      # SAR data extraction for Russian ports
│   ├── 03_viirs.py                    # NTL data extraction for Russian ports
│   ├── 04_merge.py                    # merge data for Russia application
│   ├── 05_predict.py                  # predict Russian port trade using US-trained model
│   └── run_russia.py                  # runs Russia application pipeline
├── src_simulation/
│   └── run_simulation.py              # Monte Carlo simulation (level vs. change in spatial extrapolation)
├── scr_mis/                           # figures and tables
│   ├── 01_sar_example.py              # SAR example figure (Los Angeles port)
│   ├── 02_summary_stat.py             # summary statistics table
│   ├── 03_nowcasting_plots_val.py     # nowcasting validation plots (value)
│   ├── 04_nowcasting_plots_wgt.py     # nowcasting validation plots (weight)
│   ├── 05_LOO_plots_val.py            # Leave-one-out Hawaii plots (value)
│   ├── 06_LOO_plots_wgt.py            # Leave-one-out Hawaii plots (weight)
│   ├── 07_russia_plot.py              # Russian ports application figure
│   ├── 08_flow_chart.py               # pipeline schematic diagram
│   └── run_mis.py                     # runs all figure/table generation scripts
├── output_figures/                    # generated figures (PDF)
├── output_models/                     # trained models (joblib)
├── output_tables/                     # generated tables (CSV, TeX)
└── main.py                            # main pipeline (run from project root)
```

## Pipeline execution order

1. **Data** (`run_data.py`): Ports → SAR → NTL → Trade → Merge
2. **Models** (`run_models.py`): OLS → XGBoost variants (ports, NoPorts, LOO, placebo, OnlyPorts)
3. **Russia** (`run_russia.py`): Russia ports → SAR → NTL → Merge → Predict
4. **Simulation** (`run_simulation.py`): Monte Carlo simulation for appendix
5. **Figures/Tables** (`run_mis.py`): SAR example, summary stats, nowcasting plots, LOO plots, Russia plot, flow chart
