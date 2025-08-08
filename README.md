# Time Series Forecasting Project

A comprehensive time series forecasting system for stock price prediction using ARIMA, SARIMA, and LSTM models.

## Features

- **Multiple Models**: ARIMA, SARIMA, and LSTM forecasting models
- **Enhanced Model Saving**: Comprehensive saving of models, metrics, forecasts, and visualizations
- **Model Comparison**: Automatic comparison and ranking of model performance
- **Easy Loading**: Simple utilities to load and use saved models
- **Visualization**: Automatic generation of forecast plots and comparisons
- **Data Processing**: Complete data loading and preprocessing pipeline
- **Jupyter Notebooks**: Interactive notebooks for data exploration and analysis

## Project Structure

```
Time-Series-Forecasting/
├── src/                          # Source code modules
│   ├── __init__.py              # Package initialization
│   ├── data_loader.py           # Data loading utilities
│   ├── data_preprocessing.py    # Data preprocessing pipeline
│   └── forcasting_model.py      # Main forecasting class
├── notbooks/                     # Jupyter notebooks
│   ├── data_load.ipynb          # Data loading and exploration
│   ├── data_preprocess.ipynb    # Data preprocessing analysis
│   └── Tesla_Forecasting.ipynb  # Complete forecasting workflow
├── data/                         # Data files
│   ├── raw/                     # Raw data files
│   │   ├── TSLA_data.csv        # Tesla stock data
│   │   ├── SPY_data.csv         # S&P 500 ETF data
│   │   └── BND_data.csv         # Bond ETF data
│   ├── cleaned/                  # Preprocessed data files
│   │   ├── TSLA_DATA_cleaned.csv
│   │   ├── SPY_DATA_cleaned.csv
│   │   └── BND_DATA_cleaned.csv
│   └── models/                   # Saved model results
│       ├── arima_*/             # ARIMA model outputs
│       ├── sarima_*/            # SARIMA model outputs
│       ├── lstm_*/              # LSTM model outputs
│       └── summary_*/           # Model comparison summaries
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore file
└── README.md                    # This file
```


## Model Saving Features

The project includes enhanced model saving functionality that automatically saves:

- **Trained Models**: ARIMA/SARIMA as pickle files, LSTM as Keras files
- **Forecast Results**: CSV files with actual vs predicted values
- **Performance Metrics**: JSON files with MAE, RMSE, MAPE
- **Model Metadata**: JSON files with parameters, timestamps, file paths
- **Visualization Plots**: PNG files showing training, test, and forecast data

## Data

The project includes stock data for:
- **TSLA**: Tesla Inc. stock data
- **SPY**: SPDR S&P 500 ETF Trust
- **BND**: Vanguard Total Bond Market ETF

Data is organized in raw and cleaned formats, with preprocessing scripts to handle missing values, outliers, and feature engineering.
