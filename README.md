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
│   ├── forcasting_model.py      # Main forecasting class
│   └── futureMarketForcast.py   # Future market forecasting module
├── notbooks/                     # Jupyter notebooks
│   ├── data_load.ipynb          # Data loading and exploration
│   ├── data_preprocess.ipynb    # Data preprocessing analysis
│   ├── Tesla_Forecasting.ipynb  # Complete forecasting workflow
│   └── futureMarket.ipynb       # Future market analysis notebook
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
├── forecasts/                    # Forecast outputs and analysis
│   ├── 20250808_220507/         # Forecast results from specific runs
│   ├── 20250808_220644/         # Additional forecast outputs
│   ├── 20250808_221358/         # Model comparison results
│   └── 20250808_232317/         # Latest forecast analysis
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore file
└── README.md                    # This file
```


## Data

The project includes stock data for:
- **TSLA**: Tesla Inc. stock data
- **SPY**: SPDR S&P 500 ETF Trust
- **BND**: Vanguard Total Bond Market ETF

Data is organized in raw and cleaned formats, with preprocessing scripts to handle missing values, outliers, and feature engineering.

##  Time Series Forecasting Implementation

###  Data Loading and Preprocessing
- **Data Loading**: Implemented comprehensive data loading utilities in `data_loader.py using yfinance`
  - Support for multiple stock datasets (TSLA, SPY, BND)
  - Automatic data validation and error handling
  - Flexible data source configuration
- **Data Preprocessing**: Advanced preprocessing pipeline in `data_preprocessing.py`
  - Missing value handling with interpolation and forward-fill methods
  - Outlier detection and treatment using IQR method
  - Feature engineering including technical indicators
  - Data normalization and scaling for model training
  - Time series specific transformations (differencing, seasonal decomposition)

### Model Implementation and Training
- **ARIMA Model**: 
  - Automatic parameter selection using AIC/BIC criteria
  - Grid search for optimal (p,d,q) parameters
  - Seasonal decomposition integration
  - Model validation with rolling window approach
- **SARIMA Model**:
  - Seasonal parameter optimization (P,D,Q,s)
  - Multi-seasonal pattern detection
  - Enhanced forecasting for seasonal data
  - Automatic seasonality detection
- **LSTM Model**:
  - Deep learning architecture with configurable layers
  - Sequence preparation and windowing
  - Early stopping and learning rate scheduling
  - Dropout regularization for overfitting prevention
  - Bidirectional LSTM support for enhanced pattern recognition

### Model Evaluation and Comparison
- **Performance Metrics**: Comprehensive evaluation using multiple metrics
  - Mean Absolute Error (MAE)
  - Root Mean Square Error (RMSE)
  - Mean Absolute Percentage Error (MAPE)
  - Directional Accuracy
- **Model Comparison**: Automated comparison framework
  - Side-by-side performance analysis
  - Statistical significance testing
  - Visualization of model comparisons
  - Ranking system for model selection

### Forecasting and Visualization
- **Forecast Generation**: Multi-step forecasting capabilities
  - Short-term (1-7 days) predictions
  - Medium-term (1-4 weeks) forecasts
  - Long-term (1-3 months) projections
  - Confidence intervals and uncertainty quantification
- **Visualization**: Comprehensive plotting system
  - Training vs test data visualization
  - Forecast vs actual comparisons
  - Model performance plots
  - Interactive charts for analysis
  - Automatic report generation




