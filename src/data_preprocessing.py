# task1_preprocessing.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import os
import warnings
warnings.filterwarnings('ignore')


class FinancialDataLoader:
    def __init__(self):
        self.data = {}

    def load_data_from_csv(self, folder_path):
        """Load all CSV files from folder_path into self.data dictionary."""
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                symbol = file.replace('.csv', '').upper()
                path = os.path.join(folder_path, file)
                df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
                self.data[symbol] = df
                print(f"✓ Loaded: {symbol} ({len(df)} records)")

class FinancialDataPreprocessor:
    """A class to preprocess and analyze financial data."""

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data = data_loader.data
        self.processed_data = {}

    def clean_data(self):
        print("Cleaning financial data...")
        for symbol, df in self.data.items():
            print(f"Cleaning {symbol}...")
            df = df.fillna(method='ffill').fillna(method='bfill').dropna()
            self.processed_data[symbol] = df
            print(f"✓ {symbol} cleaned: {len(df)} records")
        return self.processed_data

    def calculate_returns(self):
        print("Calculating daily returns...")
        returns_data = {}
        for symbol, df in self.processed_data.items():
            returns = df['Close'].pct_change().dropna()
            returns_data[symbol] = returns
            print(f"✓ {symbol} returns calculated")
        return pd.DataFrame(returns_data)

    def calculate_volatility(self, window=30):
        print(f"Calculating {window}-day rolling volatility...")
        returns_df = self.calculate_returns()
        volatility_data = {
            symbol: returns_df[symbol].rolling(window).std() * np.sqrt(252)
            for symbol in returns_df.columns
        }
        return pd.DataFrame(volatility_data)

    def perform_stationarity_test(self, data, symbol):
        print(f"\nADF Test for {symbol}:")
        adf_result = adfuller(data.dropna())
        print(f"  ADF Statistic: {adf_result[0]:.4f}")
        print(f"  p-value: {adf_result[1]:.4f}")
        for key, value in adf_result[4].items():
            print(f"    {key} critical value: {value:.4f}")
        if adf_result[1] <= 0.05:
            print(f"  ✓ {symbol} is stationary")
        else:
            print(f"  ✗ {symbol} is non-stationary")

    def detect_outliers(self, data, method='zscore', threshold=3):
        if method == 'zscore':
            z = np.abs(stats.zscore(data.dropna()))
            return data[z > threshold]
        elif method == 'iqr':
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            return data[(data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))]

    def calculate_risk_metrics(self):
        print("Calculating risk metrics...")
        returns_df = self.calculate_returns()
        risk_metrics = {}
        for symbol in returns_df.columns:
            returns = returns_df[symbol]
            mean_return = returns.mean() * 252
            volatility = returns.std() * np.sqrt(252)
            sharpe = mean_return / volatility if volatility != 0 else 0
            var_95 = np.percentile(returns, 5)
            cum_returns = (1 + returns).cumprod()
            drawdown = (cum_returns - cum_returns.cummax()) / cum_returns.cummax()
            max_drawdown = drawdown.min()
            risk_metrics[symbol] = {
                'Mean Return': mean_return,
                'Volatility': volatility,
                'Sharpe Ratio': sharpe,
                'VaR 95%': var_95,
                'Max Drawdown': max_drawdown
            }
            print(f"✓ {symbol} risk metrics calculated")
        return pd.DataFrame(risk_metrics).T

    def create_summary_statistics(self):
        print("Generating summary statistics...")
        summary_stats = {}
        for symbol, df in self.processed_data.items():
            close = df['Close']
            summary_stats[symbol] = {
                'Count': len(close),
                'Mean': close.mean(),
                'Std': close.std(),
                'Min': close.min(),
                '25%': close.quantile(0.25),
                '50%': close.quantile(0.5),
                '75%': close.quantile(0.75),
                'Max': close.max(),
                'Skewness': close.skew(),
                'Kurtosis': close.kurtosis()
            }
        return pd.DataFrame(summary_stats).T

    def plot_price_trends(self):
     for symbol, df in self.processed_data.items():
        plt.figure(figsize=(12, 4))
        plt.plot(df.index, df['Close'], label='Close Price')
        plt.title(f'{symbol} - Closing Price Trend')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True)
        plt.legend()
        plt.show()


    def plot_returns_distribution(self):
     returns_df = self.calculate_returns()
     for symbol in returns_df.columns:
        returns = returns_df[symbol]
        plt.figure(figsize=(8, 4))
        sns.histplot(returns, bins=50, kde=True)
        plt.title(f'{symbol} - Daily Returns Distribution')
        plt.xlabel('Returns')
        plt.grid(True)
        plt.show()


    def plot_volatility_trends(self, window=30):
     vol_df = self.calculate_volatility(window)
     plt.figure(figsize=(12, 6))
     for symbol in vol_df.columns:
        plt.plot(vol_df.index, vol_df[symbol], label=symbol)
     plt.title(f'{window}-Day Rolling Volatility')
     plt.xlabel('Date')
     plt.ylabel('Volatility')
     plt.legend()
     plt.grid(True)
     plt.show()
    def plot_correlation_matrix(self):
      returns_df = self.calculate_returns()
      corr_matrix = returns_df.corr()
      plt.figure(figsize=(6, 5))
      sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
      plt.title("Correlation Matrix of Daily Returns")
      plt.show()


    def save_clean_data(self, output_dir='../data/cleaned'):
        print(f"Saving cleaned data to `{output_dir}`...")
        os.makedirs(output_dir, exist_ok=True)
        for symbol, df in self.processed_data.items():
            output_path = os.path.join(output_dir, f"{symbol}_cleaned.csv")
            df.to_csv(output_path)
            print(f"✓ Saved: {output_path}")


def main():
    # Load data
    loader = FinancialDataLoader()
    loader.load_data_from_csv('../data/raw')  # your CSV folder path

    # Preprocess
    preprocessor = FinancialDataPreprocessor(loader)
    preprocessor.clean_data()

    # Summary stats
    print("\nSummary Statistics:")
    print(preprocessor.create_summary_statistics().round(2))

    # Returns + Stationarity
    returns_df = preprocessor.calculate_returns()
    print("\nStationarity Tests:")
    for symbol in returns_df.columns:
        preprocessor.perform_stationarity_test(returns_df[symbol], symbol)

    # Risk metrics
    print("\nRisk Metrics:")
    print(preprocessor.calculate_risk_metrics().round(4))

    # EDA Plots
    print("\nGenerating plots...")
    preprocessor.plot_price_trends()
    preprocessor.plot_returns_distribution()
    preprocessor.plot_volatility_trends()

    # Save cleaned data to ../data/cleaned
    preprocessor.save_clean_data()

if __name__ == "__main__":
    main()