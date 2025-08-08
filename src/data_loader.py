"""
Data Loading Module for GMF Investments
Handles fetching and initial processing of financial data from YFinance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class FinancialDataLoader:
    """A class to load and manage financial data from YFinance."""
    
    def __init__(self):
        self.data = {}
        self.assets = {
            'TSLA': 'Tesla Inc. - High-growth, high-risk stock',
            'BND': 'Vanguard Total Bond Market ETF - Stability and income',
            'SPY': 'SPDR S&P 500 ETF - Broad market exposure'
        }
    
    def fetch_data(self, start_date='2015-07-01', end_date='2025-07-31'):
        """Fetch historical financial data for all assets."""
        print("Fetching financial data from YFinance...")
        
        for symbol, description in self.assets.items():
            print(f"Loading {symbol}: {description}")
            
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                if data.empty:
                    print(f"Warning: No data retrieved for {symbol}")
                    continue
                
                data = data.reset_index()
                data['Date'] = pd.to_datetime(data['Date'])
                data = data.set_index('Date')
                self.data[symbol] = data
                
                print(f"✓ Successfully loaded {symbol}: {len(data)} records")
                print(f"  Date range: {data.index.min().date()} to {data.index.max().date()}")
                
            except Exception as e:
                print(f"Error loading {symbol}: {str(e)}")
        
        return self.data

    def save_data_to_csv(self, directory='../data'):
        """Save each asset's data to a separate CSV file."""
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for symbol, df in self.data.items():
            filepath = os.path.join(directory, f"{symbol}_data.csv")
            df.to_csv(filepath)
            print(f"✓ Saved {symbol} data to {filepath}")

    def get_closing_prices(self):
        """Get closing prices for all assets as a DataFrame."""
        closing_prices = {}
        for symbol, data in self.data.items():
            if 'Close' in data.columns:
                closing_prices[symbol] = data['Close']
        return pd.DataFrame(closing_prices)
    
    def get_adjusted_close(self):
        """Get adjusted closing prices for all assets as a DataFrame."""
        adj_close = {}
        for symbol, data in self.data.items():
            if 'Adj Close' in data.columns:
                adj_close[symbol] = data['Adj Close']
        return pd.DataFrame(adj_close)
    
    def get_volume_data(self):
        """Get volume data for all assets as a DataFrame."""
        volume_data = {}
        for symbol, data in self.data.items():
            if 'Volume' in data.columns:
                volume_data[symbol] = data['Volume']
        return pd.DataFrame(volume_data)

def main():
    """Main function to demonstrate data loading and saving."""
    loader = FinancialDataLoader()
    data = loader.fetch_data()
    
    print("\n" + "="*50)
    print("DATA SUMMARY")
    print("="*50)
    
    for symbol, df in data.items():
        print(f"\n{symbol}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"  Missing values: {df.isnull().sum().sum()}")
        
        if 'Close' in df.columns:
            print(f"  Close price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    
    # Save fetched data to CSV files
    loader.save_data_to_csv(directory='data')
    
    return loader

if __name__ == "__main__":
    main()
