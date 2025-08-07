"""
Data Preprocessing Module for GMF Investments
Handles data cleaning, exploration, and preparation for time series modeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

class FinancialDataPreprocessor:
    """A class to preprocess and analyze financial data."""
    
    def __init__(self, data_loader):
        """
        Initialize the preprocessor with data from the loader.
        
        Parameters:
        -----------
        data_loader : FinancialDataLoader
            Instance of the data loader containing financial data
        """
        self.data_loader = data_loader
        self.data = data_loader.data
        self.processed_data = {}
        
    def clean_data(self):
        """Clean the financial data by handling missing values and outliers."""
        print("Cleaning financial data...")
        
        for symbol, df in self.data.items():
            print(f"Cleaning {symbol}...")
            
            # Create a copy for processing
            clean_df = df.copy()
            
            # Check for missing values
            missing_values = clean_df.isnull().sum()
            if missing_values.sum() > 0:
                print(f"  Found {missing_values.sum()} missing values in {symbol}")
                
                # Forward fill for missing values (common in financial data)
                clean_df = clean_df.fillna(method='ffill')
                
                # If still missing values, backward fill
                clean_df = clean_df.fillna(method='bfill')
                
                print(f"  Missing values handled for {symbol}")
            
            # Remove any remaining rows with NaN values
            clean_df = clean_df.dropna()
            
            # Store cleaned data
            self.processed_data[symbol] = clean_df
            
            print(f"  ✓ {symbol} cleaned: {len(clean_df)} records")
        
        return self.processed_data
    
    def calculate_returns(self):
        """Calculate daily returns for all assets."""
        print("Calculating daily returns...")
        
        returns_data = {}
        
        for symbol, df in self.processed_data.items():
            if 'Close' in df.columns:
                # Calculate daily returns
                returns = df['Close'].pct_change().dropna()
                returns_data[symbol] = returns
                
                print(f"  ✓ {symbol} returns calculated: {len(returns)} records")
        
        return pd.DataFrame(returns_data)
    
    def calculate_volatility(self, window=30):
        """
        Calculate rolling volatility for all assets.
        
        Parameters:
        -----------
        window : int
            Rolling window size in days
        """
        print(f"Calculating {window}-day rolling volatility...")
        
        volatility_data = {}
        returns_df = self.calculate_returns()
        
        for symbol in returns_df.columns:
            returns = returns_df[symbol].dropna()
            volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
            volatility_data[symbol] = volatility
            
            print(f"  ✓ {symbol} volatility calculated")
        
        return pd.DataFrame(volatility_data)
    
    def perform_stationarity_test(self, data, symbol):
        """
        Perform Augmented Dickey-Fuller test for stationarity.
        
        Parameters:
        -----------
        data : pandas.Series
            Time series data to test
        symbol : str
            Asset symbol for reporting
        """
        print(f"\nPerforming stationarity test for {symbol}...")
        
        # Test on original data
        adf_result = adfuller(data.dropna())
        
        print(f"  ADF Statistic: {adf_result[0]:.6f}")
        print(f"  p-value: {adf_result[1]:.6f}")
        print(f"  Critical values:")
        for key, value in adf_result[4].items():
            print(f"    {key}: {value:.3f}")
        
        # Interpret results
        if adf_result[1] <= 0.05:
            print(f"  ✓ {symbol} is stationary (p-value <= 0.05)")
            return True
        else:
            print(f"  ✗ {symbol} is non-stationary (p-value > 0.05)")
            return False
    
    def detect_outliers(self, data, method='zscore', threshold=3):
        """
        Detect outliers in the data.
        
        Parameters:
        -----------
        data : pandas.Series
            Data to analyze for outliers
        method : str
            Method for outlier detection ('zscore' or 'iqr')
        threshold : float
            Threshold for outlier detection
        """
        outliers = []
        
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(data.dropna()))
            outliers = data[z_scores > threshold]
        elif method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[(data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))]
        
        return outliers
    
    def calculate_risk_metrics(self):
        """Calculate key risk metrics for all assets."""
        print("Calculating risk metrics...")
        
        returns_df = self.calculate_returns()
        risk_metrics = {}
        
        for symbol in returns_df.columns:
            returns = returns_df[symbol].dropna()
            
            # Calculate metrics
            mean_return = returns.mean() * 252  # Annualized
            volatility = returns.std() * np.sqrt(252)  # Annualized
            sharpe_ratio = mean_return / volatility if volatility != 0 else 0
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(returns, 5)
            
            # Maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            risk_metrics[symbol] = {
                'Mean_Return': mean_return,
                'Volatility': volatility,
                'Sharpe_Ratio': sharpe_ratio,
                'VaR_95': var_95,
                'Max_Drawdown': max_drawdown
            }
            
            print(f"  ✓ {symbol} risk metrics calculated")
        
        return pd.DataFrame(risk_metrics).T
    
    def create_summary_statistics(self):
        """Create comprehensive summary statistics for all assets."""
        print("Creating summary statistics...")
        
        summary_stats = {}
        
        for symbol, df in self.processed_data.items():
            if 'Close' in df.columns:
                stats_dict = {
                    'Count': len(df),
                    'Mean': df['Close'].mean(),
                    'Std': df['Close'].std(),
                    'Min': df['Close'].min(),
                    '25%': df['Close'].quantile(0.25),
                    '50%': df['Close'].quantile(0.50),
                    '75%': df['Close'].quantile(0.75),
                    'Max': df['Close'].max(),
                    'Skewness': df['Close'].skew(),
                    'Kurtosis': df['Close'].kurtosis()
                }
                summary_stats[symbol] = stats_dict
        
        return pd.DataFrame(summary_stats).T
    
    def plot_price_trends(self, save_path=None):
        """Plot closing price trends for all assets."""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('Closing Price Trends', fontsize=16, fontweight='bold')
        
        for i, (symbol, df) in enumerate(self.processed_data.items()):
            if 'Close' in df.columns:
                axes[i].plot(df.index, df['Close'], linewidth=1.5)
                axes[i].set_title(f'{symbol} - Closing Prices')
                axes[i].set_ylabel('Price ($)')
                axes[i].grid(True, alpha=0.3)
                
                # Add trend line
                x = np.arange(len(df))
                z = np.polyfit(x, df['Close'], 1)
                p = np.poly1d(z)
                axes[i].plot(df.index, p(x), "r--", alpha=0.8, label='Trend')
                axes[i].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_returns_distribution(self, save_path=None):
        """Plot distribution of daily returns for all assets."""
        returns_df = self.calculate_returns()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Daily Returns Distribution', fontsize=16, fontweight='bold')
        
        for i, symbol in enumerate(returns_df.columns):
            returns = returns_df[symbol].dropna()
            
            axes[i].hist(returns, bins=50, alpha=0.7, density=True, edgecolor='black')
            axes[i].set_title(f'{symbol} - Returns Distribution')
            axes[i].set_xlabel('Daily Returns')
            axes[i].set_ylabel('Density')
            axes[i].grid(True, alpha=0.3)
            
            # Add normal distribution curve for comparison
            mu, sigma = returns.mean(), returns.std()
            x = np.linspace(returns.min(), returns.max(), 100)
            normal_curve = stats.norm.pdf(x, mu, sigma)
            axes[i].plot(x, normal_curve, 'r-', linewidth=2, label='Normal Distribution')
            axes[i].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_volatility_trends(self, window=30, save_path=None):
        """Plot rolling volatility trends for all assets."""
        volatility_df = self.calculate_volatility(window)
        
        plt.figure(figsize=(15, 8))
        for symbol in volatility_df.columns:
            plt.plot(volatility_df.index, volatility_df[symbol], label=symbol, linewidth=1.5)
        
        plt.title(f'{window}-Day Rolling Volatility Trends', fontsize=16, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Annualized Volatility')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def main():
    """Main function to demonstrate data preprocessing."""
    from data_loader import FinancialDataLoader
    
    # Load data
    loader = FinancialDataLoader()
    data = loader.fetch_data()
    
    # Preprocess data
    preprocessor = FinancialDataPreprocessor(loader)
    cleaned_data = preprocessor.clean_data()
    
    # Calculate returns and test stationarity
    returns_df = preprocessor.calculate_returns()
    
    print("\n" + "="*60)
    print("STATIONARITY ANALYSIS")
    print("="*60)
    
    for symbol in returns_df.columns:
        preprocessor.perform_stationarity_test(returns_df[symbol], symbol)
    
    # Calculate risk metrics
    risk_metrics = preprocessor.calculate_risk_metrics()
    print("\n" + "="*60)
    print("RISK METRICS")
    print("="*60)
    print(risk_metrics.round(4))
    
    # Create summary statistics
    summary_stats = preprocessor.create_summary_statistics()
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(summary_stats.round(2))
    
    return preprocessor

if __name__ == "__main__":
    main()
