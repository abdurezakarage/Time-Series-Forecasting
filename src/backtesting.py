

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
import matplotlib.dates as mdates
warnings.filterwarnings('ignore')

class StrategyBacktester:

    
    def __init__(self, data_loader, optimization_results_path=None, risk_free_rate=0.02):
       
        self.data_loader = data_loader
        self.optimization_results_path = optimization_results_path
        self.risk_free_rate = risk_free_rate
        self.assets = ['TSLA', 'BND', 'SPY']
        
        # Backtesting parameters
        self.backtest_start_date = None
        self.backtest_end_date = None
        self.rebalancing_frequency = 'M'  # Monthly rebalancing
        
        # Strategy and benchmark portfolios
        self.strategy_weights = None
        self.benchmark_weights = {'SPY': 0.6, 'BND': 0.4, 'TSLA': 0.0}
        
        # Performance tracking
        self.strategy_returns = None
        self.benchmark_returns = None
        self.strategy_portfolio_values = None
        self.benchmark_portfolio_values = None
        
        # Results storage
        self.backtest_results = {}
        
    def load_optimization_results(self):
        """Load the latest optimization results to get strategy weights."""
        if not self.optimization_results_path:
            # Find the latest optimization results
            results_dir = '../optimization_results'
            if os.path.exists(results_dir):
                result_dirs = [d for d in os.listdir(results_dir) 
                             if os.path.isdir(os.path.join(results_dir, d))]
                if result_dirs:
                    latest_dir = max(result_dirs)
                    self.optimization_results_path = os.path.join(
                        results_dir, latest_dir, 'portfolio_optimization_results.json'
                    )
        
        if self.optimization_results_path and os.path.exists(self.optimization_results_path):
            with open(self.optimization_results_path, 'r') as f:
                results = json.load(f)
            
            # Extract max Sharpe portfolio weights
            max_sharpe_weights = results['max_sharpe_portfolio']['weights']
            weights_array = np.fromstring(max_sharpe_weights.strip('[]'), sep=' ')
            
            self.strategy_weights = dict(zip(self.assets, weights_array))
            print(f"Loaded strategy weights: {self.strategy_weights}")
            
            return results
        else:
            print("Warning: No optimization results found. Using equal weights.")
            self.strategy_weights = {asset: 1/len(self.assets) for asset in self.assets}
            return None
    
    def define_backtest_period(self, months_back=12):
       
        # Get the date range from the data
        closing_prices = self.data_loader.get_adjusted_close()
        
        if closing_prices.empty:
            raise ValueError("No price data available for backtesting")
        
        # Set backtest period to last N months
        self.backtest_end_date = closing_prices.index.max()
        self.backtest_start_date = self.backtest_end_date - pd.DateOffset(months=months_back)
        
        print(f"Backtest period: {self.backtest_start_date.date()} to {self.backtest_end_date.date()}")
        print(f"Total days: {(self.backtest_end_date - self.backtest_start_date).days}")
        
        return self.backtest_start_date, self.backtest_end_date
    
    def prepare_backtest_data(self):
    
        closing_prices = self.data_loader.get_adjusted_close()
        
        # Filter to backtest period
        backtest_data = closing_prices[
            (closing_prices.index >= self.backtest_start_date) & 
            (closing_prices.index <= self.backtest_end_date)
        ].copy()
        
        # Forward fill any missing values
        backtest_data = backtest_data.fillna(method='ffill')
        
        # Calculate daily returns
        daily_returns = backtest_data.pct_change().dropna()
        
        print(f"Backtest data prepared: {len(backtest_data)} days")
        print(f"Daily returns calculated: {len(daily_returns)} days")
        
        return backtest_data, daily_returns
    
    def calculate_portfolio_returns(self, weights: Dict[str, float], 
                                  daily_returns: pd.DataFrame) -> pd.Series:
      
        # Ensure weights sum to 1
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            print(f"Warning: Weights sum to {weight_sum:.6f}, normalizing to 1.0")
            weights = {k: v/weight_sum for k, v in weights.items()}
        
        # Calculate weighted portfolio returns
        portfolio_returns = pd.Series(0.0, index=daily_returns.index)
        
        for asset, weight in weights.items():
            if asset in daily_returns.columns and weight > 0:
                portfolio_returns += weight * daily_returns[asset]
        
        return portfolio_returns
    
    def calculate_cumulative_returns(self, daily_returns: pd.Series) -> pd.Series:
       
        return (1 + daily_returns).cumprod()
    
    def calculate_portfolio_values(self, initial_investment: float, 
                                 cumulative_returns: pd.Series) -> pd.Series:
      
        return initial_investment * cumulative_returns
    
    def calculate_performance_metrics(self, daily_returns: pd.Series, 
                                    portfolio_values: pd.Series) -> Dict:
       
        # Basic metrics
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
        
        # Risk metrics
        volatility = daily_returns.std() * np.sqrt(252)
        max_drawdown = self.calculate_max_drawdown(portfolio_values)
        
        # Sharpe ratio
        excess_returns = daily_returns - self.risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / daily_returns.std() * np.sqrt(252)
        
        # Additional metrics
        positive_days = (daily_returns > 0).sum()
        negative_days = (daily_returns < 0).sum()
        win_rate = positive_days / len(daily_returns)
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(daily_returns, 5)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'var_95': var_95,
            'positive_days': positive_days,
            'negative_days': negative_days,
            'total_days': len(daily_returns)
        }
    
    def calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
      
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()
    
    def run_backtest(self, initial_investment: float = 10000):
       
        print("="*60)
        print("RUNNING STRATEGY BACKTEST")
        print("="*60)
        
        # Load optimization results
        self.load_optimization_results()
        
        # Define backtest period
        self.define_backtest_period()
        
        # Prepare data
        backtest_prices, daily_returns = self.prepare_backtest_data()
        
        # Calculate strategy returns
        print("\nCalculating strategy returns...")
        self.strategy_returns = self.calculate_portfolio_returns(
            self.strategy_weights, daily_returns
        )
        
        # Calculate benchmark returns
        print("Calculating benchmark returns...")
        self.benchmark_returns = self.calculate_portfolio_returns(
            self.benchmark_weights, daily_returns
        )
        
        # Calculate cumulative returns
        strategy_cumulative = self.calculate_cumulative_returns(self.strategy_returns)
        benchmark_cumulative = self.calculate_cumulative_returns(self.benchmark_returns)
        
        # Calculate portfolio values
        self.strategy_portfolio_values = self.calculate_portfolio_values(
            initial_investment, strategy_cumulative
        )
        self.benchmark_portfolio_values = self.calculate_portfolio_values(
            initial_investment, benchmark_cumulative
        )
        
        # Calculate performance metrics
        print("Calculating performance metrics...")
        strategy_metrics = self.calculate_performance_metrics(
            self.strategy_returns, self.strategy_portfolio_values
        )
        benchmark_metrics = self.calculate_performance_metrics(
            self.benchmark_returns, self.benchmark_portfolio_values
        )
        
        # Store results
        self.backtest_results = {
            'backtest_period': {
                'start_date': self.backtest_start_date.strftime('%Y-%m-%d'),
                'end_date': self.backtest_end_date.strftime('%Y-%m-%d'),
                'total_days': len(daily_returns)
            },
            'strategy_weights': self.strategy_weights,
            'benchmark_weights': self.benchmark_weights,
            'initial_investment': initial_investment,
            'strategy_metrics': strategy_metrics,
            'benchmark_metrics': benchmark_metrics,
            'performance_comparison': {
                'return_difference': strategy_metrics['total_return'] - benchmark_metrics['total_return'],
                'sharpe_difference': strategy_metrics['sharpe_ratio'] - benchmark_metrics['sharpe_ratio'],
                'volatility_difference': strategy_metrics['volatility'] - benchmark_metrics['volatility'],
                'max_drawdown_difference': strategy_metrics['max_drawdown'] - benchmark_metrics['max_drawdown']
            }
        }
        
        print("Backtest completed successfully!")
        return self.backtest_results
    
    def plot_performance_comparison(self, save_path=None):
       
        if self.strategy_portfolio_values is None:
            print("Error: Run backtest first before plotting")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Strategy vs Benchmark Performance Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Portfolio Values Over Time
        axes[0, 0].plot(self.strategy_portfolio_values.index, self.strategy_portfolio_values.values, 
                       label='Strategy Portfolio', linewidth=2, color='#2E86AB')
        axes[0, 0].plot(self.benchmark_portfolio_values.index, self.benchmark_portfolio_values.values, 
                       label='Benchmark (60% SPY, 40% BND)', linewidth=2, color='#A23B72')
        axes[0, 0].set_title('Portfolio Values Over Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Format x-axis dates
        axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[0, 0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 2: Cumulative Returns
        strategy_cumulative = self.strategy_portfolio_values / self.strategy_portfolio_values.iloc[0]
        benchmark_cumulative = self.benchmark_portfolio_values / self.benchmark_portfolio_values.iloc[0]
        
        axes[0, 1].plot(strategy_cumulative.index, strategy_cumulative.values, 
                       label='Strategy Portfolio', linewidth=2, color='#2E86AB')
        axes[0, 1].plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                       label='Benchmark Portfolio', linewidth=2, color='#A23B72')
        axes[0, 1].set_title('Cumulative Returns')
        axes[0, 1].set_ylabel('Cumulative Return')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Format x-axis dates
        axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[0, 1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 3: Rolling Volatility (30-day window)
        strategy_rolling_vol = self.strategy_returns.rolling(window=30).std() * np.sqrt(252)
        benchmark_rolling_vol = self.benchmark_returns.rolling(window=30).std() * np.sqrt(252)
        
        axes[1, 0].plot(strategy_rolling_vol.index, strategy_rolling_vol.values, 
                       label='Strategy Portfolio', linewidth=2, color='#2E86AB')
        axes[1, 0].plot(benchmark_rolling_vol.index, benchmark_rolling_vol.values, 
                       label='Benchmark Portfolio', linewidth=2, color='#A23B72')
        axes[1, 0].set_title('Rolling Volatility (30-day window)')
        axes[1, 0].set_ylabel('Annualized Volatility')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Format x-axis dates
        axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[1, 0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 4: Performance Metrics Comparison
        metrics = ['Total Return', 'Sharpe Ratio', 'Volatility', 'Max Drawdown']
        strategy_values = [
            self.backtest_results['strategy_metrics']['total_return'],
            self.backtest_results['strategy_metrics']['sharpe_ratio'],
            self.backtest_results['strategy_metrics']['volatility'],
            abs(self.backtest_results['strategy_metrics']['max_drawdown'])
        ]
        benchmark_values = [
            self.backtest_results['benchmark_metrics']['total_return'],
            self.backtest_results['benchmark_metrics']['sharpe_ratio'],
            self.backtest_results['benchmark_metrics']['volatility'],
            abs(self.backtest_results['benchmark_metrics']['max_drawdown'])
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, strategy_values, width, label='Strategy', color='#2E86AB', alpha=0.8)
        axes[1, 1].bar(x + width/2, benchmark_values, width, label='Benchmark', color='#A23B72', alpha=0.8)
        axes[1, 1].set_title('Performance Metrics Comparison')
        axes[1, 1].set_ylabel('Metric Value')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance comparison plot saved to: {save_path}")
        
        plt.show()
    
    def print_backtest_summary(self):
        """Print a comprehensive summary of the backtest results."""
        if not self.backtest_results:
            print("Error: Run backtest first before printing summary")
            return
        
        print("\n" + "="*80)
        print("BACKTEST RESULTS SUMMARY")
        print("="*80)
        
        # Backtest period
        period = self.backtest_results['backtest_period']
        print(f"\n📅 Backtest Period: {period['start_date']} to {period['end_date']}")
        print(f"   Total Trading Days: {period['total_days']}")
        print(f"   Initial Investment: ${self.backtest_results['initial_investment']:,.2f}")
        
        # Portfolio weights
        print(f"\n📊 Portfolio Weights:")
        print(f"   Strategy Portfolio:")
        for asset, weight in self.backtest_results['strategy_weights'].items():
            print(f"     {asset}: {weight:.2%}")
        
        print(f"   Benchmark Portfolio:")
        for asset, weight in self.backtest_results['benchmark_weights'].items():
            if weight > 0:
                print(f"     {asset}: {weight:.2%}")
        
        # Performance metrics
        strategy_metrics = self.backtest_results['strategy_metrics']
        benchmark_metrics = self.backtest_results['benchmark_metrics']
        comparison = self.backtest_results['performance_comparison']
        
        print(f"\n📈 Performance Metrics:")
        print(f"   {'Metric':<20} {'Strategy':<15} {'Benchmark':<15} {'Difference':<15}")
        print(f"   {'-'*20} {'-'*15} {'-'*15} {'-'*15}")
        
        metrics_data = [
            ('Total Return', strategy_metrics['total_return'], benchmark_metrics['total_return']),
            ('Annualized Return', strategy_metrics['annualized_return'], benchmark_metrics['annualized_return']),
            ('Volatility', strategy_metrics['volatility'], benchmark_metrics['volatility']),
            ('Sharpe Ratio', strategy_metrics['sharpe_ratio'], benchmark_metrics['sharpe_ratio']),
            ('Max Drawdown', strategy_metrics['max_drawdown'], benchmark_metrics['max_drawdown']),
            ('Win Rate', strategy_metrics['win_rate'], benchmark_metrics['win_rate'])
        ]
        
        for metric, strategy_val, benchmark_val in metrics_data:
            diff = strategy_val - benchmark_val
            print(f"   {metric:<20} {strategy_val:>14.2%} {benchmark_val:>14.2%} {diff:>+14.2%}")
        
        # Final portfolio values
        final_strategy_value = self.strategy_portfolio_values.iloc[-1]
        final_benchmark_value = self.benchmark_portfolio_values.iloc[-1]
        
        print(f"\n💰 Final Portfolio Values:")
        print(f"   Strategy Portfolio: ${final_strategy_value:,.2f}")
        print(f"   Benchmark Portfolio: ${final_benchmark_value:,.2f}")
        print(f"   Absolute Difference: ${final_strategy_value - final_benchmark_value:+,.2f}")
        
        # Strategy evaluation
        print(f"\n🎯 Strategy Evaluation:")
        if comparison['return_difference'] > 0:
            print(f"   ✅ Strategy outperformed benchmark by {comparison['return_difference']:.2%}")
        else:
            print(f"   ❌ Strategy underperformed benchmark by {abs(comparison['return_difference']):.2%}")
        
        if comparison['sharpe_difference'] > 0:
            print(f"   ✅ Strategy has higher risk-adjusted returns (Sharpe ratio)")
        else:
            print(f"   ❌ Benchmark has higher risk-adjusted returns (Sharpe ratio)")
        
        if comparison['volatility_difference'] < 0:
            print(f"   ✅ Strategy has lower volatility")
        else:
            print(f"   ❌ Strategy has higher volatility")
        
        # Conclusion
        print(f"\n📋 Conclusion:")
        if (comparison['return_difference'] > 0 and comparison['sharpe_difference'] > 0):
            print("   The model-driven strategy shows promise with both higher returns")
            print("   and better risk-adjusted performance compared to the benchmark.")
        elif comparison['return_difference'] > 0:
            print("   The strategy generated higher returns but with higher risk.")
            print("   Consider risk management improvements for better risk-adjusted returns.")
        else:
            print("   The strategy underperformed the benchmark. This suggests:")
            print("   - The forecasting model may need refinement")
            print("   - Market conditions may have changed")
            print("   - The optimization approach may need adjustment")
        
        print("\n" + "="*80)
    
    def save_backtest_results(self, output_dir='../backtest_results'):
       
        if not self.backtest_results:
            print("Error: Run backtest first before saving results")
            return
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join(output_dir, f'backtest_results_{timestamp}')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save results JSON
        results_file = os.path.join(results_dir, 'backtest_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.backtest_results, f, indent=2, default=str)
        
        # Save performance comparison plot
        plot_file = os.path.join(results_dir, 'performance_comparison.png')
        self.plot_performance_comparison(save_path=plot_file)
        
        # Save portfolio values to CSV
        portfolio_values_df = pd.DataFrame({
            'Strategy_Portfolio': self.strategy_portfolio_values,
            'Benchmark_Portfolio': self.benchmark_portfolio_values,
            'Strategy_Returns': self.strategy_returns,
            'Benchmark_Returns': self.benchmark_returns
        })
        
        csv_file = os.path.join(results_dir, 'portfolio_values.csv')
        portfolio_values_df.to_csv(csv_file)
        
        print(f"Backtest results saved to: {results_dir}")
        print(f"  - Results JSON: {results_file}")
        print(f"  - Performance plot: {plot_file}")
        print(f"  - Portfolio values CSV: {csv_file}")
        
        return results_dir
    
    def run_complete_backtest(self, initial_investment: float = 10000):
       
        # Run backtest
        results = self.run_backtest(initial_investment)
        
        # Print summary
        self.print_backtest_summary()
        
        # Save results
        results_dir = self.save_backtest_results()
        
        return results, results_dir


def main():
    """Main function to demonstrate backtesting functionality."""
    from data_loader import FinancialDataLoader
    from optimizePortifolio import CleanedDataLoader
    
    print("Loading data for backtesting...")
    
    # Load data
    data_loader = CleanedDataLoader()
    data = data_loader.load_cleaned_data()
    
    if not data:
        print("Error: No data loaded. Please ensure cleaned data files exist.")
        return
    
    # Initialize and run backtesting
    backtester = StrategyBacktester(data_loader)
    results, results_dir = backtester.run_complete_backtest()
    
    print(f"\nBacktesting completed! Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
