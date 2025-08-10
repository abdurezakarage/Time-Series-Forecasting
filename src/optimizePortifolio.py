
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class CleanedDataLoader:
    """Load cleaned financial data from CSV files."""
    
    def __init__(self):
        self.data = {}
        self.assets = ['TSLA', 'BND', 'SPY']
        self.cleaned_data_dir = '../data/cleaned'
    
    def load_cleaned_data(self):
        """Load cleaned financial data from CSV files."""
        print("Loading cleaned financial data from CSV files...")
        
        for symbol in self.assets:
            print(f"Loading {symbol} data...")
            
            try:
                # Load cleaned data from CSV
                filepath = os.path.join(self.cleaned_data_dir, f"{symbol}_DATA_cleaned.csv")
                
                if not os.path.exists(filepath):
                    print(f"Warning: Cleaned data file not found for {symbol}: {filepath}")
                    continue
                
                data = pd.read_csv(filepath)
                
                # Convert Date column to datetime and set as index
                data['Date'] = pd.to_datetime(data['Date'])
                data = data.set_index('Date')
                
                self.data[symbol] = data
                
                print(f"✓ Successfully loaded {symbol}: {len(data)} records")
                print(f"  Date range: {data.index.min().date()} to {data.index.max().date()}")
                
            except Exception as e:
                print(f"Error loading {symbol}: {str(e)}")
        
        return self.data
    
    def get_adjusted_close(self):
        """Get closing prices for all assets as a DataFrame."""
        closing_prices = {}
        for symbol, data in self.data.items():
            if 'Close' in data.columns:
                closing_prices[symbol] = data['Close']
        return pd.DataFrame(closing_prices)

class PortfolioOptimizer:
  
    
    def __init__(self, data_loader, forecast_path=None, risk_free_rate=0.02):
      
        self.data_loader = data_loader
        self.forecast_path = forecast_path
        self.risk_free_rate = risk_free_rate
        self.assets = ['TSLA', 'BND', 'SPY']
        self.expected_returns = None
        self.covariance_matrix = None
        self.efficient_frontier = None
        self.max_sharpe_portfolio = None
        self.min_volatility_portfolio = None
        
    def load_forecast_data(self):
        """Load forecast data for TSLA from the latest forecast results."""
        if not self.forecast_path:
            # Find the latest forecast directory
            forecast_dir = '../forecasts'
            if os.path.exists(forecast_dir):
                forecast_dirs = [d for d in os.listdir(forecast_dir) if os.path.isdir(os.path.join(forecast_dir, d))]
                if forecast_dirs:
                    latest_dir = max(forecast_dirs)
                    self.forecast_path = os.path.join(forecast_dir, latest_dir, 'forecast_insights_report.json')
        
        if self.forecast_path and os.path.exists(self.forecast_path):
            with open(self.forecast_path, 'r') as f:
                forecast_data = json.load(f)
            
            # Extract the best performing model (highest expected return)
            trend_analysis = forecast_data['trend_analysis']
            best_model = max(trend_analysis.keys(), 
                           key=lambda x: float(trend_analysis[x]['total_return_pct']))
            
            # Get annualized expected return for TSLA
            total_return_pct = float(trend_analysis[best_model]['total_return_pct'])
            annual_return = total_return_pct / 100  # Convert to decimal
            
            print(f"Using {best_model} forecast for TSLA: {annual_return:.2%} annual return")
            return annual_return
        else:
            print("No forecast data found, using historical average for TSLA")
            return None
    
    def calculate_historical_returns(self):
        """Calculate historical daily returns for all assets."""
        # Get adjusted closing prices
        prices_df = self.data_loader.get_adjusted_close()
        
        # Calculate daily returns
        daily_returns = prices_df.pct_change().dropna()
        
        return daily_returns
    
    def calculate_expected_returns(self):
        """Calculate expected returns vector using forecast for TSLA and historical for others."""
        daily_returns = self.calculate_historical_returns()
        
        # Get forecast return for TSLA
        tsla_forecast_return = self.load_forecast_data()
        
        # Calculate historical annualized returns for BND and SPY
        historical_annual_returns = daily_returns[['BND', 'SPY']].mean() * 252
        
        # Create expected returns vector
        self.expected_returns = pd.Series(index=self.assets)
        
        if tsla_forecast_return is not None:
            self.expected_returns['TSLA'] = tsla_forecast_return
        else:
            # Use historical average if no forecast available
            self.expected_returns['TSLA'] = daily_returns['TSLA'].mean() * 252
        
        self.expected_returns['BND'] = historical_annual_returns['BND']
        self.expected_returns['SPY'] = historical_annual_returns['SPY']
        
        print("Expected Annual Returns:")
        for asset, ret in self.expected_returns.items():
            print(f"  {asset}: {ret:.2%}")
        
        return self.expected_returns
    
    def calculate_covariance_matrix(self):
        """Calculate the covariance matrix from historical daily returns."""
        daily_returns = self.calculate_historical_returns()
        
        # Calculate annualized covariance matrix
        self.covariance_matrix = daily_returns.cov() * 252
        
        print("\nAnnualized Covariance Matrix:")
        print(self.covariance_matrix.round(6))
        
        return self.covariance_matrix
    
    def portfolio_performance(self, weights):
       
        weights = np.array(weights)
        
        # Portfolio expected return
        portfolio_return = np.sum(weights * self.expected_returns)
        
        # Portfolio volatility
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def negative_sharpe_ratio(self, weights):
        """Negative Sharpe ratio for minimization."""
        _, _, sharpe = self.portfolio_performance(weights)
        return -sharpe
    
    def portfolio_volatility(self, weights):
        """Portfolio volatility for minimization."""
        _, volatility, _ = self.portfolio_performance(weights)
        return volatility
    
    def optimize_portfolios(self):
        """Optimize for maximum Sharpe ratio and minimum volatility portfolios."""
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: weights between 0 and 1 (no short selling)
        bounds = tuple((0, 1) for _ in range(len(self.assets)))
        
        # Initial guess: equal weights
        initial_weights = np.array([1/len(self.assets)] * len(self.assets))
        
        # Optimize for maximum Sharpe ratio
        print("\nOptimizing for Maximum Sharpe Ratio Portfolio...")
        max_sharpe_result = minimize(
            self.negative_sharpe_ratio,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if max_sharpe_result.success:
            self.max_sharpe_portfolio = {
                'weights': max_sharpe_result.x,
                'return': self.portfolio_performance(max_sharpe_result.x)[0],
                'volatility': self.portfolio_performance(max_sharpe_result.x)[1],
                'sharpe_ratio': self.portfolio_performance(max_sharpe_result.x)[2]
            }
            print("✓ Maximum Sharpe Ratio Portfolio found")
        else:
            print("✗ Failed to find Maximum Sharpe Ratio Portfolio")
        
        # Optimize for minimum volatility
        print("\nOptimizing for Minimum Volatility Portfolio...")
        min_vol_result = minimize(
            self.portfolio_volatility,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if min_vol_result.success:
            self.min_volatility_portfolio = {
                'weights': min_vol_result.x,
                'return': self.portfolio_performance(min_vol_result.x)[0],
                'volatility': self.portfolio_performance(min_vol_result.x)[1],
                'sharpe_ratio': self.portfolio_performance(min_vol_result.x)[2]
            }
            print("✓ Minimum Volatility Portfolio found")
        else:
            print("✗ Failed to find Minimum Volatility Portfolio")
    
    def generate_efficient_frontier(self, num_portfolios=1000):
        """Generate efficient frontier by simulating random portfolios."""
        print(f"\nGenerating Efficient Frontier with {num_portfolios} portfolios...")
        
        # Generate random weights
        np.random.seed(42)  # For reproducibility
        portfolios = []
        
        for _ in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(len(self.assets))
            weights = weights / np.sum(weights)  # Normalize to sum to 1
            
            # Calculate portfolio metrics
            ret, vol, sharpe = self.portfolio_performance(weights)
            
            portfolios.append({
                'weights': weights,
                'return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe
            })
        
        self.efficient_frontier = pd.DataFrame(portfolios)
        print("✓ Efficient Frontier generated")
        
        return self.efficient_frontier
    
    def plot_efficient_frontier(self, save_path=None):
        """Plot the efficient frontier with key portfolios marked."""
        if self.efficient_frontier is None:
            print("Error: Efficient frontier not generated. Run generate_efficient_frontier() first.")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot all portfolios
        plt.scatter(self.efficient_frontier['volatility'], 
                   self.efficient_frontier['return'], 
                   c=self.efficient_frontier['sharpe_ratio'], 
                   cmap='viridis', 
                   alpha=0.6, 
                   s=20)
        
        # Mark maximum Sharpe ratio portfolio
        if self.max_sharpe_portfolio:
            plt.scatter(self.max_sharpe_portfolio['volatility'], 
                       self.max_sharpe_portfolio['return'], 
                       color='red', 
                       s=200, 
                       marker='*', 
                       label='Maximum Sharpe Ratio Portfolio',
                       edgecolors='black', 
                       linewidth=2)
        
        # Mark minimum volatility portfolio
        if self.min_volatility_portfolio:
            plt.scatter(self.min_volatility_portfolio['volatility'], 
                       self.min_volatility_portfolio['return'], 
                       color='green', 
                       s=200, 
                       marker='s', 
                       label='Minimum Volatility Portfolio',
                       edgecolors='black', 
                       linewidth=2)
        
        # Add colorbar
        cbar = plt.colorbar()
        cbar.set_label('Sharpe Ratio', rotation=270, labelpad=20)
        
        # Labels and title
        plt.xlabel('Portfolio Volatility (Risk)', fontsize=12)
        plt.ylabel('Portfolio Expected Return', fontsize=12)
        plt.title('Efficient Frontier: TSLA, BND, SPY Portfolio Optimization', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add text annotations for portfolio details
        if self.max_sharpe_portfolio:
            plt.annotate(f"Max Sharpe\nReturn: {self.max_sharpe_portfolio['return']:.2%}\nVol: {self.max_sharpe_portfolio['volatility']:.2%}\nSharpe: {self.max_sharpe_portfolio['sharpe_ratio']:.3f}",
                        xy=(self.max_sharpe_portfolio['volatility'], self.max_sharpe_portfolio['return']),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        fontsize=9)
        
        if self.min_volatility_portfolio:
            plt.annotate(f"Min Vol\nReturn: {self.min_volatility_portfolio['return']:.2%}\nVol: {self.min_volatility_portfolio['volatility']:.2%}\nSharpe: {self.min_volatility_portfolio['sharpe_ratio']:.3f}",
                        xy=(self.min_volatility_portfolio['volatility'], self.min_volatility_portfolio['return']),
                        xytext=(10, -20), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                        fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Efficient frontier plot saved to {save_path}")
        
        plt.show()
    
    def print_portfolio_summary(self):
        """Print detailed summary of optimized portfolios."""
        print("\n" + "="*60)
        print("PORTFOLIO OPTIMIZATION SUMMARY")
        print("="*60)
        
        print(f"\nRisk-Free Rate: {self.risk_free_rate:.2%}")
        
        print("\nExpected Returns:")
        for asset, ret in self.expected_returns.items():
            print(f"  {asset}: {ret:.2%}")
        
        print("\nAsset Allocation:")
        print("-" * 40)
        
        if self.max_sharpe_portfolio:
            print("MAXIMUM SHARPE RATIO PORTFOLIO:")
            print(f"  Expected Return: {self.max_sharpe_portfolio['return']:.2%}")
            print(f"  Volatility: {self.max_sharpe_portfolio['volatility']:.2%}")
            print(f"  Sharpe Ratio: {self.max_sharpe_portfolio['sharpe_ratio']:.3f}")
            print("  Weights:")
            for asset, weight in zip(self.assets, self.max_sharpe_portfolio['weights']):
                print(f"    {asset}: {weight:.2%}")
        
        print()
        
        if self.min_volatility_portfolio:
            print("MINIMUM VOLATILITY PORTFOLIO:")
            print(f"  Expected Return: {self.min_volatility_portfolio['return']:.2%}")
            print(f"  Volatility: {self.min_volatility_portfolio['volatility']:.2%}")
            print(f"  Sharpe Ratio: {self.min_volatility_portfolio['sharpe_ratio']:.3f}")
            print("  Weights:")
            for asset, weight in zip(self.assets, self.min_volatility_portfolio['weights']):
                print(f"    {asset}: {weight:.2%}")
    
    def recommend_portfolio(self):
        """Recommend the optimal portfolio based on analysis."""
        print("\n" + "="*60)
        print("PORTFOLIO RECOMMENDATION")
        print("="*60)
        
        if not self.max_sharpe_portfolio or not self.min_volatility_portfolio:
            print("Error: Portfolio optimization not completed.")
            return
        
        # Compare portfolios
        max_sharpe_sharpe = self.max_sharpe_portfolio['sharpe_ratio']
        min_vol_sharpe = self.min_volatility_portfolio['sharpe_ratio']
        
        print("\nAnalysis:")
        print(f"Maximum Sharpe Portfolio Sharpe Ratio: {max_sharpe_sharpe:.3f}")
        print(f"Minimum Volatility Portfolio Sharpe Ratio: {min_vol_sharpe:.3f}")
        
        # Decision logic
        if max_sharpe_sharpe > min_vol_sharpe + 0.1:  # Significant difference
            recommended = self.max_sharpe_portfolio
            recommendation_type = "Maximum Sharpe Ratio Portfolio"
            reasoning = "The maximum Sharpe ratio portfolio offers significantly better risk-adjusted returns."
        elif self.min_volatility_portfolio['volatility'] < 0.15:  # Low volatility threshold
            recommended = self.min_volatility_portfolio
            recommendation_type = "Minimum Volatility Portfolio"
            reasoning = "The minimum volatility portfolio provides attractive returns with very low risk."
        else:
            recommended = self.max_sharpe_portfolio
            recommendation_type = "Maximum Sharpe Ratio Portfolio"
            reasoning = "The maximum Sharpe ratio portfolio offers the best risk-adjusted returns."
        
        print(f"\nRECOMMENDATION: {recommendation_type}")
        print(f"Reasoning: {reasoning}")
        
        print(f"\nRecommended Portfolio:")
        print(f"  Expected Annual Return: {recommended['return']:.2%}")
        print(f"  Annual Volatility: {recommended['volatility']:.2%}")
        print(f"  Sharpe Ratio: {recommended['sharpe_ratio']:.3f}")
        print(f"  Asset Allocation:")
        for asset, weight in zip(self.assets, recommended['weights']):
            print(f"    {asset}: {weight:.2%}")
        
        return recommended
    
    def save_results(self, output_dir='../optimization_results'):
        """Save optimization results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(output_dir, f"portfolio_optimization_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save portfolio data
        results = {
            'optimization_date': datetime.now().isoformat(),
            'risk_free_rate': self.risk_free_rate,
            'expected_returns': self.expected_returns.to_dict(),
            'covariance_matrix': self.covariance_matrix.to_dict(),
            'max_sharpe_portfolio': self.max_sharpe_portfolio,
            'min_volatility_portfolio': self.min_volatility_portfolio
        }
        
        with open(os.path.join(results_dir, 'portfolio_optimization_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save efficient frontier data
        if self.efficient_frontier is not None:
            self.efficient_frontier.to_csv(os.path.join(results_dir, 'efficient_frontier.csv'), index=False)
        
        # Save plot
        plot_path = os.path.join(results_dir, 'efficient_frontier.png')
        self.plot_efficient_frontier(save_path=plot_path)
        
        print(f"\n✓ Results saved to {results_dir}")
        return results_dir
    
    def run_optimization(self):
        """Run the complete portfolio optimization process."""
        print("Starting Portfolio Optimization...")
        print("="*50)
        
        # Step 1: Calculate expected returns
        self.calculate_expected_returns()
        
        # Step 2: Calculate covariance matrix
        self.calculate_covariance_matrix()
        
        # Step 3: Optimize portfolios
        self.optimize_portfolios()
        
        # Step 4: Generate efficient frontier
        self.generate_efficient_frontier()
        
        # Step 5: Print summary
        self.print_portfolio_summary()
        
        # Step 6: Make recommendation
        recommended = self.recommend_portfolio()
        
        # Step 7: Save results
        results_dir = self.save_results()
        
        return recommended, results_dir

