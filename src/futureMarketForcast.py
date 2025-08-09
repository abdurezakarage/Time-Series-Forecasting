import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
import json
import pickle
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

class FutureMarketForecaster:
  
    
    def __init__(self, data_path, models_dir='../data/models', forecast_months=12):
       
        self.data_path = data_path
        self.models_dir = models_dir
        self.forecast_months = min(max(forecast_months, 6), 12)  # Ensure 6-12 months
        self.forecast_days = self.forecast_months * 30  # Approximate days
        
        # Load historical data
        self.load_historical_data()
        
        # Initialize results storage
        self.forecast_results = {}
        self.trend_analysis = {}
        self.risk_assessment = {}
        
        # Create output directory for forecasts
        self.output_dir = f"../forecasts/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_historical_data(self):
        """Load and prepare historical data for forecasting."""
        print("Loading historical data...")
        
        # Load the data
        raw_df = pd.read_csv(self.data_path)
        if 'Date' in raw_df.columns:
            raw_df['Date'] = pd.to_datetime(raw_df['Date'], errors='coerce')
            self.df = raw_df.set_index('Date')
        else:
            self.df = raw_df.copy()
            self.df.index = pd.to_datetime(self.df.index, errors='coerce')
        
        # Clean the data
        self.df = self.df.dropna()
        self.df.sort_index(inplace=True)
        
        # Identify close price column
        close_candidates = ['Close', 'close', 'Adj Close', 'Adj_Close', 'adj_close', 'AdjClose']
        present_close = [c for c in close_candidates if c in self.df.columns]
        if not present_close:
            raise ValueError(f"No close price column found. Columns: {list(self.df.columns)}")
        self.close_col = present_close[0]
        
        # Get the full time series
        self.full_series = self.df[self.close_col]
        
        print(f"✓ Loaded {len(self.full_series)} days of historical data")
        print(f"✓ Date range: {self.full_series.index[0].strftime('%Y-%m-%d')} to {self.full_series.index[-1].strftime('%Y-%m-%d')}")
        
    def load_trained_models(self):
        """Load trained models from the models directory, prioritizing LSTM."""
        print("Loading trained models...")
        self.models = {}
        
        if not os.path.exists(self.models_dir):
            print(f"Models directory {self.models_dir} not found. Please train models first.")
            return False
            
        # Look for model directories
        model_dirs = [d for d in os.listdir(self.models_dir) if os.path.isdir(os.path.join(self.models_dir, d))]
        
        # Sort to prioritize LSTM models first
        model_dirs.sort(key=lambda x: (0 if 'lstm' in x.lower() else 1, x))
        
        for model_dir in model_dirs:
            model_path = os.path.join(self.models_dir, model_dir)
            
            # Check for metadata file
            metadata_files = [f for f in os.listdir(model_path) if f.endswith('_metadata.json')]
            if metadata_files:
                metadata_path = os.path.join(model_path, metadata_files[0])
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    model_name = metadata['model_name']
                    print(f"Loading {model_name} model from {model_dir}...")
                    
                    # Check if model file exists - handle relative paths from project root
                    model_file_path = metadata['model_path']
                    if not os.path.isabs(model_file_path):
                        # If it's a relative path, make it relative to the project root
                        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        model_file_path = os.path.join(project_root, model_file_path.replace('../', ''))
                    
                    if not os.path.exists(model_file_path):
                        print(f"✗ Model file not found: {model_file_path}")
                        continue
                    
                    # Load the model
                    if model_name == 'LSTM':
                        if not TF_AVAILABLE:
                            print(f"✗ TensorFlow not available. Cannot load LSTM model.")
                            continue
                        model = load_model(model_file_path)
                    else:
                        with open(model_file_path, 'rb') as f:
                            model = pickle.load(f)
                    
                    self.models[model_name] = {
                        'model': model,
                        'metadata': metadata
                    }
                    print(f"✓ {model_name} model loaded successfully")
                    
                except Exception as e:
                    print(f"✗ Failed to load {model_name} model: {e}")
        
        if not self.models:
            print("No trained models found. Please train models first using the forecasting_model.py script.")
            print(f"Expected models in: {self.models_dir}")
            return False
            
        print(f"✓ Loaded {len(self.models)} trained models: {', '.join(self.models.keys())}")
        return True
    
    def generate_forecasts(self):
        """Generate forecasts for all loaded models."""
        print(f"\nGenerating {self.forecast_months}-month forecasts...")
        
        for model_name, model_info in self.models.items():
            print(f"\nForecasting with {model_name}...")
            
            try:
                if model_name in ['ARIMA', 'SARIMA']:
                    forecast_result = self._forecast_statistical_model(model_name, model_info)
                elif model_name == 'LSTM':
                    forecast_result = self._forecast_lstm_model(model_info)
                else:
                    print(f"Unknown model type: {model_name}")
                    continue
                
                if forecast_result:
                    self.forecast_results[model_name] = forecast_result
                    print(f"✓ {model_name} forecast completed")
                    
            except Exception as e:
                print(f"✗ {model_name} forecast failed: {e}")
    
    def _forecast_statistical_model(self, model_name, model_info):
        """Generate forecast using ARIMA or SARIMA models."""
        model = model_info['model']
        
        # Generate forecast with confidence intervals
        if model_name == 'ARIMA':
            forecast_result = model.forecast(steps=self.forecast_days)
            conf_int = model.get_forecast(steps=self.forecast_days).conf_int()
        else:  # SARIMA
            forecast_result = model.forecast(steps=self.forecast_days)
            conf_int = model.get_forecast(steps=self.forecast_days).conf_int()
        
        # Create forecast dates
        last_date = self.full_series.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                     periods=self.forecast_days, 
                                     freq='D')
        
        # Filter to business days only
        forecast_dates = forecast_dates[forecast_dates.weekday < 5]
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecast_result[:len(forecast_dates)],
            'lower_ci': conf_int.iloc[:len(forecast_dates), 0],
            'upper_ci': conf_int.iloc[:len(forecast_dates), 1]
        })
        
        return {
            'forecast_df': forecast_df,
            'model': model,
            'metadata': model_info['metadata']
        }
    
    def _forecast_lstm_model(self, model_info):
        """Generate forecast using LSTM model."""
        model = model_info['model']
        metadata = model_info['metadata']
        
        # Get model parameters
        window_size = metadata['model_params'].get('window_size', 60)
        
        # Prepare data for LSTM
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(self.full_series.values.reshape(-1, 1))
        
        # Create sequences for forecasting
        last_sequence = scaled_data[-window_size:].reshape(1, window_size, 1)
        
        # Generate forecasts
        forecasts = []
        current_sequence = last_sequence.copy()
        
        for _ in range(self.forecast_days):
            # Predict next value
            next_pred = model.predict(current_sequence, verbose=0)
            forecasts.append(next_pred[0, 0])
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred[0, 0]
        
        # Inverse transform forecasts
        forecast_values = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
        
        # Create forecast dates
        last_date = self.full_series.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                     periods=self.forecast_days, 
                                     freq='D')
        forecast_dates = forecast_dates[forecast_dates.weekday < 5]
        
        # For LSTM, we'll create simple confidence intervals based on historical volatility
        historical_volatility = self.full_series.pct_change().std()
        forecast_std = forecast_values * historical_volatility
        
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecast_values[:len(forecast_dates)],
            'lower_ci': forecast_values[:len(forecast_dates)] - 1.96 * forecast_std[:len(forecast_dates)],
            'upper_ci': forecast_values[:len(forecast_dates)] + 1.96 * forecast_std[:len(forecast_dates)]
        })
        
        return {
            'forecast_df': forecast_df,
            'model': model,
            'metadata': metadata
        }
    
    def analyze_trends(self):
        """Analyze trends in the forecasts."""
        print("\nAnalyzing forecast trends...")
        
        for model_name, result in self.forecast_results.items():
            forecast_df = result['forecast_df']
            
            # Calculate trend metrics
            initial_price = forecast_df['forecast'].iloc[0]
            final_price = forecast_df['forecast'].iloc[-1]
            total_return = (final_price - initial_price) / initial_price * 100
            
            # Calculate monthly returns
            monthly_returns = []
            for i in range(0, len(forecast_df), 30):
                if i + 30 < len(forecast_df):
                    month_start = forecast_df['forecast'].iloc[i]
                    month_end = forecast_df['forecast'].iloc[i + 30]
                    monthly_return = (month_end - month_start) / month_start * 100
                    monthly_returns.append(monthly_return)
            
            # Trend direction
            if total_return > 5:
                trend_direction = "Strong Upward"
            elif total_return > 0:
                trend_direction = "Moderate Upward"
            elif total_return > -5:
                trend_direction = "Stable"
            elif total_return > -10:
                trend_direction = "Moderate Downward"
            else:
                trend_direction = "Strong Downward"
            
            # Volatility analysis
            forecast_volatility = forecast_df['forecast'].pct_change().std() * np.sqrt(252)
            
            # Confidence interval width analysis
            ci_widths = forecast_df['upper_ci'] - forecast_df['lower_ci']
            avg_ci_width = ci_widths.mean()
            ci_width_trend = np.polyfit(range(len(ci_widths)), ci_widths, 1)[0]
            
            self.trend_analysis[model_name] = {
                'initial_price': initial_price,
                'final_price': final_price,
                'total_return_pct': total_return,
                'trend_direction': trend_direction,
                'monthly_returns': monthly_returns,
                'avg_monthly_return': np.mean(monthly_returns) if monthly_returns else 0,
                'forecast_volatility': forecast_volatility,
                'avg_ci_width': avg_ci_width,
                'ci_width_trend': ci_width_trend,
                'confidence_degradation': ci_width_trend > 0
            }
            
            print(f"✓ {model_name} trend analysis completed")
    
    def assess_risks(self):
        """Assess risks and opportunities based on forecasts."""
        print("\nAssessing risks and opportunities...")
        
        for model_name, result in self.forecast_results.items():
            forecast_df = result['forecast_df']
            trend_info = self.trend_analysis[model_name]
            
            # Risk metrics
            max_drawdown = self._calculate_max_drawdown(forecast_df['forecast'])
            var_95 = np.percentile(forecast_df['forecast'].pct_change().dropna(), 5)
            
            # Opportunity metrics
            upside_potential = (forecast_df['upper_ci'].max() - forecast_df['forecast'].iloc[0]) / forecast_df['forecast'].iloc[0] * 100
            downside_risk = (forecast_df['lower_ci'].min() - forecast_df['forecast'].iloc[0]) / forecast_df['forecast'].iloc[0] * 100
            
            # Risk level assessment
            if trend_info['forecast_volatility'] > 0.5:
                risk_level = "High"
            elif trend_info['forecast_volatility'] > 0.3:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            # Market opportunity assessment
            if trend_info['total_return_pct'] > 10:
                opportunity_level = "High"
            elif trend_info['total_return_pct'] > 5:
                opportunity_level = "Medium"
            elif trend_info['total_return_pct'] > 0:
                opportunity_level = "Low"
            else:
                opportunity_level = "Negative"
            
            self.risk_assessment[model_name] = {
                'risk_level': risk_level,
                'opportunity_level': opportunity_level,
                'max_drawdown_pct': max_drawdown,
                'var_95': var_95,
                'upside_potential_pct': upside_potential,
                'downside_risk_pct': downside_risk,
                'volatility_risk': trend_info['forecast_volatility'],
                'confidence_uncertainty': trend_info['confidence_degradation']
            }
            
            print(f"✓ {model_name} risk assessment completed")
    
    def _calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown percentage."""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak * 100
        return drawdown.min()
    
    def create_visualizations(self):
        """Create comprehensive visualizations of the forecasts."""
        print("\nCreating forecast visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'Tesla Stock: {self.forecast_months}-Month Future Market Forecast Analysis', 
                    fontsize=16, fontweight='bold')
        
        # 1. Historical data and forecasts
        ax1 = axes[0, 0]
        ax1.plot(self.full_series.index, self.full_series.values, 
                label='Historical Data', linewidth=2, color='black', alpha=0.8)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (model_name, result) in enumerate(self.forecast_results.items()):
            color = colors[i % len(colors)]
            forecast_df = result['forecast_df']
            
            # Plot forecast line
            ax1.plot(forecast_df['date'], forecast_df['forecast'], 
                    label=f'{model_name} Forecast', linewidth=2, color=color)
            
            # Plot confidence intervals
            ax1.fill_between(forecast_df['date'], 
                           forecast_df['lower_ci'], 
                           forecast_df['upper_ci'], 
                           alpha=0.2, color=color)
        
        ax1.set_title('Historical Data and Forecasts with Confidence Intervals')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Stock Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Forecast comparison
        ax2 = axes[0, 1]
        for i, (model_name, result) in enumerate(self.forecast_results.items()):
            color = colors[i % len(colors)]
            forecast_df = result['forecast_df']
            ax2.plot(forecast_df['date'], forecast_df['forecast'], 
                    label=f'{model_name}', linewidth=2, color=color)
        
        ax2.set_title('Forecast Comparison Across Models')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Stock Price ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Confidence interval width analysis
        ax3 = axes[1, 0]
        for i, (model_name, result) in enumerate(self.forecast_results.items()):
            color = colors[i % len(colors)]
            forecast_df = result['forecast_df']
            ci_widths = forecast_df['upper_ci'] - forecast_df['lower_ci']
            ax3.plot(forecast_df['date'], ci_widths, 
                    label=f'{model_name} CI Width', linewidth=2, color=color)
        
        ax3.set_title('Confidence Interval Width Over Time')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('CI Width ($)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Risk vs Opportunity scatter plot
        ax4 = axes[1, 1]
        for model_name in self.risk_assessment.keys():
            risk_info = self.risk_assessment[model_name]
            trend_info = self.trend_analysis[model_name]
            
            # Color code by risk level
            if risk_info['risk_level'] == 'High':
                color = 'red'
            elif risk_info['risk_level'] == 'Medium':
                color = 'orange'
            else:
                color = 'green'
            
            ax4.scatter(risk_info['volatility_risk'], trend_info['total_return_pct'], 
                       s=100, color=color, alpha=0.7, label=f'{model_name}')
            ax4.annotate(model_name, (risk_info['volatility_risk'], trend_info['total_return_pct']), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax4.set_title('Risk vs Opportunity Analysis')
        ax4.set_xlabel('Forecast Volatility')
        ax4.set_ylabel('Expected Return (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'forecast_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create individual model plots
        self._create_individual_model_plots()
    
    def _create_individual_model_plots(self):
        """Create individual detailed plots for each model."""
        for model_name, result in self.forecast_results.items():
            forecast_df = result['forecast_df']
            trend_info = self.trend_analysis[model_name]
            risk_info = self.risk_assessment[model_name]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
            fig.suptitle(f'{model_name} Model: Detailed Forecast Analysis', fontsize=14, fontweight='bold')
            
            # Main forecast plot
            ax1.plot(self.full_series.index, self.full_series.values, 
                    label='Historical Data', linewidth=2, color='black', alpha=0.8)
            ax1.plot(forecast_df['date'], forecast_df['forecast'], 
                    label=f'{model_name} Forecast', linewidth=2, color='red')
            ax1.fill_between(forecast_df['date'], 
                           forecast_df['lower_ci'], 
                           forecast_df['upper_ci'], 
                           alpha=0.3, color='red', label='95% Confidence Interval')
            
            ax1.set_title(f'Forecast: {trend_info["trend_direction"]} Trend, {trend_info["total_return_pct"]:.1f}% Expected Return')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Stock Price ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Risk metrics
            ax2.bar(['Risk Level', 'Opportunity Level', 'Max Drawdown (%)', 'Upside Potential (%)'], 
                   [0, 0, risk_info['max_drawdown_pct'], risk_info['upside_potential_pct']], 
                   color=['red', 'green', 'orange', 'blue'], alpha=0.7)
            ax2.set_title(f'Risk Assessment: {risk_info["risk_level"]} Risk, {risk_info["opportunity_level"]} Opportunity')
            ax2.set_ylabel('Percentage (%)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{model_name.lower()}_detailed_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_insights_report(self):
        """Generate a comprehensive insights report."""
        print("\nGenerating insights report...")
        
        report = {
            'forecast_period': f"{self.forecast_months} months",
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models_analyzed': list(self.forecast_results.keys()),
            'trend_analysis': self.trend_analysis,
            'risk_assessment': self.risk_assessment,
            'key_insights': self._generate_key_insights(),
            'market_opportunities': self._identify_market_opportunities(),
            'risk_factors': self._identify_risk_factors(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        report_path = os.path.join(self.output_dir, 'forecast_insights_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4, default=str)
        
        # Print summary
        self._print_insights_summary(report)
        
        return report
    
    def _generate_key_insights(self):
        """Generate key insights from the analysis."""
        insights = []
        
        # Overall trend consensus
        trend_directions = [info['trend_direction'] for info in self.trend_analysis.values()]
        most_common_trend = max(set(trend_directions), key=trend_directions.count)
        insights.append(f"Consensus Trend: {most_common_trend} across {len(self.models)} models")
        
        # Average expected return
        avg_return = np.mean([info['total_return_pct'] for info in self.trend_analysis.values()])
        insights.append(f"Average Expected Return: {avg_return:.1f}% over {self.forecast_months} months")
        
        # Volatility assessment
        avg_volatility = np.mean([info['forecast_volatility'] for info in self.trend_analysis.values()])
        if avg_volatility > 0.4:
            insights.append(f"High Volatility Expected: {avg_volatility:.2f} annualized volatility")
        else:
            insights.append(f"Moderate Volatility Expected: {avg_volatility:.2f} annualized volatility")
        
        # Confidence interval analysis
        confidence_degradation_count = sum(1 for info in self.trend_analysis.values() 
                                         if info['confidence_degradation'])
        if confidence_degradation_count > 0:
            insights.append(f"Uncertainty Increases: {confidence_degradation_count} models show degrading confidence over time")
        
        return insights
    
    def _identify_market_opportunities(self):
        """Identify potential market opportunities."""
        opportunities = []
        
        for model_name, trend_info in self.trend_analysis.items():
            if trend_info['total_return_pct'] > 10:
                opportunities.append({
                    'model': model_name,
                    'opportunity': f"Strong upside potential: {trend_info['total_return_pct']:.1f}% expected return",
                    'confidence': 'High' if trend_info['avg_ci_width'] < 50 else 'Medium'
                })
            elif trend_info['total_return_pct'] > 5:
                opportunities.append({
                    'model': model_name,
                    'opportunity': f"Moderate growth opportunity: {trend_info['total_return_pct']:.1f}% expected return",
                    'confidence': 'Medium'
                })
        
        return opportunities
    
    def _identify_risk_factors(self):
        """Identify key risk factors."""
        risks = []
        
        for model_name, risk_info in self.risk_assessment.items():
            if risk_info['risk_level'] == 'High':
                risks.append({
                    'model': model_name,
                    'risk': f"High volatility risk: {risk_info['volatility_risk']:.2f} annualized volatility",
                    'severity': 'High'
                })
            
            if risk_info['max_drawdown_pct'] < -20:
                risks.append({
                    'model': model_name,
                    'risk': f"Potential significant drawdown: {risk_info['max_drawdown_pct']:.1f}%",
                    'severity': 'High'
                })
            
            if risk_info['confidence_uncertainty']:
                risks.append({
                    'model': model_name,
                    'risk': "Forecast uncertainty increases over time",
                    'severity': 'Medium'
                })
        
        return risks
    
    def _generate_recommendations(self):
        """Generate actionable recommendations."""
        recommendations = []
        
        # Analyze consensus
        positive_models = sum(1 for info in self.trend_analysis.values() 
                            if info['total_return_pct'] > 0)
        total_models = len(self.trend_analysis)
        
        if positive_models > total_models / 2:
            recommendations.append({
                'type': 'Investment',
                'recommendation': 'Consider long position given majority of models predict positive returns',
                'confidence': 'Medium' if positive_models < total_models * 0.8 else 'High'
            })
        else:
            recommendations.append({
                'type': 'Investment',
                'recommendation': 'Exercise caution - majority of models predict negative returns',
                'confidence': 'High'
            })
        
        # Risk management recommendations
        high_risk_models = sum(1 for info in self.risk_assessment.values() 
                              if info['risk_level'] == 'High')
        
        if high_risk_models > 0:
            recommendations.append({
                'type': 'Risk Management',
                'recommendation': 'Implement strict stop-loss orders due to high volatility predictions',
                'confidence': 'High'
            })
        
        # Monitoring recommendations
        recommendations.append({
            'type': 'Monitoring',
            'recommendation': f'Monitor forecasts monthly and adjust positions based on model convergence',
            'confidence': 'Medium'
        })
        
        return recommendations
    
    def _print_insights_summary(self, report):
        """Print a summary of key insights."""
        print("\n" + "="*80)
        print("FUTURE MARKET FORECAST INSIGHTS SUMMARY")
        print("="*80)
        
        print(f"\nForecast Period: {report['forecast_period']}")
        print(f"Models Analyzed: {', '.join(report['models_analyzed'])}")
        
        print("\nKEY INSIGHTS:")
        for insight in report['key_insights']:
            print(f"• {insight}")
        
        print("\nMARKET OPPORTUNITIES:")
        for opp in report['market_opportunities']:
            print(f"• {opp['model']}: {opp['opportunity']} (Confidence: {opp['confidence']})")
        
        print("\nRISK FACTORS:")
        for risk in report['risk_factors']:
            print(f"• {risk['model']}: {risk['risk']} (Severity: {risk['severity']})")
        
        print("\nRECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"• {rec['type']}: {rec['recommendation']} (Confidence: {rec['confidence']})")
        
        print(f"\nDetailed report saved to: {self.output_dir}")
        print("="*80)
    
    def run_complete_analysis(self):
        """Run the complete future market analysis pipeline."""
        print("Starting Future Market Forecast Analysis...")
        print("="*60)
        
        # Load models
        if not self.load_trained_models():
            return None
        
        # Generate forecasts
        self.generate_forecasts()
        
        # Analyze trends
        self.analyze_trends()
        
        # Assess risks
        self.assess_risks()
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate insights report
        report = self.generate_insights_report()
        
        print(f"\nAnalysis complete! Results saved to: {self.output_dir}")
        return report


def main():
    """Main function to run the future market forecasting analysis using LSTM model."""
    # Configuration
    data_path = "data/cleaned/TSLA_DATA_cleaned.csv"  # Adjust path as needed
    models_dir = "data/models"
    forecast_months = 12  # Can be adjusted between 6-12 months
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the cleaned Tesla stock data file is available.")
        return
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        print(f"Error: Models directory not found at {models_dir}")
        print("Please ensure trained models are available.")
        return
    
    try:
        # Initialize and run analysis
        print(f"Initializing Future Market Forecaster for {forecast_months}-month forecast...")
        forecaster = FutureMarketForecaster(
            data_path=data_path,
            models_dir=models_dir,
            forecast_months=forecast_months
        )
        
        # Run complete analysis
        results = forecaster.run_complete_analysis()
        
        if results:
            print("\n" + "="*50)
            print("FUTURE MARKET FORECAST ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*50)
            print(f"\nResults saved to: {forecaster.output_dir}")
            print("\nGenerated files:")
            print("• forecast_analysis.png - Main analysis visualization")
            print("• individual_model_plots.png - Individual model forecasts")
            print("• future_market_insights_report.json - Comprehensive insights report")
            
            # Print a quick summary
            print(f"\nQuick Summary:")
            print(f"• Forecast Period: {forecast_months} months")
            print(f"• Models Analyzed: {', '.join(results['models_analyzed'])}")
            print(f"• Key Insights: {len(results['key_insights'])} insights generated")
            print(f"• Risk Factors: {len(results['risk_factors'])} identified")
            print(f"• Recommendations: {len(results['recommendations'])} provided")
            
        else:
            print("\nAnalysis failed. Please check the error messages above.")
            
    except Exception as e:
        print(f"\nError during analysis: {e}")
        print("Please ensure all dependencies are installed and data is available.")


if __name__ == "__main__":
    main()
