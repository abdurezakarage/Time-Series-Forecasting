import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf
import warnings
import os
import pickle
import json
from datetime import datetime

try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except Exception:
    PMDARIMA_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

warnings.filterwarnings('ignore')


class TSLAForecaster:
    def __init__(self, data_path, models_dir='../data/models', cutoff_date='2024-01-01'):
        self.data_path = data_path
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

        raw_df = pd.read_csv(data_path)
        if 'Date' in raw_df.columns:
            raw_df['Date'] = pd.to_datetime(raw_df['Date'], errors='coerce')
            self.df = raw_df.set_index('Date')
        else:
            self.df = raw_df.copy()
            self.df.index = pd.to_datetime(self.df.index, errors='coerce')

        # Force index to tz-naive DatetimeIndex robustly
        idx = pd.to_datetime(self.df.index, utc=True, errors='coerce')
        valid_mask = ~idx.isna()
        self.df = self.df.loc[valid_mask].copy()
        self.df.index = pd.DatetimeIndex(idx[valid_mask]).tz_localize(None)
        self.df.sort_index(inplace=True)

        close_candidates = ['Close', 'close', 'Adj Close', 'Adj_Close', 'adj_close', 'AdjClose']
        present_close = [c for c in close_candidates if c in self.df.columns]
        if not present_close:
            raise ValueError(f"No close price column found. Columns: {list(self.df.columns)}")
        self.close_col = present_close[0]

        # Build tz-naive cutoff and compare directly (safe because index is tz-naive)
        cutoff = pd.to_datetime(cutoff_date).tz_localize(None)
        mask_train = self.df.index < cutoff
        self.train = self.df.loc[mask_train, self.close_col].dropna()
        self.test = self.df.loc[~mask_train, self.close_col].dropna()

        if len(self.train) == 0 or len(self.test) == 0:
            raise ValueError("Train or test split ended up empty. Check cutoff_date and data.")

        print(f"Train: {len(self.train)}, Test: {len(self.test)}")

        self.scaler = MinMaxScaler()
        self.results = {}

    def is_tf_available(self):
        return TF_AVAILABLE

    def is_pmdarima_available(self):
        return PMDARIMA_AVAILABLE

    def _detect_seasonality(self, series, max_lag=300, strong_threshold=0.3):
        from statsmodels.tsa.stattools import acf
        series_clean = series.dropna().astype(float)
        max_lag = min(max_lag, len(series_clean) - 2)
        if max_lag < 2:
            return 5
        try:
            acf_vals = acf(series_clean, nlags=max_lag, fft=True)
            acf_vals[0] = 0
            lag = int(np.argmax(np.abs(acf_vals)))
            value = acf_vals[lag]
            if value > strong_threshold and lag >= 2:
                print(f"Detected seasonality lag {lag} with ACF={value:.3f}")
                return lag
        except Exception:
            pass
        print("No strong seasonality detected, using default m=5")
        return 5

    def calculate_metrics(self, y_true, y_pred, model_name='model'):
        y_true = np.asarray(y_true).astype(float)
        y_pred = np.asarray(y_pred).astype(float)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
        mse = mean_squared_error(y_true, y_pred)
        denom = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / denom) if denom != 0 else np.nan
        return {'model': model_name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'MSE': mse, 'R2': r2}

    def plot_forecast_results(self, train, test, forecast, model_name, metrics):
        plt.figure(figsize=(14, 7))
        plt.plot(train.index, train, label='Train')
        plt.plot(test.index, test.values, label='Actual (Test)')
        plt.plot(test.index, forecast, label=f'{model_name} Forecast')
        plt.title(f'{model_name} Forecast')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        txt = f"MAE: {metrics['MAE']:.3f}\nRMSE: {metrics['RMSE']:.3f}\nMAPE: {metrics['MAPE']:.2f}%"
        plt.gca().text(0.02, 0.98, txt, transform=plt.gca().transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.show()

    def run_arima(self, auto_optimize=True, order=None):
        print("Running ARIMA...")
        series = self.train
        if auto_optimize and self.is_pmdarima_available():
            step = auto_arima(series, seasonal=False, suppress_warnings=True, stepwise=True)
            order = step.order
            print(f"Auto ARIMA order: {order}")
        elif auto_optimize:
            order = order or (1, 1, 1)
            print(f"Fallback ARIMA order: {order}")
        else:
            order = order or (1, 1, 1)
            print(f"Using ARIMA order: {order}")

        model = ARIMA(series, order=order)
        res = model.fit()
        forecast = res.forecast(steps=len(self.test))

        metrics = self.calculate_metrics(self.test.values, forecast, 'ARIMA')
        self.plot_forecast_results(self.train, self.test, forecast, 'ARIMA', metrics)

        # Save comprehensive results
        model_params = {'order': order, 'auto_optimize': auto_optimize}
        self.save_model_results('ARIMA', res, forecast, metrics, model_params)

        self.results['ARIMA'] = {'model': res, 'forecast': forecast, 'metrics': metrics}
        return metrics

    def run_sarima(self, auto_optimize=True, order=None, seasonal_order=None):
        print("Running SARIMA...")
        m = self._detect_seasonality(self.train)
        if auto_optimize and self.is_pmdarima_available():
            step = auto_arima(self.train, seasonal=True, m=m, suppress_warnings=True, stepwise=True)
            order = step.order
            seasonal_order = step.seasonal_order
            print(f"Auto SARIMA order: {order}, seasonal_order: {seasonal_order}")
        elif auto_optimize:
            order = order or (1, 1, 1)
            seasonal_order = seasonal_order or (1, 1, 1, m)
            print(f"Fallback SARIMA order: {order}, seasonal_order: {seasonal_order}")
        else:
            order = order or (1, 1, 1)
            seasonal_order = seasonal_order or (1, 1, 1, m)
            print(f"Using SARIMA order: {order}, seasonal_order: {seasonal_order}")

        model = SARIMAX(self.train, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        forecast = res.forecast(steps=len(self.test))

        metrics = self.calculate_metrics(self.test.values, forecast, 'SARIMA')
        self.plot_forecast_results(self.train, self.test, forecast, 'SARIMA', metrics)

        # Save comprehensive results
        model_params = {
            'order': order, 
            'seasonal_order': seasonal_order, 
            'auto_optimize': auto_optimize,
            'seasonality_period': m
        }
        self.save_model_results('SARIMA', res, forecast, metrics, model_params)

        self.results['SARIMA'] = {'model': res, 'forecast': forecast, 'metrics': metrics}
        return metrics

    def prepare_lstm_data(self, window_size=60):
        scaled_train = self.scaler.fit_transform(self.train.values.reshape(-1, 1))
        scaled_test = self.scaler.transform(self.test.values.reshape(-1, 1))

        def create_sequences(data, window):
            X, y = [], []
            for i in range(window, len(data)):
                X.append(data[i - window:i, 0])
                y.append(data[i, 0])
            X = np.array(X)
            y = np.array(y)
            return X.reshape((X.shape[0], X.shape[1], 1)), y

        X_train, y_train = create_sequences(scaled_train, window_size)
        combined = np.vstack([scaled_train[-window_size:], scaled_test])
        X_test, y_test = create_sequences(combined, window_size)

        return X_train, y_train, X_test, y_test, scaled_test

    def build_lstm_model(self, window_size, units=50, layers=2, dropout=0.2):
        if not self.is_tf_available():
            raise RuntimeError("TensorFlow is not available.")

        model = Sequential()
        if layers <= 1:
            model.add(LSTM(units, return_sequences=False, input_shape=(window_size, 1)))
            model.add(Dropout(dropout))
        else:
            # First LSTM with sequences
            model.add(LSTM(units, return_sequences=True, input_shape=(window_size, 1)))
            model.add(Dropout(dropout))
            # Middle LSTM layers (if any)
            for _ in range(max(0, layers - 2)):
                model.add(LSTM(units, return_sequences=True))
                model.add(Dropout(dropout))
            # Final LSTM without sequences
            model.add(LSTM(units, return_sequences=False))
            model.add(Dropout(dropout))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    def run_lstm(self, window_size=60, units=50, layers=2, epochs=50, batch_size=32, dropout=0.2):
        if not self.is_tf_available():
            print("TensorFlow not available. Skipping LSTM.")
            return None

        X_train, y_train, X_test, y_test, scaled_test = self.prepare_lstm_data(window_size)
        # Ensure y shapes are compatible with Dense(1) output
        if y_train.ndim == 1:
            y_train_fit = y_train
        else:
            y_train_fit = y_train.squeeze()

        model = self.build_lstm_model(window_size, units=units, layers=layers, dropout=dropout)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
        history = model.fit(X_train, y_train_fit, validation_split=0.2, epochs=epochs,
                            batch_size=batch_size, callbacks=callbacks, verbose=1)

        # Use the model with restored best weights
        y_pred_scaled = model.predict(X_test)
        # Ensure 2D for inverse_transform
        if y_pred_scaled.ndim == 1:
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)
        y_pred = self.scaler.inverse_transform(y_pred_scaled).flatten()

        n_test = len(scaled_test)
        y_pred_aligned = y_pred[-n_test:]
        actual_aligned = self.test.iloc[:len(y_pred_aligned)]

        metrics = self.calculate_metrics(actual_aligned.values, y_pred_aligned, 'LSTM')
        self.plot_forecast_results(self.train, actual_aligned, y_pred_aligned, 'LSTM', metrics)

        # Save comprehensive results
        model_params = {
            'window_size': window_size,
            'units': units,
            'layers': layers,
            'epochs': epochs,
            'batch_size': batch_size,
            'dropout': dropout
        }
        self.save_model_results('LSTM', model, y_pred_aligned, metrics, model_params)

        self.results['LSTM'] = {'model': model, 'forecast': y_pred_aligned, 'metrics': metrics}
        return metrics

    def compare_and_save_all(self):
        print("Running all models and comparing results...")
        all_metrics = {}
        all_metrics['ARIMA'] = self.run_arima()
        all_metrics['SARIMA'] = self.run_sarima()
        if self.is_tf_available():
            lstm_metrics = self.run_lstm()
            if lstm_metrics:
                all_metrics['LSTM'] = lstm_metrics

        sorted_metrics = sorted(all_metrics.items(), key=lambda x: x[1]['RMSE'])
        print("\nModel comparison (sorted by RMSE):")
        for model_name, met in sorted_metrics:
            print(f"{model_name}: MAE={met['MAE']:.3f}, RMSE={met['RMSE']:.3f}, MAPE={met['MAPE']:.2f}%")

        return all_metrics

    def save_model_results(self, model_name, model, forecast, metrics, model_params=None):
        """
        Save comprehensive model results including model, forecast, metrics, and metadata.
        
        Args:
            model_name (str): Name of the model (e.g., 'ARIMA', 'SARIMA', 'LSTM')
            model: The trained model object
            forecast (array): Model predictions
            metrics (dict): Performance metrics
            model_params (dict): Model parameters used for training
        """
        # Create timestamp for unique naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create model-specific directory
        model_dir = os.path.join(self.models_dir, f"{model_name.lower()}_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        if model_name == 'LSTM':
            model_path = os.path.join(model_dir, f"{model_name.lower()}_model.keras")
            model.save(model_path)
        else:
            model_path = os.path.join(model_dir, f"{model_name.lower()}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save forecast results
        forecast_df = pd.DataFrame({
            'date': self.test.index[:len(forecast)],
            'actual': self.test.values[:len(forecast)],
            'predicted': forecast
        })
        forecast_path = os.path.join(model_dir, f"{model_name.lower()}_forecast.csv")
        forecast_df.to_csv(forecast_path, index=False)
        
        # Save metrics
        metrics_path = os.path.join(model_dir, f"{model_name.lower()}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Save model metadata
        metadata = {
            'model_name': model_name,
            'timestamp': timestamp,
            'model_path': model_path,
            'forecast_path': forecast_path,
            'metrics_path': metrics_path,
            'train_samples': len(self.train),
            'test_samples': len(self.test),
            'cutoff_date': self.train.index[-1].strftime('%Y-%m-%d'),
            'model_params': model_params or {}
        }
        metadata_path = os.path.join(model_dir, f"{model_name.lower()}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Save visualization
        plt.figure(figsize=(14, 7))
        plt.plot(self.train.index, self.train, label='Training Data', alpha=0.7)
        plt.plot(self.test.index, self.test, label='Actual Test Data', alpha=0.7)
        plt.plot(forecast_df['date'], forecast_df['predicted'], label=f'{model_name} Forecast', alpha=0.8)
        plt.title(f'{model_name} Model - Training, Test, and Forecast')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(model_dir, f"{model_name.lower()}_forecast_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{model_name} results saved to: {model_dir}")
        print(f"  - Model: {model_path}")
        print(f"  - Forecast: {forecast_path}")
        print(f"  - Metrics: {metrics_path}")
        print(f"  - Plot: {plot_path}")
        
        return model_dir

    def load_model_results(self, model_dir):
        """
        Load saved model results from a directory.
        
        Args:
            model_dir (str): Path to the model results directory
            
        Returns:
            dict: Dictionary containing loaded model, forecast, metrics, and metadata
        """
        # Load metadata
        metadata_files = [f for f in os.listdir(model_dir) if f.endswith('_metadata.json')]
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {model_dir}")
        
        metadata_path = os.path.join(model_dir, metadata_files[0])
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load model
        model_name = metadata['model_name']
        if model_name == 'LSTM':
            from tensorflow.keras.models import load_model
            model = load_model(metadata['model_path'])
        else:
            with open(metadata['model_path'], 'rb') as f:
                model = pickle.load(f)
        
        # Load forecast
        forecast_df = pd.read_csv(metadata['forecast_path'])
        
        # Load metrics
        with open(metadata['metrics_path'], 'r') as f:
            metrics = json.load(f)
        
        return {
            'model': model,
            'forecast': forecast_df,
            'metrics': metrics,
            'metadata': metadata
        }

    def save_all_results_summary(self):
        """
        Save a comprehensive summary of all model results.
        """
        if not self.results:
            print("No results to save. Run models first.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_dir = os.path.join(self.models_dir, f"summary_{timestamp}")
        os.makedirs(summary_dir, exist_ok=True)
        
        # Create comparison table
        comparison_data = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': model_name,
                'MAE': f"{metrics['MAE']:.4f}",
                'RMSE': f"{metrics['RMSE']:.4f}",
                'MAPE': f"{metrics['MAPE']:.2f}%"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = os.path.join(summary_dir, "model_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        
        # Save summary metrics
        summary_metrics = {
            'timestamp': timestamp,
            'total_models': len(self.results),
            'best_model_rmse': min([r['metrics']['RMSE'] for r in self.results.values()]),
            'best_model_name': min(self.results.items(), key=lambda x: x[1]['metrics']['RMSE'])[0],
            'model_details': {name: result['metrics'] for name, result in self.results.items()}
        }
        
        summary_path = os.path.join(summary_dir, "summary_metrics.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_metrics, f, indent=4)
        
        # Create combined forecast plot
        plt.figure(figsize=(16, 8))
        plt.plot(self.train.index, self.train, label='Training Data', alpha=0.7, linewidth=2)
        plt.plot(self.test.index, self.test, label='Actual Test Data', alpha=0.7, linewidth=2)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (model_name, result) in enumerate(self.results.items()):
            color = colors[i % len(colors)]
            forecast = result['forecast']
            plt.plot(self.test.index[:len(forecast)], forecast, 
                    label=f'{model_name} Forecast', alpha=0.8, linewidth=1.5, color=color)
        
        plt.title('All Models - Training, Test, and Forecasts Comparison')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        combined_plot_path = os.path.join(summary_dir, "all_models_comparison.png")
        plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nAll results summary saved to: {summary_dir}")
        print(f"  - Model comparison: {comparison_path}")
        print(f"  - Summary metrics: {summary_path}")
        print(f"  - Combined plot: {combined_plot_path}")
        
        return summary_dir

    def compare_models(self):
        """
        Run all available models and compare their performance.
        This is an alias for compare_and_save_all() for better naming.
        """
        return self.compare_and_save_all()
