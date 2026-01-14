"""
Evaluation module for stock prediction models.
Implements financial ML metrics and visualization.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import io
import base64

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import MODELS_DIR, EXPERIMENTS_DIR, DEFAULT_BATCH_SIZE
from training.train import load_trained_model
from training.dataset import StockDataFetcher, StockSequenceDataset


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAPE as percentage
    """
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy (percentage of correct direction predictions).

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Directional accuracy as percentage
    """
    # Calculate returns
    true_returns = np.diff(y_true)
    pred_returns = np.diff(y_pred)

    # Check if directions match
    correct = np.sign(true_returns) == np.sign(pred_returns)
    return np.mean(correct) * 100


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Daily returns
        risk_free_rate: Annual risk-free rate

    Returns:
        Annualized Sharpe ratio
    """
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    if np.std(excess_returns) == 0:
        return 0
    return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)


class ModelEvaluator:
    """
    Evaluates trained models on test data.
    """

    def __init__(
        self,
        ticker: str,
        model_version: str = 'v1',
        device: Optional[str] = None
    ):
        """
        Initialize evaluator.

        Args:
            ticker: Stock ticker
            model_version: Model version string
            device: Device to run on
        """
        self.ticker = ticker
        self.model_version = model_version

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load model and scalers
        self.model, self.scalers, self.metadata = load_trained_model(
            ticker, model_version, 'best', str(self.device)
        )

    def evaluate(
        self,
        test_data: Optional[pd.DataFrame] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE
    ) -> Dict:
        """
        Evaluate model on test data.

        Args:
            test_data: Optional pre-loaded test data
            start_date: Start date if fetching new data
            end_date: End date if fetching new data
            batch_size: Batch size for inference

        Returns:
            Dictionary of evaluation metrics
        """
        # Prepare test data
        if test_data is None:
            if start_date is None or end_date is None:
                # Use dates from metadata
                start_date = self.metadata.get('test_dates', ['2023-01-01'])[0][:10]
                end_date = self.metadata.get('test_dates', ['2024-01-01'])[1][:10]

            fetcher = StockDataFetcher(self.ticker, start_date, end_date)
            test_data = fetcher.fetch_data()
            test_data = fetcher.add_technical_indicators()
            test_data = fetcher.create_targets()

        lookback = self.metadata.get('lookback', 60)
        target_col = self.metadata.get('target_col', 'Target_Price')
        feature_cols = self.scalers['feature_cols']

        # Create dataset
        test_dataset = StockSequenceDataset(
            test_data,
            lookback=lookback,
            target_col=target_col,
            feature_cols=feature_cols,
            scaler=self.scalers['feature_scaler'],
            fit_scaler=False
        )
        test_dataset.target_scaler = self.scalers['target_scaler']
        test_dataset.scaled_targets = self.scalers['target_scaler'].transform(
            test_dataset.original_targets.reshape(-1, 1)
        ).flatten()
        test_dataset.X, test_dataset.y = test_dataset._create_sequences()

        # Run inference
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        predictions_scaled = []
        actuals_scaled = []

        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                preds = self.model(batch_x)
                predictions_scaled.extend(preds.cpu().numpy().flatten())
                actuals_scaled.extend(batch_y.numpy().flatten())

        predictions_scaled = np.array(predictions_scaled)
        actuals_scaled = np.array(actuals_scaled)

        # Inverse transform to original scale
        predictions = test_dataset.inverse_transform_targets(predictions_scaled)
        actuals = test_dataset.inverse_transform_targets(actuals_scaled)

        # Calculate metrics
        metrics = self._calculate_metrics(actuals, predictions)

        # Store for plotting
        self.actuals = actuals
        self.predictions = predictions
        self.test_dates = test_data.index[lookback:lookback + len(actuals)]

        return metrics

    def _calculate_metrics(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray
    ) -> Dict:
        """Calculate all evaluation metrics."""
        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(actuals, predictions))),
            'mae': float(mean_absolute_error(actuals, predictions)),
            'mape': float(calculate_mape(actuals, predictions)),
            'directional_accuracy': float(calculate_directional_accuracy(actuals, predictions)),
            'correlation': float(np.corrcoef(actuals, predictions)[0, 1]),
            'n_samples': len(actuals)
        }

        # Calculate strategy returns
        actual_returns = np.diff(actuals) / actuals[:-1]
        pred_returns = np.diff(predictions) / predictions[:-1]

        # Strategy: go long when predicted return > 0, else hold cash
        strategy_returns = np.where(pred_returns > 0, actual_returns, 0)

        metrics['buy_hold_return'] = float((actuals[-1] / actuals[0] - 1) * 100)
        metrics['strategy_return'] = float((np.prod(1 + strategy_returns) - 1) * 100)
        metrics['buy_hold_sharpe'] = float(calculate_sharpe_ratio(actual_returns))
        metrics['strategy_sharpe'] = float(calculate_sharpe_ratio(strategy_returns))

        return metrics

    def plot_predictions(self, save_path: Optional[str] = None) -> str:
        """
        Plot predicted vs actual prices.

        Args:
            save_path: Optional path to save the plot

        Returns:
            Base64 encoded image or file path
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Price comparison
        ax1 = axes[0]
        ax1.plot(self.test_dates, self.actuals, label='Actual', alpha=0.8, linewidth=1.5)
        ax1.plot(self.test_dates, self.predictions, label='Predicted', alpha=0.8, linewidth=1.5)
        ax1.set_title(f'{self.ticker} - Predicted vs Actual Prices', fontsize=14)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Prediction error
        ax2 = axes[1]
        errors = self.predictions - self.actuals
        ax2.bar(self.test_dates, errors, alpha=0.6, width=1)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax2.set_title('Prediction Error Over Time', fontsize=14)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Error ($)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            # Return base64 encoded image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            return base64.b64encode(buf.getvalue()).decode('utf-8')

    def plot_cumulative_returns(self, save_path: Optional[str] = None) -> str:
        """
        Plot cumulative returns comparison.

        Args:
            save_path: Optional path to save the plot

        Returns:
            Base64 encoded image or file path
        """
        # Calculate returns
        actual_returns = np.diff(self.actuals) / self.actuals[:-1]
        pred_returns = np.diff(self.predictions) / self.predictions[:-1]

        # Strategy returns
        strategy_returns = np.where(pred_returns > 0, actual_returns, 0)

        # Cumulative returns
        cum_bh = np.cumprod(1 + actual_returns) - 1
        cum_strategy = np.cumprod(1 + strategy_returns) - 1

        fig, ax = plt.subplots(figsize=(14, 6))

        dates = self.test_dates[1:]
        ax.plot(dates, cum_bh * 100, label='Buy & Hold', linewidth=2)
        ax.plot(dates, cum_strategy * 100, label='Model Strategy', linewidth=2)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        ax.set_title(f'{self.ticker} - Cumulative Returns Comparison', fontsize=14)
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            return base64.b64encode(buf.getvalue()).decode('utf-8')

    def plot_training_history(self, save_path: Optional[str] = None) -> str:
        """
        Plot training loss curves.

        Args:
            save_path: Optional path to save the plot

        Returns:
            Base64 encoded image or file path
        """
        history_path = EXPERIMENTS_DIR / f"{self.ticker}_{self.model_version}_history.json"

        if not history_path.exists():
            raise FileNotFoundError(f"Training history not found: {history_path}")

        with open(history_path, 'r') as f:
            history_data = json.load(f)

        history = history_data['history']

        fig, ax = plt.subplots(figsize=(12, 6))

        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], label='Training Loss', linewidth=2)
        ax.plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2)

        ax.set_title(f'{self.ticker} - Training Loss Curves', fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Mark best epoch
        best_epoch = np.argmin(history['val_loss']) + 1
        best_loss = min(history['val_loss'])
        ax.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
        ax.scatter([best_epoch], [best_loss], color='r', s=100, zorder=5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            return base64.b64encode(buf.getvalue()).decode('utf-8')

    def generate_report(self, save_dir: Optional[str] = None) -> Dict:
        """
        Generate a complete evaluation report.

        Args:
            save_dir: Directory to save plots

        Returns:
            Report dictionary with metrics and plot paths/data
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)

            pred_plot = str(save_dir / f"{self.ticker}_predictions.png")
            returns_plot = str(save_dir / f"{self.ticker}_returns.png")
            history_plot = str(save_dir / f"{self.ticker}_history.png")
        else:
            pred_plot = None
            returns_plot = None
            history_plot = None

        # Generate plots
        predictions_img = self.plot_predictions(pred_plot)
        returns_img = self.plot_cumulative_returns(returns_plot)

        try:
            history_img = self.plot_training_history(history_plot)
        except FileNotFoundError:
            history_img = None

        report = {
            'ticker': self.ticker,
            'model_version': self.model_version,
            'metrics': self.evaluate() if not hasattr(self, 'actuals') else self._calculate_metrics(self.actuals, self.predictions),
            'plots': {
                'predictions': predictions_img,
                'cumulative_returns': returns_img,
                'training_history': history_img
            }
        }

        return report


def predict_next_day(
    ticker: str,
    model_version: str = 'v1',
    device: Optional[str] = None
) -> Dict:
    """
    Generate prediction for the next trading day.

    Args:
        ticker: Stock ticker
        model_version: Model version
        device: Device to run on

    Returns:
        Prediction results
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model and scalers
    model, scalers, metadata = load_trained_model(ticker, model_version, 'best', device)
    model.eval()

    lookback = metadata.get('lookback', 60)
    feature_cols = scalers['feature_cols']

    # Fetch recent data
    from datetime import datetime, timedelta
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=lookback + 100)).strftime('%Y-%m-%d')

    fetcher = StockDataFetcher(ticker, start_date, end_date)
    data = fetcher.fetch_data()
    data = fetcher.add_technical_indicators()

    # Get the most recent lookback days
    recent_data = data.iloc[-lookback:]

    # Scale features
    features = recent_data[feature_cols].values
    scaled_features = scalers['feature_scaler'].transform(features)

    # Create input sequence
    x = torch.FloatTensor(scaled_features).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        pred_scaled = model(x).cpu().numpy()[0, 0]

    # Inverse transform
    prediction = scalers['target_scaler'].inverse_transform([[pred_scaled]])[0, 0]

    # Calculate predicted return
    current_price = data['Close'].iloc[-1]
    predicted_return = ((prediction - current_price) / current_price) * 100

    # Confidence interval (simple heuristic based on volatility)
    volatility = data['Close'].pct_change().std() * np.sqrt(252)
    daily_vol = volatility / np.sqrt(252)
    confidence_interval = {
        'lower': float(prediction * (1 - 2 * daily_vol)),
        'upper': float(prediction * (1 + 2 * daily_vol)),
        'confidence': 0.95
    }

    return {
        'ticker': ticker,
        'current_price': float(current_price),
        'predicted_price': float(prediction),
        'predicted_return_pct': float(predicted_return),
        'prediction_direction': 'UP' if predicted_return > 0 else 'DOWN',
        'confidence_interval': confidence_interval,
        'last_data_date': str(data.index[-1]),
        'model_version': model_version
    }


if __name__ == "__main__":
    # Example evaluation
    evaluator = ModelEvaluator("AAPL", "v1")
    metrics = evaluator.evaluate()
    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Generate plots
    evaluator.plot_predictions("predictions.png")
    evaluator.plot_cumulative_returns("returns.png")
