"""
Data fetching, feature engineering, and dataset preparation for stock prediction.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import ta
import joblib

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_DIR, MAG7_STOCKS, DEFAULT_LOOKBACK


class StockDataFetcher:
    """
    Fetches and processes stock data from Yahoo Finance.
    """

    def __init__(self, ticker: str, start_date: str, end_date: str):
        """
        Initialize the data fetcher.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        """
        if ticker not in MAG7_STOCKS:
            raise ValueError(f"Ticker must be one of MAG-7: {list(MAG7_STOCKS.keys())}")

        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.scaler = MinMaxScaler()

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance.

        Returns:
            DataFrame with OHLCV data
        """
        print(f"Fetching data for {self.ticker} from {self.start_date} to {self.end_date}...")

        stock = yf.Ticker(self.ticker)
        self.data = stock.history(start=self.start_date, end=self.end_date)

        if self.data.empty:
            raise ValueError(f"No data found for {self.ticker}")

        # Clean column names and reset index
        self.data = self.data[['Open', 'High', 'Low', 'Close', 'Volume']]
        self.data.index = pd.to_datetime(self.data.index).tz_localize(None)

        print(f"Fetched {len(self.data)} trading days")
        return self.data

    def add_technical_indicators(self) -> pd.DataFrame:
        """
        Add technical indicators to the dataset.

        Returns:
            DataFrame with technical indicators
        """
        if self.data is None:
            raise ValueError("Must fetch data first")

        df = self.data.copy()

        # Moving Averages
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)

        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_Upper'] = bollinger.bollinger_hband()
        df['BB_Middle'] = bollinger.bollinger_mavg()
        df['BB_Lower'] = bollinger.bollinger_lband()

        # ATR (Average True Range)
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])

        # Log Returns
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

        # Volatility (20-day rolling std of returns)
        df['Volatility_20'] = df['Log_Return'].rolling(window=20).std()

        # Drop NaN rows
        df = df.dropna()

        self.data = df
        print(f"Added technical indicators. Final dataset: {len(df)} rows, {len(df.columns)} features")

        return df

    def create_targets(self) -> pd.DataFrame:
        """
        Create prediction targets: next-day price and return.

        Returns:
            DataFrame with targets added
        """
        if self.data is None:
            raise ValueError("Must fetch data first")

        df = self.data.copy()

        # Next-day close price
        df['Target_Price'] = df['Close'].shift(-1)

        # Next-day return (percentage)
        df['Target_Return'] = ((df['Close'].shift(-1) - df['Close']) / df['Close']) * 100

        # Direction (1 = up, 0 = down)
        df['Target_Direction'] = (df['Target_Return'] > 0).astype(int)

        # Drop last row (no target)
        df = df.dropna()

        self.data = df
        return df

    def save_data(self, format: str = 'parquet') -> str:
        """
        Save processed data to disk.

        Args:
            format: 'parquet' or 'csv'

        Returns:
            Path to saved file
        """
        if self.data is None:
            raise ValueError("No data to save")

        filename = f"{self.ticker}_{self.start_date}_{self.end_date}"

        if format == 'parquet':
            filepath = DATA_DIR / f"{filename}.parquet"
            self.data.to_parquet(filepath)
        else:
            filepath = DATA_DIR / f"{filename}.csv"
            self.data.to_csv(filepath)

        print(f"Data saved to {filepath}")
        return str(filepath)

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load processed data from disk.

        Args:
            filepath: Path to data file

        Returns:
            Loaded DataFrame
        """
        filepath = Path(filepath)

        if filepath.suffix == '.parquet':
            self.data = pd.read_parquet(filepath)
        else:
            self.data = pd.read_csv(filepath, index_col=0, parse_dates=True)

        return self.data


class StockSequenceDataset(Dataset):
    """
    PyTorch Dataset for stock price sequences.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        lookback: int = DEFAULT_LOOKBACK,
        target_col: str = 'Target_Price',
        feature_cols: Optional[List[str]] = None,
        scaler: Optional[MinMaxScaler] = None,
        fit_scaler: bool = True
    ):
        """
        Initialize the dataset.

        Args:
            data: DataFrame with features and targets
            lookback: Number of past days to use for prediction
            target_col: Column name for the target variable
            feature_cols: List of feature column names (None = use all except targets)
            scaler: Pre-fitted scaler (for validation/test sets)
            fit_scaler: Whether to fit the scaler on this data
        """
        self.lookback = lookback
        self.target_col = target_col

        # Define feature columns
        target_cols = ['Target_Price', 'Target_Return', 'Target_Direction']
        if feature_cols is None:
            self.feature_cols = [c for c in data.columns if c not in target_cols]
        else:
            self.feature_cols = feature_cols

        # Extract features and targets
        features = data[self.feature_cols].values
        targets = data[target_col].values

        # Scale features
        if scaler is None:
            self.scaler = MinMaxScaler()
            if fit_scaler:
                self.scaled_features = self.scaler.fit_transform(features)
            else:
                raise ValueError("Must provide scaler if fit_scaler=False")
        else:
            self.scaler = scaler
            self.scaled_features = self.scaler.transform(features)

        # Scale targets (for price prediction)
        self.target_scaler = MinMaxScaler()
        if fit_scaler:
            self.scaled_targets = self.target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
        else:
            self.scaled_targets = targets  # Will be set externally

        # Store original values for inverse transform
        self.original_targets = targets

        # Create sequences
        self.X, self.y = self._create_sequences()

    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input sequences and corresponding targets.

        Returns:
            Tuple of (sequences, targets) arrays
        """
        X, y = [], []

        for i in range(self.lookback, len(self.scaled_features)):
            X.append(self.scaled_features[i - self.lookback:i])
            y.append(self.scaled_targets[i])

        return np.array(X), np.array(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.FloatTensor(self.X[idx]),
            torch.FloatTensor([self.y[idx]])
        )

    def inverse_transform_targets(self, scaled_targets: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled targets to original scale.

        Args:
            scaled_targets: Scaled target values

        Returns:
            Original scale values
        """
        return self.target_scaler.inverse_transform(scaled_targets.reshape(-1, 1)).flatten()


def prepare_data_splits(
    data: pd.DataFrame,
    lookback: int = DEFAULT_LOOKBACK,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    target_col: str = 'Target_Price'
) -> Tuple[StockSequenceDataset, StockSequenceDataset, StockSequenceDataset, Dict]:
    """
    Prepare time-aware train/validation/test splits.

    Args:
        data: Processed DataFrame with features and targets
        lookback: Sequence length for LSTM
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        target_col: Target column name

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, metadata)
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]

    print(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Create datasets
    train_dataset = StockSequenceDataset(
        train_data, lookback=lookback, target_col=target_col, fit_scaler=True
    )

    # Use train scaler for val and test
    val_dataset = StockSequenceDataset(
        val_data, lookback=lookback, target_col=target_col,
        scaler=train_dataset.scaler, fit_scaler=False,
        feature_cols=train_dataset.feature_cols
    )
    val_dataset.target_scaler = train_dataset.target_scaler
    val_dataset.scaled_targets = train_dataset.target_scaler.transform(
        val_dataset.original_targets.reshape(-1, 1)
    ).flatten()
    val_dataset.X, val_dataset.y = val_dataset._create_sequences()

    test_dataset = StockSequenceDataset(
        test_data, lookback=lookback, target_col=target_col,
        scaler=train_dataset.scaler, fit_scaler=False,
        feature_cols=train_dataset.feature_cols
    )
    test_dataset.target_scaler = train_dataset.target_scaler
    test_dataset.scaled_targets = train_dataset.target_scaler.transform(
        test_dataset.original_targets.reshape(-1, 1)
    ).flatten()
    test_dataset.X, test_dataset.y = test_dataset._create_sequences()

    metadata = {
        'lookback': lookback,
        'n_features': len(train_dataset.feature_cols),
        'feature_cols': train_dataset.feature_cols,
        'target_col': target_col,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'train_dates': (str(train_data.index[0]), str(train_data.index[-1])),
        'val_dates': (str(val_data.index[0]), str(val_data.index[-1])),
        'test_dates': (str(test_data.index[0]), str(test_data.index[-1]))
    }

    return train_dataset, val_dataset, test_dataset, metadata


def save_scalers(
    train_dataset: StockSequenceDataset,
    ticker: str,
    model_version: str = 'v1'
) -> str:
    """
    Save scalers for later inference.

    Args:
        train_dataset: Training dataset with fitted scalers
        ticker: Stock ticker
        model_version: Version string

    Returns:
        Path to saved scalers
    """
    from config import MODELS_DIR

    scaler_path = MODELS_DIR / f"{ticker}_{model_version}_scalers.joblib"

    scalers = {
        'feature_scaler': train_dataset.scaler,
        'target_scaler': train_dataset.target_scaler,
        'feature_cols': train_dataset.feature_cols
    }

    joblib.dump(scalers, scaler_path)
    print(f"Scalers saved to {scaler_path}")

    return str(scaler_path)


def load_scalers(ticker: str, model_version: str = 'v1') -> Dict:
    """
    Load saved scalers.

    Args:
        ticker: Stock ticker
        model_version: Version string

    Returns:
        Dictionary with scalers
    """
    from config import MODELS_DIR

    scaler_path = MODELS_DIR / f"{ticker}_{model_version}_scalers.joblib"
    return joblib.load(scaler_path)


if __name__ == "__main__":
    # Test data fetching and processing
    fetcher = StockDataFetcher("AAPL", "2020-01-01", "2024-01-01")
    data = fetcher.fetch_data()
    data = fetcher.add_technical_indicators()
    data = fetcher.create_targets()
    fetcher.save_data()

    # Test dataset creation
    train_ds, val_ds, test_ds, meta = prepare_data_splits(data)
    print(f"\nDataset shapes:")
    print(f"Train: X={train_ds.X.shape}, y={train_ds.y.shape}")
    print(f"Val: X={val_ds.X.shape}, y={val_ds.y.shape}")
    print(f"Test: X={test_ds.X.shape}, y={test_ds.y.shape}")
