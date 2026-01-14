"""
Training pipeline for stock prediction models.
Simplified version without MLflow for Streamlit Cloud deployment.
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import joblib

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    MODELS_DIR, EXPERIMENTS_DIR, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE, DEFAULT_HIDDEN_SIZE, DEFAULT_NUM_LAYERS,
    DEFAULT_DROPOUT, DEFAULT_LOOKBACK
)
from training.model import create_model, ModelType
from training.dataset import (
    StockDataFetcher, prepare_data_splits, save_scalers
)


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class ModelTrainer:
    """
    Handles model training, validation, and saving.
    Simplified version without MLflow for Streamlit Cloud.
    """

    def __init__(
        self,
        ticker: str,
        model_type: ModelType = ModelType.LSTM,
        lookback: int = DEFAULT_LOOKBACK,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        epochs: int = DEFAULT_EPOCHS,
        target_col: str = 'Target_Price',
        device: Optional[str] = None,
        model_version: str = 'v1',
        progress_callback=None,
        **model_kwargs
    ):
        """Initialize the trainer."""
        self.ticker = ticker
        self.model_type = model_type
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.target_col = target_col
        self.model_version = model_version
        self.model_kwargs = model_kwargs
        self.progress_callback = progress_callback

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Will be initialized later
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.metadata = None
        self.history = {'train_loss': [], 'val_loss': []}

    def prepare_data(
        self,
        start_date: str,
        end_date: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Dict:
        """Fetch and prepare data for training."""
        # Fetch and process data
        fetcher = StockDataFetcher(self.ticker, start_date, end_date)
        data = fetcher.fetch_data()
        data = fetcher.add_technical_indicators()
        data = fetcher.create_targets()
        fetcher.save_data()

        # Create train/val/test splits
        self.train_dataset, self.val_dataset, self.test_dataset, self.metadata = \
            prepare_data_splits(
                data,
                lookback=self.lookback,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                target_col=self.target_col
            )

        # Save scalers
        save_scalers(self.train_dataset, self.ticker, self.model_version)

        # Store additional metadata
        self.metadata.update({
            'ticker': self.ticker,
            'model_type': self.model_type.value,
            'start_date': start_date,
            'end_date': end_date,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs
        })

        return self.metadata

    def build_model(self) -> nn.Module:
        """Build the model based on configuration."""
        if self.train_dataset is None:
            raise ValueError("Must prepare data first")

        n_features = self.metadata['n_features']

        self.model = create_model(
            model_type=self.model_type,
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            **self.model_kwargs
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.metadata['total_params'] = total_params
        self.metadata['trainable_params'] = trainable_params

        return self.model

    def train(self, patience: int = 15, scheduler_type: str = 'plateau') -> Dict:
        """Train the model."""
        if self.model is None:
            raise ValueError("Must build model first")

        start_time = time.time()

        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )

        # Learning rate scheduler
        if scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)

        # Early stopping
        early_stopping = EarlyStopping(patience=patience)

        # Training loop
        best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': []}

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_losses = []

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(batch_x)
                loss = criterion(predictions, batch_y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(loss.item())

            # Validation phase
            val_loss = self._validate(val_loader, criterion)

            # Record history
            avg_train_loss = np.mean(train_losses)
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss)

            # Progress callback for Streamlit
            if self.progress_callback:
                self.progress_callback(
                    epoch + 1,
                    self.epochs,
                    avg_train_loss,
                    val_loss
                )

            # Learning rate scheduling
            if scheduler_type == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_model('best')

            # Early stopping check
            if early_stopping(val_loss):
                break

        # Training complete
        training_time = time.time() - start_time

        # Save final model
        self._save_model('final')

        # Save training history
        self._save_history()

        results = {
            'best_val_loss': best_val_loss,
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'epochs_trained': len(self.history['train_loss']),
            'training_time_seconds': training_time,
            'early_stopped': early_stopping.early_stop
        }

        return results

    def _validate(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                predictions = self.model(batch_x)
                loss = criterion(predictions, batch_y)
                val_losses.append(loss.item())

        return np.mean(val_losses)

    def _save_model(self, suffix: str = 'final') -> str:
        """Save model checkpoint."""
        model_path = MODELS_DIR / f"{self.ticker}_{self.model_version}_{suffix}.pt"

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type.value,
            'input_size': self.metadata['n_features'],
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'lookback': self.lookback,
            'feature_cols': self.metadata['feature_cols'],
            'target_col': self.target_col,
            'model_kwargs': self.model_kwargs,
            'metadata': self.metadata
        }

        torch.save(checkpoint, model_path)
        return str(model_path)

    def _save_history(self) -> str:
        """Save training history."""
        history_path = EXPERIMENTS_DIR / f"{self.ticker}_{self.model_version}_history.json"

        with open(history_path, 'w') as f:
            json.dump({
                'history': self.history,
                'metadata': {k: str(v) if not isinstance(v, (int, float, list, dict, str, bool)) else v
                            for k, v in self.metadata.items()}
            }, f, indent=2)

        return str(history_path)


def load_trained_model(
    ticker: str,
    model_version: str = 'v1',
    suffix: str = 'best',
    device: Optional[str] = None
) -> Tuple[nn.Module, Dict, Dict]:
    """Load a trained model with its scalers and metadata."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model checkpoint
    model_path = MODELS_DIR / f"{ticker}_{model_version}_{suffix}.pt"
    checkpoint = torch.load(model_path, map_location=device)

    # Create and load model
    model = create_model(
        model_type=ModelType(checkpoint['model_type']),
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint['dropout'],
        **checkpoint.get('model_kwargs', {})
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load scalers
    scaler_path = MODELS_DIR / f"{ticker}_{model_version}_scalers.joblib"
    scalers = joblib.load(scaler_path)

    metadata = checkpoint.get('metadata', {})

    return model, scalers, metadata


def get_trained_models() -> list:
    """Get list of trained models."""
    models = []
    for model_file in MODELS_DIR.glob("*_best.pt"):
        parts = model_file.stem.split('_')
        if len(parts) >= 2:
            ticker = parts[0]
            version = parts[1]
            models.append({'ticker': ticker, 'version': version, 'path': str(model_file)})
    return models
