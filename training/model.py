"""
Deep Learning Models for Stock Price Prediction.
Implements LSTM, GRU, Temporal CNN, and Transformer architectures.
"""
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from enum import Enum


class ModelType(str, Enum):
    """Supported model types."""
    LSTM = "lstm"
    GRU = "gru"
    TCN = "tcn"
    TRANSFORMER = "transformer"


class LSTMPredictor(nn.Module):
    """
    LSTM-based model for stock price prediction.

    Architecture:
    - Multi-layer LSTM with dropout
    - Fully connected output layer
    - Optional attention mechanism
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        use_attention: bool = False
    ):
        """
        Initialize LSTM predictor.

        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
            use_attention: Apply attention mechanism
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_directions = 2 if bidirectional else 1

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Attention layer
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * self.num_directions, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, features)

        Returns:
            Predictions of shape (batch, 1)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        if self.use_attention:
            # Attention weights
            attention_weights = torch.softmax(
                self.attention(lstm_out).squeeze(-1), dim=1
            )
            # Weighted sum
            context = torch.bmm(
                attention_weights.unsqueeze(1), lstm_out
            ).squeeze(1)
            out = self.dropout(context)
        else:
            # Use last hidden state
            if self.bidirectional:
                out = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                out = h_n[-1]
            out = self.dropout(out)

        return self.fc(out)


class GRUPredictor(nn.Module):
    """
    GRU-based model for stock price prediction.
    Similar to LSTM but with simpler gating mechanism.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        """
        Initialize GRU predictor.

        Args:
            input_size: Number of input features
            hidden_size: GRU hidden state size
            num_layers: Number of GRU layers
            dropout: Dropout probability
            bidirectional: Use bidirectional GRU
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        _, h_n = self.gru(x)

        if self.num_directions == 2:
            out = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            out = h_n[-1]

        out = self.dropout(out)
        return self.fc(out)


class TemporalBlock(nn.Module):
    """
    Temporal Convolutional Block with residual connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )

        self.chomp1 = padding
        self.chomp2 = padding

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = out[:, :, :-self.chomp1] if self.chomp1 > 0 else out
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, :-self.chomp2] if self.chomp2 > 0 else out
        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNPredictor(nn.Module):
    """
    Temporal Convolutional Network for stock prediction.
    Uses dilated causal convolutions.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        """
        Initialize TCN predictor.

        Args:
            input_size: Number of input features
            hidden_size: Number of channels in hidden layers
            num_layers: Number of temporal blocks
            kernel_size: Convolution kernel size
            dropout: Dropout probability
        """
        super().__init__()

        layers = []
        for i in range(num_layers):
            in_ch = input_size if i == 0 else hidden_size
            dilation = 2 ** i
            layers.append(
                TemporalBlock(in_ch, hidden_size, kernel_size, dilation, dropout)
            )

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, seq_len, features)

        Returns:
            Predictions of shape (batch, 1)
        """
        # Transpose for conv1d: (batch, features, seq_len)
        x = x.transpose(1, 2)
        out = self.tcn(x)
        # Take last timestep
        out = out[:, :, -1]
        return self.fc(out)


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    """
    Transformer-based model for stock price prediction.
    Uses self-attention mechanism.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 500
    ):
        """
        Initialize Transformer predictor.

        Args:
            input_size: Number of input features
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super().__init__()

        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, seq_len, features)

        Returns:
            Predictions of shape (batch, 1)
        """
        # Project input to d_model dimensions
        x = self.input_projection(x)
        x = self.pos_encoder(x)

        # Transformer encoding
        out = self.transformer_encoder(x)

        # Use last timestep for prediction
        out = out[:, -1, :]
        return self.fc(out)


def create_model(
    model_type: ModelType,
    input_size: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    **kwargs
) -> nn.Module:
    """
    Factory function to create a model.

    Args:
        model_type: Type of model to create
        input_size: Number of input features
        hidden_size: Hidden layer size
        num_layers: Number of layers
        dropout: Dropout probability
        **kwargs: Additional model-specific arguments

    Returns:
        Initialized model
    """
    if model_type == ModelType.LSTM:
        return LSTMPredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=kwargs.get('bidirectional', False),
            use_attention=kwargs.get('use_attention', False)
        )

    elif model_type == ModelType.GRU:
        return GRUPredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=kwargs.get('bidirectional', False)
        )

    elif model_type == ModelType.TCN:
        return TCNPredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            kernel_size=kwargs.get('kernel_size', 3),
            dropout=dropout
        )

    elif model_type == ModelType.TRANSFORMER:
        return TransformerPredictor(
            input_size=input_size,
            d_model=hidden_size,
            nhead=kwargs.get('nhead', 8),
            num_layers=num_layers,
            dim_feedforward=kwargs.get('dim_feedforward', 256),
            dropout=dropout
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test all models
    batch_size = 32
    seq_len = 60
    n_features = 19

    x = torch.randn(batch_size, seq_len, n_features)

    print("Testing models with input shape:", x.shape)

    for model_type in ModelType:
        model = create_model(model_type, input_size=n_features)
        out = model(x)
        print(f"{model_type.value}: output shape = {out.shape}, params = {sum(p.numel() for p in model.parameters()):,}")
