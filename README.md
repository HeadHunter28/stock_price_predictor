# 📈 MAG-7 Stock Price Predictor (Streamlit Cloud Version)

A standalone Streamlit application for predicting stock prices using deep learning models for the Magnificent 7 (MAG-7) tech stocks.

## 🌟 Features

- **Multiple Deep Learning Architectures**: LSTM, GRU, TCN, Transformer
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, and more
- **Interactive Training**: Train models directly in the browser
- **Real-time Predictions**: Generate next-day price forecasts
- **Comprehensive Evaluation**: RMSE, MAE, MAPE, Directional Accuracy

## 🚀 Deploy to Streamlit Cloud

### 1. Fork or Clone this Repository

```bash
git clone https://github.com/YOUR_USERNAME/stock-predictor-streamlit.git
cd stock-predictor-streamlit
```

### 2. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/stock-predictor-streamlit.git
git push -u origin main
```

### 3. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file path to `app.py`
6. Click "Deploy"

## 🏃 Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## 📁 Project Structure

```
stock-predictor-streamlit/
├── app.py                 # Main Streamlit application
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml        # Streamlit configuration
├── training/
│   ├── dataset.py         # Data fetching & feature engineering
│   ├── model.py           # Model architectures (LSTM, GRU, TCN, Transformer)
│   ├── train.py           # Training pipeline
│   └── evaluate.py        # Evaluation metrics & visualization
├── data/                  # Processed data storage
├── models/                # Saved model checkpoints
└── experiments/           # Training history
```

## 🧠 Model Architectures

| Model | Description | Best For |
|-------|-------------|----------|
| **LSTM** | Long Short-Term Memory | General purpose, long-term patterns |
| **GRU** | Gated Recurrent Unit | Fast training, smaller datasets |
| **TCN** | Temporal Convolutional Network | Production systems, speed |
| **Transformer** | Self-attention based | Complex patterns, large datasets |

## 📊 Supported Stocks (MAG-7)

- AAPL (Apple Inc.)
- MSFT (Microsoft Corporation)
- GOOGL (Alphabet Inc.)
- AMZN (Amazon.com Inc.)
- META (Meta Platforms Inc.)
- NVDA (NVIDIA Corporation)
- TSLA (Tesla Inc.)

## ⚠️ Important Notes for Streamlit Cloud

1. **Resource Limits**: Streamlit Cloud has memory limits. Training large models with many epochs may time out.
2. **No Persistent Storage**: Models trained on Streamlit Cloud are not saved permanently. For persistent storage, run locally or use external storage.
3. **CPU Only**: Streamlit Cloud runs on CPU, so training will be slower than with GPU.

### Recommended Settings for Cloud Deployment

- **Lookback**: 30-60 days
- **Hidden Size**: 64-128
- **Epochs**: 20-50
- **Batch Size**: 32

## ⚠️ Disclaimer

This tool is for **educational purposes only**. Stock price predictions are inherently uncertain and should not be used as the sole basis for investment decisions.

## 📄 License

MIT License
