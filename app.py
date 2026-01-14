"""
Streamlit Frontend for Stock Price Prediction System.
Standalone version that runs without a separate API server.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import base64
from io import BytesIO
from PIL import Image
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import MAG7_STOCKS, MODELS_DIR, EXPERIMENTS_DIR
from training.model import ModelType
from training.train import ModelTrainer, load_trained_model, get_trained_models
from training.evaluate import ModelEvaluator, predict_next_day

# Model architectures with display names
MODEL_TYPES = {
    "lstm": "LSTM (Long Short-Term Memory)",
    "gru": "GRU (Gated Recurrent Unit)",
    "tcn": "TCN (Temporal Convolutional Network)",
    "transformer": "Transformer"
}

# Page configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def display_base64_image(b64_string: str, caption: str = ""):
    """Display a base64 encoded image."""
    if b64_string:
        image_data = base64.b64decode(b64_string)
        image = Image.open(BytesIO(image_data))
        st.image(image, caption=caption, use_container_width=True)


def get_system_status():
    """Get system status - trained models, etc."""
    trained_models = get_trained_models()
    return {
        'available_tickers': list(MAG7_STOCKS.keys()),
        'trained_models': trained_models
    }


# Sidebar
with st.sidebar:
    st.title("📈 Stock Predictor")
    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🎯 Train Model", "📊 Evaluate", "🔮 Predict", "📈 Dashboard", "📚 Model Guide"],
        index=0
    )

    st.markdown("---")
    st.markdown("### MAG-7 Stocks")
    for symbol, name in MAG7_STOCKS.items():
        st.text(f"{symbol}: {name}")


# Main content
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">📈 MAG-7 Stock Price Predictor</h1>', unsafe_allow_html=True)

    st.markdown("""
    Welcome to the **MAG-7 Stock Price Prediction System**! This application uses deep learning
    models to predict stock prices for the Magnificent 7 tech stocks.

    ### Features:
    - 🧠 **Multiple Model Architectures**: LSTM, GRU, TCN, Transformer
    - 📊 **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, and more
    - 🎯 **Next-Day Predictions**: Price and return forecasts
    - 📈 **Comprehensive Evaluation**: RMSE, MAE, MAPE, Directional Accuracy

    ### Getting Started:
    1. **Train a Model**: Select a stock and configure training parameters
    2. **Evaluate Performance**: View metrics and visualizations
    3. **Make Predictions**: Get next-day price forecasts
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("🎯 **Train Model**\n\nTrain a new deep learning model on historical stock data.")

    with col2:
        st.info("📊 **Evaluate**\n\nAnalyze model performance with comprehensive metrics.")

    with col3:
        st.info("🔮 **Predict**\n\nGenerate next-day price predictions for any MAG-7 stock.")

    # System Status
    st.markdown("---")
    st.subheader("System Status")

    status_data = get_system_status()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Available Tickers", len(status_data.get('available_tickers', [])))

    with col2:
        st.metric("Trained Models", len(status_data.get('trained_models', [])))

    if status_data.get('trained_models'):
        st.markdown("#### Trained Models")
        models_df = pd.DataFrame(status_data['trained_models'])
        st.dataframe(models_df, use_container_width=True)


elif page == "🎯 Train Model":
    st.header("🎯 Train a New Model")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Stock Selection")
        ticker = st.selectbox(
            "Select Stock",
            options=list(MAG7_STOCKS.keys()),
            format_func=lambda x: f"{x} - {MAG7_STOCKS[x]}"
        )

        st.subheader("Date Range")
        default_start = datetime.now() - timedelta(days=365*5)
        default_end = datetime.now()

        start_date = st.date_input("Start Date", value=default_start)
        end_date = st.date_input("End Date", value=default_end)

        st.subheader("Model Configuration")
        model_type = st.selectbox(
            "Model Architecture",
            options=list(MODEL_TYPES.keys()),
            format_func=lambda x: MODEL_TYPES[x],
            index=0
        )

        target_type = st.selectbox(
            "Prediction Target",
            options=["Target_Price", "Target_Return"],
            format_func=lambda x: "Next-Day Price" if x == "Target_Price" else "Next-Day Return"
        )

    with col2:
        st.subheader("Hyperparameters")

        lookback = st.slider("Lookback Window (days)", min_value=10, max_value=120, value=60)
        hidden_size = st.select_slider("Hidden Size", options=[32, 64, 128, 256, 512], value=128)
        num_layers = st.slider("Number of Layers", min_value=1, max_value=4, value=2)
        dropout = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.2, step=0.05)

        st.subheader("Training Parameters")
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.001
        )
        batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
        epochs = st.slider("Max Epochs", min_value=10, max_value=200, value=100)
        patience = st.slider("Early Stopping Patience", min_value=5, max_value=30, value=15)

        model_version = st.text_input("Model Version", value="v1")

    st.markdown("---")

    # Training button
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        # Progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_placeholder = st.empty()

        def progress_callback(epoch, total_epochs, train_loss, val_loss):
            progress = epoch / total_epochs
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch}/{total_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        try:
            status_text.text("Initializing trainer...")

            trainer = ModelTrainer(
                ticker=ticker,
                model_type=ModelType(model_type),
                lookback=lookback,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs,
                target_col=target_type,
                model_version=model_version,
                progress_callback=progress_callback
            )

            status_text.text("Preparing data...")
            trainer.prepare_data(str(start_date), str(end_date))

            status_text.text("Building model...")
            trainer.build_model()

            status_text.text("Training model...")
            results = trainer.train(patience=patience)

            progress_bar.progress(1.0)
            status_text.text("Training complete!")

            st.success("✅ Training Complete!")

            # Display results
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Final Train Loss", f"{results['final_train_loss']:.6f}")
            with col2:
                st.metric("Final Val Loss", f"{results['final_val_loss']:.6f}")
            with col3:
                st.metric("Epochs Trained", results['epochs_trained'])
            with col4:
                st.metric("Training Time", f"{results['training_time_seconds']:.1f}s")

            if results.get('early_stopped'):
                st.info("⚡ Training stopped early due to no improvement")

        except Exception as e:
            st.error(f"Training failed: {str(e)}")


elif page == "📊 Evaluate":
    st.header("📊 Model Evaluation")

    col1, col2 = st.columns([1, 2])

    with col1:
        ticker = st.selectbox(
            "Select Stock",
            options=list(MAG7_STOCKS.keys()),
            format_func=lambda x: f"{x} - {MAG7_STOCKS[x]}",
            key="eval_ticker"
        )

        model_version = st.text_input("Model Version", value="v1", key="eval_version")

        include_plots = st.checkbox("Include Visualizations", value=True)

        evaluate_btn = st.button("📊 Evaluate Model", type="primary", use_container_width=True)

    with col2:
        if evaluate_btn:
            with st.spinner("Evaluating model..."):
                try:
                    evaluator = ModelEvaluator(ticker, model_version)
                    metrics = evaluator.evaluate()

                    st.success("✅ Evaluation Complete!")

                    # Display metrics
                    st.subheader("Performance Metrics")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("RMSE", f"${metrics['rmse']:.2f}")
                    with col2:
                        st.metric("MAE", f"${metrics['mae']:.2f}")
                    with col3:
                        st.metric("MAPE", f"{metrics['mape']:.2f}%")
                    with col4:
                        st.metric("Direction Accuracy", f"{metrics['directional_accuracy']:.1f}%")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Correlation", f"{metrics['correlation']:.4f}")
                    with col2:
                        st.metric("Buy & Hold Return", f"{metrics['buy_hold_return']:.2f}%")
                    with col3:
                        st.metric("Strategy Return", f"{metrics['strategy_return']:.2f}%")
                    with col4:
                        st.metric("Strategy Sharpe", f"{metrics['strategy_sharpe']:.2f}")

                    # Display plots
                    if include_plots:
                        st.subheader("Visualizations")

                        st.markdown("#### Predicted vs Actual Prices")
                        pred_img = evaluator.plot_predictions()
                        display_base64_image(pred_img)

                        st.markdown("#### Cumulative Returns Comparison")
                        returns_img = evaluator.plot_cumulative_returns()
                        display_base64_image(returns_img)

                        try:
                            st.markdown("#### Training Loss Curves")
                            history_img = evaluator.plot_training_history()
                            display_base64_image(history_img)
                        except FileNotFoundError:
                            st.info("Training history not available")

                except FileNotFoundError:
                    st.warning(f"No trained model found for {ticker} version {model_version}. Please train a model first.")
                except Exception as e:
                    st.error(f"Evaluation failed: {str(e)}")


elif page == "🔮 Predict":
    st.header("🔮 Price Predictions")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Single Prediction")
        ticker = st.selectbox(
            "Select Stock",
            options=list(MAG7_STOCKS.keys()),
            format_func=lambda x: f"{x} - {MAG7_STOCKS[x]}",
            key="predict_ticker"
        )

        model_version = st.text_input("Model Version", value="v1", key="predict_version")

        predict_btn = st.button("🔮 Generate Prediction", type="primary", use_container_width=True)

    with col2:
        if predict_btn:
            with st.spinner("Generating prediction..."):
                try:
                    result = predict_next_day(ticker, model_version)

                    st.success("✅ Prediction Generated!")

                    # Display prediction
                    direction_emoji = "📈" if result['prediction_direction'] == "UP" else "📉"

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "Current Price",
                            f"${result['current_price']:.2f}"
                        )
                        st.metric(
                            "Predicted Price",
                            f"${result['predicted_price']:.2f}",
                            f"{result['predicted_return_pct']:+.2f}%"
                        )

                    with col2:
                        st.markdown(f"""
                        ### Prediction: {direction_emoji} {result['prediction_direction']}

                        **Confidence Interval (95%)**
                        - Lower: ${result['confidence_interval']['lower']:.2f}
                        - Upper: ${result['confidence_interval']['upper']:.2f}

                        **Last Data Date**: {result['last_data_date'][:10]}
                        """)

                except FileNotFoundError:
                    st.warning(f"No trained model found for {ticker}. Please train a model first.")
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

    st.markdown("---")

    # Batch predictions
    st.subheader("Batch Predictions")
    selected_tickers = st.multiselect(
        "Select Multiple Stocks",
        options=list(MAG7_STOCKS.keys()),
        default=["AAPL", "MSFT"]
    )

    if st.button("🔮 Generate Batch Predictions", use_container_width=True):
        if selected_tickers:
            with st.spinner("Generating predictions..."):
                predictions = []
                for t in selected_tickers:
                    try:
                        pred = predict_next_day(t, "v1")
                        predictions.append({
                            'Ticker': t,
                            'Current': f"${pred['current_price']:.2f}",
                            'Predicted': f"${pred['predicted_price']:.2f}",
                            'Return': f"{pred['predicted_return_pct']:+.2f}%",
                            'Direction': pred['prediction_direction']
                        })
                    except Exception as e:
                        predictions.append({
                            'Ticker': t,
                            'Current': 'N/A',
                            'Predicted': 'N/A',
                            'Return': 'N/A',
                            'Direction': f'Error: {str(e)[:20]}'
                        })

                df = pd.DataFrame(predictions)
                st.dataframe(df, use_container_width=True)


elif page == "📈 Dashboard":
    st.header("📈 Stock Dashboard")

    col1, col2 = st.columns([1, 3])

    with col1:
        ticker = st.selectbox(
            "Select Stock",
            options=list(MAG7_STOCKS.keys()),
            format_func=lambda x: f"{x} - {MAG7_STOCKS[x]}",
            key="dashboard_ticker"
        )

        period = st.selectbox(
            "Time Period",
            options=["1M", "3M", "6M", "1Y", "2Y", "5Y"],
            index=3
        )

    with col2:
        try:
            import yfinance as yf

            # Fetch data
            stock = yf.Ticker(ticker)
            data = stock.history(period=period.lower())

            if not data.empty:
                # Price chart
                fig = go.Figure()

                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price'
                ))

                fig.update_layout(
                    title=f"{ticker} - {MAG7_STOCKS[ticker]}",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=500,
                    xaxis_rangeslider_visible=False
                )

                st.plotly_chart(fig, use_container_width=True)

                # Volume chart
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color='lightblue'
                ))
                fig_vol.update_layout(
                    title="Trading Volume",
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    height=200
                )
                st.plotly_chart(fig_vol, use_container_width=True)

                # Stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
                with col2:
                    change = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
                    st.metric("Period Return", f"{change:+.2f}%")
                with col3:
                    st.metric("High", f"${data['High'].max():.2f}")
                with col4:
                    st.metric("Low", f"${data['Low'].min():.2f}")

        except Exception as e:
            st.error(f"Error loading stock data: {e}")


elif page == "📚 Model Guide":
    st.header("📚 Model Architecture Guide")

    st.markdown("""
    This guide explains the different deep learning architectures available for stock price prediction.
    Each model has unique strengths and is suited to different scenarios.
    """)

    st.markdown("---")

    # LSTM Section
    st.subheader("🧠 LSTM (Long Short-Term Memory)")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Overview:**
        LSTM is a type of Recurrent Neural Network (RNN) designed to learn long-term dependencies
        in sequential data. It uses a sophisticated gating mechanism to control information flow.

        **How it Works:**
        - **Forget Gate:** Decides what information to discard from the cell state
        - **Input Gate:** Decides what new information to store in the cell state
        - **Output Gate:** Decides what parts of the cell state to output

        **Strengths:**
        - Excellent at capturing long-term patterns in time series
        - Handles vanishing gradient problem well
        - Proven track record in financial forecasting

        **Best For:**
        - Data with long-term trends and seasonality
        - When historical patterns significantly influence future prices
        - General-purpose time series prediction
        """)

    with col2:
        st.info("""
        **Recommended Settings:**
        - Lookback: 60-90 days
        - Hidden Size: 128-256
        - Layers: 2-3
        - Dropout: 0.2-0.3
        """)

    st.markdown("---")

    # GRU Section
    st.subheader("⚡ GRU (Gated Recurrent Unit)")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Overview:**
        GRU is a simplified version of LSTM with fewer gates, making it faster to train
        while maintaining similar performance for many tasks.

        **How it Works:**
        - **Reset Gate:** Controls how much of the past information to forget
        - **Update Gate:** Controls how much of the past information to keep

        **Strengths:**
        - Faster training than LSTM (fewer parameters)
        - Often performs similarly to LSTM with less computation
        - Better for smaller datasets where LSTM might overfit

        **Best For:**
        - When training time is a concern
        - Smaller datasets
        - Simpler temporal patterns
        - Quick experimentation and prototyping
        """)

    with col2:
        st.info("""
        **Recommended Settings:**
        - Lookback: 30-60 days
        - Hidden Size: 64-128
        - Layers: 2
        - Dropout: 0.1-0.2
        """)

    st.markdown("---")

    # TCN Section
    st.subheader("🌊 TCN (Temporal Convolutional Network)")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Overview:**
        TCN uses dilated causal convolutions to process sequential data. Unlike RNNs,
        it can process sequences in parallel, making it highly efficient.

        **How it Works:**
        - **Causal Convolutions:** Ensures predictions only depend on past data
        - **Dilated Convolutions:** Exponentially increases receptive field without increasing parameters
        - **Residual Connections:** Helps train deeper networks

        **Strengths:**
        - Parallelizable - much faster training than RNNs
        - Very long effective memory through dilations
        - Stable gradients - easier to train
        - No vanishing gradient problem

        **Best For:**
        - Large datasets where training speed matters
        - Very long sequences
        - When you need consistent inference speed
        - Production environments
        """)

    with col2:
        st.info("""
        **Recommended Settings:**
        - Lookback: 60-120 days
        - Hidden Size: 64-128
        - Layers: 3-4
        - Dropout: 0.2
        """)

    st.markdown("---")

    # Transformer Section
    st.subheader("🤖 Transformer")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Overview:**
        Transformers use self-attention mechanisms to process entire sequences at once,
        capturing complex relationships between any two time steps regardless of distance.

        **How it Works:**
        - **Self-Attention:** Weighs the importance of different time steps
        - **Multi-Head Attention:** Captures different types of relationships
        - **Positional Encoding:** Adds information about sequence order
        - **Feed-Forward Networks:** Processes attended features

        **Strengths:**
        - Captures complex, non-linear relationships
        - Handles very long-range dependencies
        - State-of-the-art for many sequence tasks
        - Highly parallelizable

        **Best For:**
        - Complex market dynamics
        - When relationships between distant time points matter
        - Larger datasets (needs more data to train well)
        - Capturing market regime changes
        """)

    with col2:
        st.info("""
        **Recommended Settings:**
        - Lookback: 60-90 days
        - Hidden Size: 128-256
        - Layers: 2-4
        - Dropout: 0.1-0.2
        """)

    st.markdown("---")

    # Comparison Table
    st.subheader("📊 Model Comparison")

    comparison_data = {
        "Feature": ["Training Speed", "Memory Efficiency", "Long-term Patterns", "Short-term Patterns",
                   "Data Requirements", "Interpretability", "Best Use Case"],
        "LSTM": ["Medium", "Medium", "Excellent", "Good", "Medium", "Low", "General purpose"],
        "GRU": ["Fast", "Good", "Good", "Good", "Low", "Low", "Quick experiments"],
        "TCN": ["Very Fast", "Excellent", "Excellent", "Excellent", "Medium", "Medium", "Production systems"],
        "Transformer": ["Medium", "Low", "Excellent", "Good", "High", "Medium", "Complex patterns"]
    }

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Tips Section
    st.subheader("💡 Tips for Model Selection")

    st.markdown("""
    1. **Start with LSTM or GRU** - These are reliable baselines for most stock prediction tasks

    2. **Use TCN for speed** - If you need fast training and inference, TCN is often the best choice

    3. **Try Transformer for complex patterns** - When simpler models plateau, Transformers can capture
       more nuanced relationships

    4. **Consider your data size:**
       - Small data (< 1 year): GRU
       - Medium data (1-3 years): LSTM or TCN
       - Large data (3+ years): Transformer or TCN

    5. **Experiment!** - The best model depends on the specific stock and market conditions.
    """)


# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888;">
        Stock Price Predictor v1.0 | Built with Streamlit & PyTorch
        <br>
        <small>Disclaimer: This tool is for educational purposes only. Not financial advice.</small>
    </div>
    """,
    unsafe_allow_html=True
)
