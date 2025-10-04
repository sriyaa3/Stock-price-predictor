import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Import custom modules
from data_fetcher import StockDataFetcher
from model import LSTMStockPredictor
from utils import save_model, load_model, export_to_csv

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

# Initialize session state
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'training_history' not in st.session_state:
    st.session_state.training_history = {}

def main():
    st.markdown('<h1 class="main-header">📈 Stock Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown("*Powered by LSTM Neural Networks | Predict 7 days ahead*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Stock selection
        st.subheader("Select Stocks")
        stock_input = st.text_input(
            "Enter stock symbols (comma-separated)",
            value="AAPL, GOOGL, MSFT",
            help="Enter valid stock ticker symbols separated by commas"
        )
        stocks = [s.strip().upper() for s in stock_input.split(',') if s.strip()]
        
        # Date range
        st.subheader("Data Range")
        today = dt.now().date()
        default_end = today
        default_start = today - timedelta(days=730)  # 2 years
        
        date_range = st.date_input(
            "Select date range",
            value=(default_start, default_end),
            max_value=today,
            help="Select historical date range (cannot be in the future)"
        )
        
        # Model hyperparameters
        st.subheader("Model Hyperparameters")
        with st.expander("Advanced Settings", expanded=False):
            epochs = st.slider("Training Epochs", 10, 100, 50, 5)
            batch_size = st.slider("Batch Size", 16, 128, 32, 16)
            lookback = st.slider("Lookback Period (days)", 30, 90, 60, 10)
            lstm_units = st.slider("LSTM Units", 32, 128, 64, 16)
        
        st.divider()
        
        # Action buttons
        fetch_button = st.button("📊 Fetch Data", use_container_width=True)
        train_button = st.button("🚀 Train Models", use_container_width=True, type="primary")
        predict_button = st.button("🔮 Generate Predictions", use_container_width=True)
    
    # Main content area
    tabs = st.tabs(["📊 Data Overview", "🎯 Training & Predictions", "📈 Comparison", "💾 Export"])
    
    # Tab 1: Data Overview
    with tabs[0]:
        st.header("Historical Stock Data")
        
        if fetch_button:
            with st.spinner("Fetching stock data..."):
                fetcher = StockDataFetcher()
                progress_bar = st.progress(0)
                
                # Ensure dates are not in future
                end_dt = date_range[1] if len(date_range) > 1 else date_range[0]
                today = dt.now().date()
                
                if end_dt > today:
                    st.warning(f"⚠️ End date adjusted to today ({today}) as future dates are not available.")
                    end_dt = today
                
                start_dt = date_range[0]
                
                for idx, symbol in enumerate(stocks):
                    try:
                        df = fetcher.fetch_stock_data(
                            symbol, 
                            start_date=start_dt.strftime('%Y-%m-%d'),
                            end_date=end_dt.strftime('%Y-%m-%d')
                        )
                        st.session_state[f'data_{symbol}'] = df
                        st.success(f"✅ Fetched {len(df)} days of data for {symbol}")
                    except Exception as e:
                        st.error(f"❌ Error fetching {symbol}: {str(e)}")
                        st.info(f"💡 Tip: Make sure '{symbol}' is a valid stock ticker symbol")
                    
                    progress_bar.progress((idx + 1) / len(stocks))
        
        # Display data for each stock
        for symbol in stocks:
            if f'data_{symbol}' in st.session_state:
                with st.expander(f"📊 {symbol} Data", expanded=True):
                    df = st.session_state[f'data_{symbol}']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Days", len(df))
                    with col2:
                        st.metric("Latest Price", f"${df['Close'].iloc[-1]:.2f}")
                    with col3:
                        change = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100)
                        st.metric("Period Change", f"{change:.2f}%")
                    with col4:
                        st.metric("Avg Volume", f"{df['Volume'].mean()/1e6:.2f}M")
                    
                    # Price chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df.index, 
                        y=df['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    fig.update_layout(
                        title=f"{symbol} Historical Prices",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        hovermode='x unified',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Data table
                    st.dataframe(df.tail(10), use_container_width=True)
    
    # Tab 2: Training & Predictions
    with tabs[1]:
        st.header("Model Training & Predictions")
        
        if train_button:
            if not any(f'data_{symbol}' in st.session_state for symbol in stocks):
                st.warning("⚠️ Please fetch data first!")
            else:
                for symbol in stocks:
                    if f'data_{symbol}' in st.session_state:
                        with st.expander(f"🎯 Training {symbol}", expanded=True):
                            df = st.session_state[f'data_{symbol}']
                            
                            progress_text = st.empty()
                            progress_bar = st.progress(0)
                            
                            try:
                                # Initialize and train model
                                model = LSTMStockPredictor(
                                    lookback_period=lookback,
                                    lstm_units=lstm_units
                                )
                                
                                progress_text.text("Preparing data...")
                                progress_bar.progress(0.2)
                                
                                history = model.train(
                                    df['Close'].values,
                                    epochs=epochs,
                                    batch_size=batch_size
                                )
                                
                                progress_text.text("Training complete!")
                                progress_bar.progress(1.0)
                                
                                # Save model
                                st.session_state.trained_models[symbol] = model
                                st.session_state.training_history[symbol] = history
                                save_model(model, symbol)
                                
                                # Display metrics
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Training Loss", f"{history['train_loss'][-1]:.4f}")
                                with col2:
                                    st.metric("Validation Loss", f"{history['val_loss'][-1]:.4f}")
                                
                                # Plot training history
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    y=history['train_loss'],
                                    name='Training Loss',
                                    mode='lines'
                                ))
                                fig.add_trace(go.Scatter(
                                    y=history['val_loss'],
                                    name='Validation Loss',
                                    mode='lines'
                                ))
                                fig.update_layout(
                                    title=f"{symbol} Training History",
                                    xaxis_title="Epoch",
                                    yaxis_title="Loss (MSE)",
                                    height=300
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.success(f"✅ Model trained successfully for {symbol}!")
                                
                            except Exception as e:
                                st.error(f"❌ Training failed: {str(e)}")
                                progress_bar.empty()
        
        if predict_button:
            if not st.session_state.trained_models:
                st.warning("⚠️ Please train models first!")
            else:
                st.subheader("7-Day Predictions")
                
                for symbol in stocks:
                    if symbol in st.session_state.trained_models:
                        with st.expander(f"🔮 {symbol} Predictions", expanded=True):
                            model = st.session_state.trained_models[symbol]
                            df = st.session_state[f'data_{symbol}']
                            
                            # Generate predictions
                            predictions = model.predict_future(df['Close'].values, days=7)
                            st.session_state.predictions[symbol] = predictions
                            
                            # Create prediction dates
                            last_date = df.index[-1]
                            pred_dates = pd.date_range(
                                start=last_date + timedelta(days=1),
                                periods=7,
                                freq='D'
                            )
                            
                            # Metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
                            with col2:
                                st.metric("7-Day Prediction", f"${predictions[-1]:.2f}")
                            with col3:
                                change = ((predictions[-1] - df['Close'].iloc[-1]) / df['Close'].iloc[-1] * 100)
                                st.metric("Expected Change", f"{change:+.2f}%", delta=f"{change:+.2f}%")
                            
                            # Visualization
                            fig = go.Figure()
                            
                            # Historical data (last 30 days)
                            historical = df['Close'].tail(30)
                            fig.add_trace(go.Scatter(
                                x=historical.index,
                                y=historical.values,
                                mode='lines',
                                name='Historical',
                                line=dict(color='#1f77b4', width=2)
                            ))
                            
                            # Predictions
                            fig.add_trace(go.Scatter(
                                x=pred_dates,
                                y=predictions,
                                mode='lines+markers',
                                name='Predicted',
                                line=dict(color='#ff7f0e', width=2, dash='dash'),
                                marker=dict(size=8)
                            ))
                            
                            fig.update_layout(
                                title=f"{symbol} - 7 Day Forecast",
                                xaxis_title="Date",
                                yaxis_title="Price (USD)",
                                hovermode='x unified',
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Prediction table
                            pred_df = pd.DataFrame({
                                'Date': pred_dates,
                                'Predicted Price': [f"${p:.2f}" for p in predictions]
                            })
                            st.dataframe(pred_df, use_container_width=True, hide_index=True)
    
    # Tab 3: Comparison
    with tabs[2]:
        st.header("Multi-Stock Comparison")
        
        if st.session_state.predictions:
            # Comparison metrics
            st.subheader("Expected 7-Day Changes")
            
            comparison_data = []
            for symbol in stocks:
                if symbol in st.session_state.predictions and f'data_{symbol}' in st.session_state:
                    current = st.session_state[f'data_{symbol}']['Close'].iloc[-1]
                    predicted = st.session_state.predictions[symbol][-1]
                    change = ((predicted - current) / current * 100)
                    comparison_data.append({
                        'Symbol': symbol,
                        'Current Price': current,
                        'Predicted Price': predicted,
                        'Change (%)': change
                    })
            
            if comparison_data:
                comp_df = pd.DataFrame(comparison_data)
                comp_df = comp_df.sort_values('Change (%)', ascending=False)
                
                # Display as cards
                cols = st.columns(len(comparison_data))
                for idx, row in comp_df.iterrows():
                    with cols[comparison_data.index(comp_df.to_dict('records')[idx])]:
                        delta_color = "normal" if row['Change (%)'] >= 0 else "inverse"
                        st.metric(
                            label=row['Symbol'],
                            value=f"${row['Predicted Price']:.2f}",
                            delta=f"{row['Change (%)']:+.2f}%"
                        )
                
                # Comparison chart
                fig = go.Figure()
                
                for symbol in stocks:
                    if symbol in st.session_state.predictions:
                        predictions = st.session_state.predictions[symbol]
                        pred_dates = pd.date_range(
                            start=dt.now() + timedelta(days=1),
                            periods=7,
                            freq='D'
                        )
                        fig.add_trace(go.Scatter(
                            x=pred_dates,
                            y=predictions,
                            mode='lines+markers',
                            name=symbol,
                            line=dict(width=2),
                            marker=dict(size=6)
                        ))
                
                fig.update_layout(
                    title="7-Day Prediction Comparison",
                    xaxis_title="Date",
                    yaxis_title="Predicted Price (USD)",
                    hovermode='x unified',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Table comparison
                st.dataframe(comp_df.style.background_gradient(subset=['Change (%)'], cmap='RdYlGn'), 
                           use_container_width=True, hide_index=True)
        else:
            st.info("👆 Generate predictions first to see comparisons")
    
    # Tab 4: Export
    with tabs[3]:
        st.header("Export Data")
        
        if st.session_state.predictions:
            export_options = st.multiselect(
                "Select stocks to export",
                stocks,
                default=stocks
            )
            
            if st.button("📥 Download Predictions (CSV)", use_container_width=True):
                all_data = []
                
                for symbol in export_options:
                    if symbol in st.session_state.predictions:
                        predictions = st.session_state.predictions[symbol]
                        pred_dates = pd.date_range(
                            start=datetime.now() + timedelta(days=1),
                            periods=7,
                            freq='D'
                        )
                        
                        for date, price in zip(pred_dates, predictions):
                            all_data.append({
                                'Symbol': symbol,
                                'Date': date.strftime('%Y-%m-%d'),
                                'Predicted Price': price
                            })
                
                export_df = pd.DataFrame(all_data)
                csv = export_df.to_csv(index=False)
                
                st.download_button(
                    label="💾 Download CSV",
                    data=csv,
                    file_name=f"stock_predictions_{dt.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                st.success("✅ Ready to download!")
                st.dataframe(export_df, use_container_width=True, hide_index=True)
        else:
            st.info("👆 Generate predictions first to export data")
    
    # Footer
    st.divider()
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Built with ❤️ using Streamlit | LSTM Neural Networks | Yahoo Finance API</p>
            <p><small>Disclaimer: Predictions are for educational purposes only. Not financial advice.</small></p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()