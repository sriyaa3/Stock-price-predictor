# 📈 Stock Price Predictor

An AI-powered web application that forecasts stock prices using LSTM neural networks. Built with Streamlit, TensorFlow, and Yahoo Finance API.

## 🌟 Features

- **Multi-Stock Support**: Compare predictions for multiple stocks simultaneously
- **7-Day Forecasting**: Predict stock prices up to 7 days ahead
- **Interactive Visualization**: Beautiful charts with Plotly
- **Configurable Training**: Adjust epochs, batch size, and lookback period
- **Real-Time Data**: Fetch live stock data from Yahoo Finance
- **Model Persistence**: Save and load trained models locally
- **CSV Export**: Download predictions for analysis
- **Performance Metrics**: View training loss, validation metrics, and predictions

## 🚀 Quick Start

### Local Installation

1. **Clone or download the project**
```bash
git clone <your-repo-url>
cd stock-price-predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Open your browser**
The app will automatically open at `http://localhost:8501`

### Project Structure

```
stock-price-predictor/
├── app.py              # Main Streamlit application
├── model.py            # LSTM model implementation
├── data_fetcher.py     # Yahoo Finance data fetching
├── utils.py            # Helper functions
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── models/            # Saved models (auto-created)
└── exports/           # CSV exports (auto-created)
```

## 📊 How to Use

### 1. Configure Settings
- Enter stock symbols (e.g., `AAPL, GOOGL, MSFT`)
- Select date range for historical data
- Adjust model hyperparameters if needed

### 2. Fetch Data
- Click "📊 Fetch Data" to download historical stock prices
- View historical price charts and statistics

### 3. Train Models
- Click "🚀 Train Models" to train LSTM networks
- Monitor training progress and loss curves
- Models are automatically saved locally

### 4. Generate Predictions
- Click "🔮 Generate Predictions" for 7-day forecasts
- View predictions with interactive charts
- See expected price changes

### 5. Compare & Export
- Compare multiple stocks side-by-side
- Export predictions to CSV for further analysis

## 🔧 Configuration Options

### Model Hyperparameters

- **Epochs** (10-100): Number of training iterations
  - More epochs = Better learning but longer training
  - Default: 50

- **Batch Size** (16-128): Number of samples per training batch
  - Larger batches = Faster training but more memory
  - Default: 32

- **Lookback Period** (30-90 days): Historical window for predictions
  - Longer period = More context but slower training
  - Default: 60 days

- **LSTM Units** (32-128): Neural network complexity
  - More units = More capacity but slower training
  - Default: 64

## 🎯 Model Architecture

The predictor uses a 3-layer LSTM neural network:

```
Layer 1: LSTM (64 units) + Dropout (0.2)
Layer 2: LSTM (64 units) + Dropout (0.2)
Layer 3: LSTM (64 units) + Dropout (0.2)
Output:  Dense (1 unit)
```

- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: Mean Absolute Error (MAE)

## 📦 Deployment to Streamlit Cloud

### Step 1: Prepare Your Repository

1. Create a GitHub repository
2. Upload all files:
   - `app.py`
   - `model.py`
   - `data_fetcher.py`
   - `utils.py`
   - `requirements.txt`
   - `README.md`

### Step 2: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file path: `app.py`
6. Click "Deploy"

### Step 3: Configuration (if needed)

Create a `.streamlit/config.toml` file (optional):

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[server]
maxUploadSize = 200
```

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **ML Framework**: TensorFlow/Keras
- **Data Source**: Yahoo Finance (yfinance)
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy
- **Model Storage**: Joblib

## ⚠️ Important Notes

### Limitations

- **Educational Purpose**: This tool is for learning and research only
- **Not Financial Advice**: Do not use predictions for actual trading decisions
- **Market Volatility**: Stock markets are unpredictable; past performance ≠ future results
- **Data Accuracy**: Relies on Yahoo Finance data availability

### Best Practices

- Use at least 2 years of historical data for training
- Train models regularly with updated data
- Compare multiple models before making decisions
- Always verify data quality before training

## 🐛 Troubleshooting

### Common Issues

**Issue**: "No data found for symbol"
- **Solution**: Verify the stock symbol is correct and actively traded

**Issue**: "Model training failed"
- **Solution**: Ensure you have enough historical data (at least 100 days)

**Issue**: "Memory error during training"
- **Solution**: Reduce batch size or LSTM units in settings

**Issue**: "Slow training"
- **Solution**: Reduce epochs or use smaller lookback period

## 📈 Performance Tips

1. **Faster Training**:
   - Reduce epochs to 30-40
   - Increase batch size to 64
   - Use shorter lookback period (30-45 days)

2. **Better Accuracy**:
   - Increase epochs to 70-100
   - Use longer lookback period (60-90 days)
   - Train with more historical data (3+ years)

3. **Resource Optimization**:
   - Train one stock at a time for large models
   - Clear browser cache if experiencing slowdowns
   - Close other applications during training

## 🤝 Contributing

Feel free to fork this project and submit pull requests! Areas for improvement:

- Add more technical indicators (RSI, MACD, etc.)
- Implement ensemble models
- Add real-time predictions
- Support for cryptocurrency prices
- Multi-variate analysis

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io)
- Data from [Yahoo Finance](https://finance.yahoo.com)
- Powered by [TensorFlow](https://tensorflow.org)

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Review Streamlit logs in the terminal
3. Open an issue on GitHub

---

**Disclaimer**: This application is for educational and research purposes only. Stock price predictions are based on historical data and do not guarantee future performance. Always consult with financial professionals before making investment decisions.

Happy Predicting! 📊✨
