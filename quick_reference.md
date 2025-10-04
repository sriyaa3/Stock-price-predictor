# 📋 Quick Reference - Stock Price Predictor

## Essential Commands

```bash
# Install & Setup
pip install -r requirements.txt
python test_app.py

# Run Application
streamlit run app.py

# Stop Application
Ctrl + C
```

## File Structure

```
📁 stock-price-predictor/
├── 📄 app.py              # Main Streamlit app
├── 📄 model.py            # LSTM neural network
├── 📄 data_fetcher.py     # Yahoo Finance integration
├── 📄 utils.py            # Helper functions
├── 📄 requirements.txt    # Dependencies
├── 📄 test_app.py         # Test script
├── 📄 README.md           # Full documentation
├── 📄 SETUP_GUIDE.md      # Installation guide
├── 📄 .gitignore          # Git ignore rules
├── 📁 .streamlit/         # Streamlit config
│   └── config.toml
├── 📁 models/             # Saved models (auto-created)
└── 📁 exports/            # CSV exports (auto-created)
```

## Usage Workflow

### Step 1: Fetch Data 📊
- Enter stock symbols: `AAPL, GOOGL, MSFT`
- Click **"📊 Fetch Data"**
- Wait ~20 seconds

### Step 2: Train Models 🚀
- Adjust hyperparameters (optional)
- Click **"🚀 Train Models"**
- Wait 1-3 minutes per stock

### Step 3: Generate Predictions 🔮
- Click **"🔮 Generate Predictions"**
- View 7-day forecasts
- Instant results!

### Step 4: Export Data 💾
- Go to **"Export"** tab
- Select stocks
- Download CSV

## Default Settings

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| **Epochs** | 50 | 10-100 | Training iterations |
| **Batch Size** | 32 | 16-128 | Training batch size |
| **Lookback** | 60 | 30-90 | Days of history used |
| **LSTM Units** | 64 | 32-128 | Network complexity |
| **Prediction Days** | 7 | Fixed | Forecast period |
| **Historical Data** | 2 years | Fixed | Training data range |

## Popular Stocks to Try

### Tech Giants
- AAPL (Apple)
- MSFT (Microsoft)
- GOOGL (Google)
- META (Meta/Facebook)
- AMZN (Amazon)

### Finance
- JPM (JPMorgan)
- BAC (Bank of America)
- GS (Goldman Sachs)
- V (Visa)

### Other
- TSLA (Tesla)
- NVDA (NVIDIA)
- DIS (Disney)
- NKE (Nike)
- NFLX (Netflix)

## Performance Tips

### ⚡ Faster Training
```
Epochs: 30
Batch Size: 64
Lookback: 30
LSTM Units: 32
Time: ~30 seconds/stock
```

### 🎯 Better Accuracy
```
Epochs: 80
Batch Size: 32
Lookback: 90
LSTM Units: 128
Time: ~4 minutes/stock
```

### ⚖️ Balanced (Recommended)
```
Epochs: 50
Batch Size: 32
Lookback: 60
LSTM Units: 64
Time: ~2 minutes/stock
```

## Keyboard Shortcuts (Streamlit)

| Shortcut | Action |
|----------|--------|
| `Ctrl + R` | Rerun app |
| `Ctrl + C` | Stop app (in terminal) |
| `C` | Clear cache |
| `?` | Show keyboard shortcuts |

## Common Stock Symbols

| Symbol | Company | Sector |
|--------|---------|--------|
| AAPL | Apple | Technology |
| MSFT | Microsoft | Technology |
| GOOGL | Alphabet (Google) | Technology |
| AMZN | Amazon | Retail/Tech |
| TSLA | Tesla | Automotive |
| NVDA | NVIDIA | Semiconductors |
| JPM | JPMorgan | Finance |
| JNJ | Johnson & Johnson | Healthcare |
| WMT | Walmart | Retail |
| XOM | Exxon Mobil | Energy |

## Troubleshooting Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| **Import errors** | `pip install -r requirements.txt` |
| **Port in use** | `streamlit run app.py --server.port 8502` |
| **Out of memory** | Reduce LSTM units to 32 |
| **Slow training** | Reduce epochs to 30 |
| **No data found** | Check stock symbol is valid |
| **App won't start** | Check Python version (need 3.9+) |

## API Endpoints (Yahoo Finance)

The app uses these yfinance functions:

```python
# Fetch historical data
yf.Ticker(symbol).history(start=date, end=date)

# Get stock info
yf.Ticker(symbol).info

# Check validity
yf.Ticker(symbol).history(period='5d')
```

**No API key required!** ✅

## Model Architecture Quick View

```
Input: [batch_size, 60, 1]
   ↓
LSTM Layer 1 (64 units) → Dropout (0.2)
   ↓
LSTM Layer 2 (64 units) → Dropout (0.2)
   ↓
LSTM Layer 3 (64 units) → Dropout (0.2)
   ↓
Dense Layer (1 unit)
   ↓
Output: Next day price prediction
```

## Data Flow

```
Yahoo Finance → Raw Data → Scaling (0-1) → 
Sequences (60 days) → LSTM Model → 
Predictions → Inverse Scaling → Final Prices
```

## Metrics Explained

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Training Loss** | Error on training data | < 0.01 |
| **Validation Loss** | Error on test data | < 0.02 |
| **MAE** | Mean Absolute Error | < 5.0 |
| **RMSE** | Root Mean Square Error | < 10.0 |
| **MAPE** | Mean Absolute % Error | < 5% |

## Deployment Checklist

- [ ] All files present
- [ ] requirements.txt complete
- [ ] GitHub repo created
- [ ] .gitignore configured
- [ ] README.md updated
- [ ] Pushed to GitHub
- [ ] Streamlit Cloud account ready
- [ ] App deployed
- [ ] Test deployment
- [ ] Share URL!

## Environment Variables (Optional)

Create `.env` file for custom settings:
```bash
DEFAULT_STOCKS=AAPL,GOOGL,MSFT
DEFAULT_EPOCHS=50
DEFAULT_BATCH_SIZE=32
```

## Useful Links

- **Streamlit Docs**: https://docs.streamlit.io
- **TensorFlow Guide**: https://tensorflow.org/guide
- **Yahoo Finance**: https://finance.yahoo.com
- **Stock Symbols**: https://finance.yahoo.com/lookup

## File Sizes

| Item | Typical Size |
|------|-------------|
| Trained Model | 2-5 MB |
| Historical Data (2 years) | 10-50 KB |
| CSV Export | 1-5 KB |
| Full Installation | ~500 MB |

## Training Time Estimates

**Single Stock:**
- CPU: 1-3 minutes
- GPU: 20-40 seconds

**Multiple Stocks (3):**
- CPU: 3-9 minutes
- GPU: 1-2 minutes

## Memory Usage

| Configuration | RAM Usage |
|--------------|-----------|
| **Minimal** (1 stock, 32 units) | ~500 MB |
| **Normal** (3 stocks, 64 units) | ~1-2 GB |
| **Heavy** (5+ stocks, 128 units) | ~3-4 GB |

## Best Practices

✅ **DO:**
- Use 2+ years of data
- Train regularly (weekly)
- Compare multiple models
- Verify data quality
- Save trained models
- Export predictions for records

❌ **DON'T:**
- Use predictions for real trading
- Train with < 100 days data
- Ignore validation loss
- Overtrain (> 100 epochs)
- Use invalid stock symbols
- Rely on single prediction

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025 | Initial release |

## Support Commands

```bash
# Check versions
python --version
pip list

# Clear Streamlit cache
streamlit cache clear

# View logs
streamlit run app.py --logger.level=debug

# Update all packages
pip install --upgrade -r requirements.txt

# Create requirements from current env
pip freeze > requirements.txt
```

## Export Formats

### CSV Structure
```csv
Symbol,Date,Predicted_Price
AAPL,2025-10-05,175.23
AAPL,2025-10-06,176.45
...
```

### Model Files
- Format: `.pkl` (pickle/joblib)
- Location: `models/SYMBOL_model.pkl`
- Size: 2-5 MB each

## Quick Debugging

```python
# Test data fetching
from data_fetcher import StockDataFetcher
fetcher = StockDataFetcher()
df = fetcher.fetch_stock_data('AAPL')
print(df.head())

# Test model
from model import LSTMStockPredictor
import numpy as np
model = LSTMStockPredictor()
data = np.random.randn(200) * 10 + 100
model.train(data, epochs=5)
predictions = model.predict_future(data, days=7)
print(predictions)
```

## Contact & Contribution

- **Issues**: GitHub Issues
- **PRs**: Always welcome!
- **Questions**: Check README.md first
- **Updates**: Watch the repo

---

**Last Updated**: October 2025  
**Version**: 1.0  
**License**: Open Source (Educational)

🚀 Happy Predicting! 📈