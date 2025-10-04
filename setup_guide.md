# 🚀 Setup Guide - Stock Price Predictor

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Internet connection (for fetching stock data)

## Step-by-Step Installation

### 1. Download/Clone the Project

Save all the project files in a folder:
```
stock-price-predictor/
├── app.py
├── model.py
├── data_fetcher.py
├── utils.py
├── requirements.txt
├── README.md
├── test_app.py
└── .gitignore
```

### 2. Create a Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- Streamlit (web interface)
- TensorFlow (LSTM neural networks)
- yfinance (stock data)
- Pandas, NumPy (data processing)
- Plotly (visualizations)
- Scikit-learn (data preprocessing)
- Joblib (model saving)

**Installation time**: 2-5 minutes depending on your internet speed

### 4. Verify Installation

Run the test script:
```bash
python test_app.py
```

Expected output:
```
✅ Streamlit imported successfully
✅ TensorFlow imported successfully
✅ yfinance imported successfully
...
🎉 ALL TESTS PASSED!
```

### 5. Run the Application

```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## First Time Usage

### Quick Start Tutorial

1. **Launch the app**
   ```bash
   streamlit run app.py
   ```

2. **Enter stock symbols** in the sidebar
   - Example: `AAPL, GOOGL, MSFT`
   - Use comma-separated valid ticker symbols

3. **Click "📊 Fetch Data"**
   - Downloads 2 years of historical data
   - Wait 10-30 seconds for data to load

4. **Click "🚀 Train Models"**
   - Trains LSTM models for each stock
   - Takes 1-3 minutes per stock
   - You'll see training progress and loss curves

5. **Click "🔮 Generate Predictions"**
   - Creates 7-day forecasts
   - View predictions with interactive charts

6. **Explore other tabs**
   - Compare multiple stocks
   - Export predictions to CSV

## Troubleshooting

### Issue: "Module not found"
**Solution:** Make sure you activated the virtual environment and ran `pip install -r requirements.txt`

### Issue: "TensorFlow installation failed"
**Solution for Mac (M1/M2):**
```bash
pip install tensorflow-macos
pip install tensorflow-metal
```

**Solution for Windows/Linux:**
```bash
pip install tensorflow==2.15.0
```

### Issue: "Port 8501 already in use"
**Solution:** Kill the existing process or use a different port:
```bash
streamlit run app.py --server.port 8502
```

### Issue: "Cannot fetch stock data"
**Solution:** 
- Check your internet connection
- Verify the stock symbol is correct
- Try a different stock (e.g., AAPL)

### Issue: "Out of memory during training"
**Solution:** Reduce model complexity in the sidebar:
- Decrease LSTM units to 32
- Reduce batch size to 16
- Shorten lookback period to 30 days

## System Requirements

### Minimum
- **CPU**: Dual-core processor
- **RAM**: 4 GB
- **Storage**: 500 MB free space
- **OS**: Windows 10, macOS 10.14+, or Linux

### Recommended
- **CPU**: Quad-core processor
- **RAM**: 8 GB or more
- **Storage**: 2 GB free space
- **GPU**: Optional (speeds up training)

## Performance Expectations

### Training Time (per stock)
- **Fast settings** (30 epochs, 30 lookback): 30-60 seconds
- **Default settings** (50 epochs, 60 lookback): 1-2 minutes
- **High accuracy** (100 epochs, 90 lookback): 3-5 minutes

### Data Fetching Time
- **Single stock**: 5-10 seconds
- **Multiple stocks** (3-5): 15-30 seconds

### Prediction Time
- Nearly instant (< 1 second per stock)

## Deployment to Streamlit Cloud

### Prerequisites
1. GitHub account
2. Streamlit Cloud account (free at share.streamlit.io)

### Steps

1. **Create GitHub Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Click "New app"
   - Select your repository
   - Set main file: `app.py`
   - Click "Deploy"

3. **Wait for deployment** (2-5 minutes)

4. **Your app is live!** 🎉
   - You'll get a URL like: `https://your-app.streamlit.app`

### Deployment Notes
- Free tier includes: 1 GB RAM, 1 CPU core
- Apps sleep after inactivity (wake up on access)
- Perfect for demos and personal use

## Development Tips

### Modifying the Code

1. **Change default stocks**: Edit `app.py` line ~50
   ```python
   value="AAPL, GOOGL, MSFT"  # Change these
   ```

2. **Adjust model architecture**: Edit `model.py` lines ~40-60
   ```python
   LSTM(units=self.lstm_units, ...)  # Modify layers
   ```

3. **Change prediction days**: Edit `app.py` line ~300
   ```python
   predictions = model.predict_future(df['Close'].values, days=7)  # Change 7
   ```

### Running in Development Mode

```bash
streamlit run app.py --server.runOnSave true
```
This auto-reloads the app when you save changes.

### Debugging

View detailed logs:
```bash
streamlit run app.py --logger.level=debug
```

## Updating the App

### Update Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Pull Latest Changes
```bash
git pull origin main
pip install -r requirements.txt
```

## Uninstalling

1. **Deactivate virtual environment**
   ```bash
   deactivate
   ```

2. **Delete project folder**
   - Windows: Delete the folder
   - Mac/Linux: `rm -rf stock-price-predictor`

## Getting Help

### Documentation Links
- [Streamlit Docs](https://docs.streamlit.io)
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [yfinance Docs](https://pypi.org/project/yfinance/)

### Common Commands Reference

```bash
# Start app
streamlit run app.py

# Run tests
python test_app.py

# Install packages
pip install -r requirements.txt

# Update packages
pip install --upgrade -r requirements.txt

# Check Python version
python --version

# Check installed packages
pip list

# Activate environment
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Deactivate environment
deactivate
```

---

**Need more help?** Check the README.md file or open an issue on GitHub.

Happy coding! 🚀📈
