"""
Simple tests to verify the application components
Run with: python test_app.py
"""

import sys
import numpy as np

def test_imports():
    """Test if all required packages are installed"""
    print("Testing imports...")
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
        
        import tensorflow
        print("✅ TensorFlow imported successfully")
        
        import yfinance
        print("✅ yfinance imported successfully")
        
        import pandas
        print("✅ Pandas imported successfully")
        
        import plotly
        print("✅ Plotly imported successfully")
        
        import sklearn
        print("✅ Scikit-learn imported successfully")
        
        import joblib
        print("✅ Joblib imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_fetcher():
    """Test the data fetcher module"""
    print("\nTesting data fetcher...")
    try:
        from data_fetcher import StockDataFetcher
        
        fetcher = StockDataFetcher()
        print("✅ StockDataFetcher initialized")
        
        # Test with a known stock
        df = fetcher.fetch_stock_data('AAPL', start_date='2024-01-01', end_date='2024-01-31')
        print(f"✅ Fetched {len(df)} days of AAPL data")
        
        # Test validation
        is_valid = fetcher.validate_symbol('AAPL')
        print(f"✅ Symbol validation: {is_valid}")
        
        return True
    except Exception as e:
        print(f"❌ Data fetcher error: {e}")
        return False

def test_model():
    """Test the LSTM model"""
    print("\nTesting LSTM model...")
    try:
        from model import LSTMStockPredictor
        
        # Create model
        model = LSTMStockPredictor(lookback_period=30, lstm_units=32)
        print("✅ Model initialized")
        
        # Create dummy data
        dummy_data = np.random.randn(200) * 10 + 100  # 200 days of mock stock prices
        
        # Train with minimal settings
        print("Training model (this may take a minute)...")
        history = model.train(dummy_data, epochs=5, batch_size=16, verbose=0)
        print("✅ Model trained successfully")
        
        # Test prediction
        predictions = model.predict_future(dummy_data, days=7)
        print(f"✅ Generated {len(predictions)} predictions")
        
        return True
    except Exception as e:
        print(f"❌ Model error: {e}")
        return False

def test_utils():
    """Test utility functions"""
    print("\nTesting utilities...")
    try:
        from utils import save_model, load_model, calculate_percentage_change
        from model import LSTMStockPredictor
        
        # Test model save/load
        model = LSTMStockPredictor(lookback_period=30, lstm_units=32)
        dummy_data = np.random.randn(100) * 10 + 100
        model.train(dummy_data, epochs=2, batch_size=16, verbose=0)
        
        save_model(model, 'TEST')
        print("✅ Model saved")
        
        loaded_model = load_model('TEST')
        if loaded_model:
            print("✅ Model loaded")
        
        # Test percentage change
        change = calculate_percentage_change(100, 110)
        assert abs(change - 10.0) < 0.01, "Percentage calculation error"
        print("✅ Percentage calculation correct")
        
        return True
    except Exception as e:
        print(f"❌ Utils error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("STOCK PRICE PREDICTOR - SYSTEM TESTS")
    print("=" * 50)
    
    results = {
        "Imports": test_imports(),
        "Data Fetcher": test_data_fetcher(),
        "LSTM Model": test_model(),
        "Utilities": test_utils()
    }
    
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Ready to run the app.")
        print("\nRun the app with: streamlit run app.py")
    else:
        print("⚠️  SOME TESTS FAILED. Please check the errors above.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    print("=" * 50)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
