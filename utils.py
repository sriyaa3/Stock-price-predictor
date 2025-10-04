import os
import joblib
import pandas as pd
from datetime import datetime

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('exports', exist_ok=True)

def save_model(model, symbol):
    """
    Save trained model to disk
    
    Args:
        model: Trained LSTMStockPredictor instance
        symbol (str): Stock symbol
    """
    try:
        filename = f"models/{symbol}_model.pkl"
        joblib.dump(model, filename)
        return True
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return False

def load_model(symbol):
    """
    Load trained model from disk
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        LSTMStockPredictor: Loaded model or None if not found
    """
    try:
        filename = f"models/{symbol}_model.pkl"
        if os.path.exists(filename):
            return joblib.load(filename)
        return None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def export_to_csv(predictions_dict, filename=None):
    """
    Export predictions to CSV file
    
    Args:
        predictions_dict (dict): Dictionary with symbol: predictions mapping
        filename (str): Output filename (optional)
        
    Returns:
        str: Path to saved file
    """
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"exports/predictions_{timestamp}.csv"
    
    all_data = []
    
    for symbol, predictions in predictions_dict.items():
        for day, price in enumerate(predictions, 1):
            all_data.append({
                'Symbol': symbol,
                'Day': day,
                'Predicted_Price': price
            })
    
    df = pd.DataFrame(all_data)
    df.to_csv(filename, index=False)
    
    return filename

def calculate_metrics(actual, predicted):
    """
    Calculate performance metrics
    
    Args:
        actual (array): Actual values
        predicted (array): Predicted values
        
    Returns:
        dict: Performance metrics
    """
    import numpy as np
    
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

def format_price(price):
    """
    Format price with currency symbol
    
    Args:
        price (float): Price value
        
    Returns:
        str: Formatted price string
    """
    return f"${price:.2f}"

def calculate_percentage_change(old_value, new_value):
    """
    Calculate percentage change between two values
    
    Args:
        old_value (float): Original value
        new_value (float): New value
        
    Returns:
        float: Percentage change
    """
    if old_value == 0:
        return 0
    return ((new_value - old_value) / old_value) * 100

def get_available_models():
    """
    Get list of available saved models
    
    Returns:
        list: List of stock symbols with saved models
    """
    models_dir = 'models'
    if not os.path.exists(models_dir):
        return []
    
    models = [f.replace('_model.pkl', '') for f in os.listdir(models_dir) 
              if f.endswith('_model.pkl')]
    return models