from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle
import io
import json
import base64

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(title="Stock Price Predictor API", version="1.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Pydantic Models
class StockData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    data: List[Dict[str, Any]]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StockRequest(BaseModel):
    symbol: str
    period: str = "2y"  # Default 2 years

class ModelTrainRequest(BaseModel):
    symbol: str
    epochs: int = 50
    batch_size: int = 32
    sequence_length: int = 60
    
class PredictionRequest(BaseModel):
    symbol: str
    days: int = 7

class ModelInfo(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    model_data: str  # Base64 encoded model
    scaler_data: str  # Base64 encoded scaler
    metrics: Dict[str, float]
    sequence_length: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class PredictionResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    predictions: List[Dict[str, Any]]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# Helper Functions
def prepare_lstm_data(data, sequence_length=60):
    """Prepare data for LSTM training"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler

def create_lstm_model(sequence_length):
    """Create LSTM model architecture"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def serialize_model(model):
    """Serialize Keras model to base64 string"""
    model_json = model.to_json()
    model_weights = model.get_weights()
    
    # Serialize weights
    weights_bytes = pickle.dumps(model_weights)
    
    model_data = {
        'json': model_json,
        'weights': base64.b64encode(weights_bytes).decode('utf-8')
    }
    
    return base64.b64encode(json.dumps(model_data).encode()).decode('utf-8')

def deserialize_model(model_data_str):
    """Deserialize base64 string to Keras model"""
    model_data = json.loads(base64.b64decode(model_data_str).decode())
    
    # Recreate model from JSON
    model = tf.keras.models.model_from_json(model_data['json'])
    
    # Load weights
    weights_bytes = base64.b64decode(model_data['weights'])
    weights = pickle.loads(weights_bytes)
    model.set_weights(weights)
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def serialize_scaler(scaler):
    """Serialize scaler to base64 string"""
    scaler_bytes = pickle.dumps(scaler)
    return base64.b64encode(scaler_bytes).decode('utf-8')

def deserialize_scaler(scaler_data_str):
    """Deserialize base64 string to scaler"""
    scaler_bytes = base64.b64decode(scaler_data_str)
    return pickle.loads(scaler_bytes)


# API Endpoints
@api_router.get("/")
async def root():
    return {"message": "Stock Price Predictor API is running!"}

@api_router.post("/fetch-stock")
async def fetch_stock_data(request: StockRequest):
    """Fetch stock data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(request.symbol.upper())
        hist = ticker.history(period=request.period)
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {request.symbol}")
        
        # Convert to list of dictionaries
        data_list = []
        for date, row in hist.iterrows():
            data_list.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": int(row['Volume'])
            })
        
        # Store in MongoDB
        stock_data = StockData(symbol=request.symbol.upper(), data=data_list)
        await db.stock_data.insert_one(stock_data.dict())
        
        return {
            "symbol": request.symbol.upper(),
            "data": data_list,
            "message": f"Fetched {len(data_list)} data points for {request.symbol.upper()}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stock data: {str(e)}")

@api_router.get("/stock-data/{symbol}")
async def get_stock_data(symbol: str):
    """Get stored stock data for a symbol"""
    try:
        stock_data = await db.stock_data.find_one(
            {"symbol": symbol.upper()}, 
            {"_id": 0},  # Exclude MongoDB ObjectId
            sort=[("created_at", -1)]
        )
        
        if not stock_data:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
        
        return stock_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving stock data: {str(e)}")

@api_router.post("/train-model")
async def train_model(request: ModelTrainRequest, background_tasks: BackgroundTasks):
    """Train LSTM model for stock prediction"""
    try:
        # Get stock data
        stock_data = await db.stock_data.find_one(
            {"symbol": request.symbol.upper()}, 
            sort=[("created_at", -1)]
        )
        
        if not stock_data:
            raise HTTPException(
                status_code=404, 
                detail=f"No stock data found for {request.symbol}. Please fetch data first."
            )
        
        # Prepare data
        df = pd.DataFrame(stock_data['data'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        close_prices = df['close'].values
        
        if len(close_prices) < request.sequence_length + 20:
            raise HTTPException(
                status_code=400, 
                detail=f"Not enough data points. Need at least {request.sequence_length + 20} points."
            )
        
        # Prepare LSTM data
        X, y, scaler = prepare_lstm_data(close_prices, request.sequence_length)
        
        # Split data
        split_ratio = 0.8
        split_index = int(len(X) * split_ratio)
        
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        # Reshape for LSTM
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Create and train model
        model = create_lstm_model(request.sequence_length)
        
        history = model.fit(
            X_train, y_train,
            batch_size=request.batch_size,
            epochs=request.epochs,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # Calculate metrics
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Inverse transform for metrics calculation
        train_pred_original = scaler.inverse_transform(train_pred)
        test_pred_original = scaler.inverse_transform(test_pred)
        y_train_original = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        metrics = {
            "train_mse": float(mean_squared_error(y_train_original, train_pred_original)),
            "test_mse": float(mean_squared_error(y_test_original, test_pred_original)),
            "train_mae": float(mean_absolute_error(y_train_original, train_pred_original)),
            "test_mae": float(mean_absolute_error(y_test_original, test_pred_original)),
            "final_loss": float(history.history['loss'][-1]),
            "final_val_loss": float(history.history['val_loss'][-1])
        }
        
        # Serialize and store model
        model_data_str = serialize_model(model)
        scaler_data_str = serialize_scaler(scaler)
        
        model_info = ModelInfo(
            symbol=request.symbol.upper(),
            model_data=model_data_str,
            scaler_data=scaler_data_str,
            metrics=metrics,
            sequence_length=request.sequence_length
        )
        
        # Remove old models for this symbol and store new one
        await db.models.delete_many({"symbol": request.symbol.upper()})
        await db.models.insert_one(model_info.dict())
        
        return {
            "symbol": request.symbol.upper(),
            "metrics": metrics,
            "message": "Model trained successfully!"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@api_router.post("/predict")
async def predict_prices(request: PredictionRequest):
    """Predict future stock prices"""
    try:
        # Get trained model
        model_info = await db.models.find_one(
            {"symbol": request.symbol.upper()}, 
            sort=[("created_at", -1)]
        )
        
        if not model_info:
            raise HTTPException(
                status_code=404, 
                detail=f"No trained model found for {request.symbol}. Please train a model first."
            )
        
        # Get stock data
        stock_data = await db.stock_data.find_one(
            {"symbol": request.symbol.upper()}, 
            sort=[("created_at", -1)]
        )
        
        if not stock_data:
            raise HTTPException(
                status_code=404, 
                detail=f"No stock data found for {request.symbol}"
            )
        
        # Deserialize model and scaler
        model = deserialize_model(model_info['model_data'])
        scaler = deserialize_scaler(model_info['scaler_data'])
        sequence_length = model_info['sequence_length']
        
        # Prepare recent data for prediction
        df = pd.DataFrame(stock_data['data'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        close_prices = df['close'].values
        recent_data = close_prices[-sequence_length:]
        
        # Scale the recent data
        scaled_recent = scaler.transform(recent_data.reshape(-1, 1))
        
        # Generate predictions
        predictions = []
        current_sequence = scaled_recent.copy()
        
        last_date = df['date'].iloc[-1]
        
        for i in range(request.days):
            # Reshape for prediction
            X_pred = current_sequence[-sequence_length:].reshape((1, sequence_length, 1))
            
            # Predict next value
            next_pred = model.predict(X_pred, verbose=0)
            
            # Inverse transform to get actual price
            next_price = scaler.inverse_transform(next_pred)[0][0]
            
            # Calculate prediction date (skip weekends)
            pred_date = last_date + pd.Timedelta(days=i+1)
            while pred_date.weekday() >= 5:  # Skip Saturday (5) and Sunday (6)
                pred_date += pd.Timedelta(days=1)
            
            predictions.append({
                "date": pred_date.strftime("%Y-%m-%d"),
                "predicted_price": float(next_price),
                "day": i + 1
            })
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence, next_pred)
        
        # Store predictions
        prediction_result = PredictionResult(
            symbol=request.symbol.upper(),
            predictions=predictions
        )
        await db.predictions.insert_one(prediction_result.dict())
        
        return {
            "symbol": request.symbol.upper(),
            "predictions": predictions,
            "model_metrics": model_info['metrics'],
            "message": f"Generated {request.days} day predictions for {request.symbol.upper()}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating predictions: {str(e)}")

@api_router.get("/models")
async def get_trained_models():
    """Get list of all trained models"""
    try:
        models = await db.models.find({}, {"model_data": 0, "scaler_data": 0, "_id": 0}).to_list(100)
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving models: {str(e)}")

@api_router.get("/predictions/{symbol}")
async def get_predictions(symbol: str):
    """Get latest predictions for a symbol"""
    try:
        predictions = await db.predictions.find_one(
            {"symbol": symbol.upper()}, 
            {"_id": 0},  # Exclude MongoDB ObjectId
            sort=[("created_at", -1)]
        )
        
        if not predictions:
            raise HTTPException(status_code=404, detail=f"No predictions found for symbol {symbol}")
        
        return predictions
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving predictions: {str(e)}")

@api_router.get("/export-predictions/{symbol}")
async def export_predictions_csv(symbol: str):
    """Export predictions as CSV file"""
    try:
        predictions = await db.predictions.find_one(
            {"symbol": symbol.upper()}, 
            {"_id": 0},  # Exclude MongoDB ObjectId
            sort=[("created_at", -1)]
        )
        
        if not predictions:
            raise HTTPException(status_code=404, detail=f"No predictions found for symbol {symbol}")
        
        # Create CSV content
        df = pd.DataFrame(predictions['predictions'])
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        # Return as streaming response
        return StreamingResponse(
            io.StringIO(csv_content),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={symbol}_predictions.csv"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting predictions: {str(e)}")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()