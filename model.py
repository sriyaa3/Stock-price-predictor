import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class LSTMStockPredictor:
    """
    LSTM-based stock price prediction model
    """
    
    def __init__(self, lookback_period=60, lstm_units=64):
        """
        Initialize the LSTM model
        
        Args:
            lookback_period (int): Number of days to look back for prediction
            lstm_units (int): Number of LSTM units in each layer
        """
        self.lookback_period = lookback_period
        self.lstm_units = lstm_units
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        
    def _create_model(self):
        """Create the LSTM neural network architecture"""
        model = Sequential([
            # First LSTM layer with dropout
            LSTM(units=self.lstm_units, return_sequences=True, 
                 input_shape=(self.lookback_period, 1)),
            Dropout(0.2),
            
            # Second LSTM layer with dropout
            LSTM(units=self.lstm_units, return_sequences=True),
            Dropout(0.2),
            
            # Third LSTM layer with dropout
            LSTM(units=self.lstm_units, return_sequences=False),
            Dropout(0.2),
            
            # Output layer
            Dense(units=1)
        ])
        
        # Compile model
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        
        return model
    
    def _prepare_data(self, data):
        """
        Prepare data for training
        
        Args:
            data (np.array): Stock price data
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        # Reshape data
        data = data.reshape(-1, 1)
        
        # Scale data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(self.lookback_period, len(scaled_data)):
            X.append(scaled_data[i-self.lookback_period:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Reshape for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Split into train and test (80-20 split)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, y_train, X_test, y_test
    
    def train(self, data, epochs=50, batch_size=32, verbose=0):
        """
        Train the LSTM model
        
        Args:
            data (np.array): Stock price data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            verbose (int): Verbosity level
            
        Returns:
            dict: Training history
        """
        # Prepare data
        X_train, y_train, X_test, y_test = self._prepare_data(data)
        
        # Create model
        self.model = self._create_model()
        
        # Early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=verbose
        )
        
        # Store history
        self.history = {
            'train_loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'train_mae': history.history['mae'],
            'val_mae': history.history['val_mae']
        }
        
        return self.history
    
    def predict_future(self, data, days=7):
        """
        Predict future stock prices
        
        Args:
            data (np.array): Historical stock price data
            days (int): Number of days to predict
            
        Returns:
            np.array: Predicted prices
        """
        if self.model is None:
            raise Exception("Model not trained yet!")
        
        # Prepare the last sequence
        data = data.reshape(-1, 1)
        scaled_data = self.scaler.transform(data)
        
        # Get the last lookback_period days
        last_sequence = scaled_data[-self.lookback_period:]
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # Reshape for prediction
            current_batch = current_sequence.reshape(1, self.lookback_period, 1)
            
            # Predict next value
            next_pred = self.model.predict(current_batch, verbose=0)[0, 0]
            predictions.append(next_pred)
            
            # Update sequence: remove first value, append prediction
            current_sequence = np.append(current_sequence[1:], [[next_pred]], axis=0)
        
        # Inverse transform to get actual prices
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()
    
    def evaluate(self, data):
        """
        Evaluate model performance
        
        Args:
            data (np.array): Stock price data
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise Exception("Model not trained yet!")
        
        X_train, y_train, X_test, y_test = self._prepare_data(data)
        
        # Predictions
        train_pred = self.model.predict(X_train, verbose=0)
        test_pred = self.model.predict(X_test, verbose=0)
        
        # Calculate metrics
        train_rmse = np.sqrt(np.mean((train_pred - y_train)**2))
        test_rmse = np.sqrt(np.mean((test_pred - y_test)**2))
        
        train_mae = np.mean(np.abs(train_pred - y_train))
        test_mae = np.mean(np.abs(test_pred - y_test))
        
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae
        }
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model is None:
            return "Model not created yet"
        return self.model.summary()