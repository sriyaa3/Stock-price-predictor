# 🚀 Stock Price Predictor

An AI-powered stock price prediction web application using LSTM neural networks, built with FastAPI backend, React frontend, and MongoDB database.

## 🎯 Features

### 🤖 Machine Learning
- **LSTM Neural Network**: 3-layer LSTM with dropout regularization
- **Real Stock Data**: Fetches live data using Yahoo Finance API
- **Configurable Training**: Adjustable epochs, batch size, and sequence length
- **Multi-Day Predictions**: Generate forecasts for 1-30 days ahead
- **Performance Metrics**: MSE, MAE, and loss tracking

### 📊 Data Visualization
- **Interactive Charts**: Historical price visualization using Recharts
- **Prediction Overlays**: Combined historical + predicted price charts
- **Real-time Updates**: Dynamic chart updates with new predictions
- **Responsive Design**: Works on desktop, tablet, and mobile

### 🌐 Web Interface
- **Modern UI**: Clean, professional design with shadcn/ui components
- **Tabbed Navigation**: Data, Visualize, Train Model, and Predict sections
- **Real-time Feedback**: Loading states and success/error notifications
- **CSV Export**: Download predictions for external analysis

### 🔧 Technical Features
- **RESTful API**: FastAPI backend with automatic OpenAPI documentation
- **Database Storage**: MongoDB for persistent data and model storage
- **Model Serialization**: Save and load trained LSTM models
- **Error Handling**: Comprehensive error handling and validation
- **CORS Support**: Cross-origin resource sharing enabled

## 🛠️ Technology Stack

**Backend:**
- FastAPI (Python web framework)
- TensorFlow/Keras (LSTM neural networks)
- yfinance (Stock data fetching)
- MongoDB with Motor (Async database driver)
- scikit-learn (Data preprocessing)
- pandas & numpy (Data manipulation)

**Frontend:**
- React (User interface)
- Recharts (Data visualization)
- shadcn/ui (UI components)
- Tailwind CSS (Styling)
- Axios (API communication)

## 🚀 Local Deployment Guide

### Prerequisites
- Docker and Docker Compose
- Node.js (v16+) and yarn
- Python (v3.9+) and pip
- MongoDB (or use Docker)

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd stock-predictor
```

### 2. Environment Setup

**Backend Environment (.env):**
```bash
# /app/backend/.env
MONGO_URL=mongodb://localhost:27017
DB_NAME=stock_predictor
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

**Frontend Environment (.env):**
```bash
# /app/frontend/.env
REACT_APP_BACKEND_URL=http://localhost:8001
```

### 3. Start MongoDB
```bash
# Using Docker
docker run --name mongodb -p 27017:27017 -d mongo:latest

# Or use local MongoDB installation
mongod --dbpath /path/to/data
```

### 4. Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

### 5. Frontend Setup
```bash
cd frontend

# Install dependencies
yarn install

# Start React development server
yarn start
```

### 6. Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8001
- **API Documentation**: http://localhost:8001/docs

## 🎮 Usage Guide

### 1. Fetch Stock Data
1. Enter a stock symbol (e.g., AAPL, TSLA, GOOGL)
2. Select time period (1y, 2y, 5y, max)
3. Click "Fetch Data" button
4. View data summary in the Data tab

### 2. Visualize Historical Data
1. Navigate to the "Visualize" tab
2. View interactive price chart
3. Analyze price trends and patterns

### 3. Train LSTM Model
1. Go to "Train Model" tab
2. Configure training parameters:
   - Epochs: 10-200 (default: 50)
   - Batch size: 16-64 (default: 32)
   - Sequence length: 30-120 (default: 60)
3. Click "Train LSTM Model"
4. Monitor training progress and metrics

### 4. Generate Predictions
1. Navigate to "Predict" tab
2. Set prediction days (1-30)
3. Click "Generate Predictions"
4. View prediction table and chart
5. Download results as CSV

## 📡 API Endpoints

### Stock Data
- `POST /api/fetch-stock` - Fetch stock data from Yahoo Finance
- `GET /api/stock-data/{symbol}` - Get stored stock data

### Model Management
- `POST /api/train-model` - Train LSTM model
- `GET /api/models` - List trained models

### Predictions
- `POST /api/predict` - Generate price predictions
- `GET /api/predictions/{symbol}` - Get stored predictions
- `GET /api/export-predictions/{symbol}` - Export predictions as CSV

## 🧪 Testing

### Backend API Testing
```bash
# Run comprehensive API tests
python backend_test.py

# Test specific endpoint
curl -X POST "http://localhost:8001/api/fetch-stock" \
     -H "Content-Type: application/json" \
     -d '{"symbol": "AAPL", "period": "2y"}'
```

### Frontend Testing
```bash
cd frontend
yarn test
```

## 🐳 Docker Deployment

### Docker Compose
```yaml
version: '3.8'
services:
  mongodb:
    image: mongo:latest
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db

  backend:
    build: ./backend
    ports:
      - "8001:8001"
    environment:
      - MONGO_URL=mongodb://mongodb:27017
      - DB_NAME=stock_predictor
    depends_on:
      - mongodb

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_BACKEND_URL=http://localhost:8001
    depends_on:
      - backend

volumes:
  mongodb_data:
```

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🔧 Configuration

### Model Parameters
- **Sequence Length**: Days of historical data for prediction (default: 60)
- **Epochs**: Training iterations (default: 50)
- **Batch Size**: Training batch size (default: 32)
- **LSTM Layers**: 3 layers with 50 units each
- **Dropout Rate**: 0.2 for regularization

### Supported Stock Symbols
- Major US stocks (AAPL, GOOGL, MSFT, TSLA, AMZN, etc.)
- International symbols (check Yahoo Finance availability)
- Crypto symbols (BTC-USD, ETH-USD, etc.)

## 🐛 Troubleshooting

### Common Issues

**MongoDB Connection Error:**
```bash
# Check MongoDB status
mongod --version
ps aux | grep mongod

# Restart MongoDB
sudo systemctl restart mongod
```

**Missing Dependencies:**
```bash
# Backend
pip install -r requirements.txt

# Frontend
yarn install
```

**CORS Issues:**
- Ensure frontend URL is in CORS_ORIGINS
- Check .env file configuration
- Restart backend server

**Model Training Slow:**
- Reduce epochs for testing
- Use smaller sequence length
- Check system resources

## 📈 Performance Tips

### Backend Optimization
- Use GPU for TensorFlow operations
- Implement model caching
- Add database indexing
- Use Redis for session storage

### Frontend Optimization
- Implement chart virtualization
- Add data pagination
- Use React.memo for components
- Optimize bundle size

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Yahoo Finance for stock data API
- TensorFlow team for ML framework
- React community for frontend tools
- shadcn for UI components

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the API documentation at `/docs`
- Review the troubleshooting section

---

**Built with ❤️ for financial analysis and machine learning enthusiasts**
