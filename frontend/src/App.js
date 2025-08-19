import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Label } from './components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Badge } from './components/ui/badge';
import { Loader2, TrendingUp, Download, BarChart3, Brain, Target } from 'lucide-react';
import { useToast } from './hooks/use-toast';
import { Toaster } from './components/ui/toaster';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [stockSymbol, setStockSymbol] = useState('AAPL');
  const [period, setPeriod] = useState('2y');
  const [stockData, setStockData] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [trainedModels, setTrainedModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [epochs, setEpochs] = useState(50);
  const [predictionDays, setPredictionDays] = useState(7);
  const [activeTab, setActiveTab] = useState('data');
  const { toast } = useToast();

  // Fetch stock data
  const fetchStockData = async () => {
    if (!stockSymbol.trim()) {
      toast({
        title: "Error",
        description: "Please enter a stock symbol",
        variant: "destructive",
      });
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API}/fetch-stock`, {
        symbol: stockSymbol.toUpperCase(),
        period: period
      });
      
      setStockData(response.data);
      toast({
        title: "Success",
        description: `Fetched ${response.data.data.length} data points for ${stockSymbol.toUpperCase()}`,
      });
      setActiveTab('visualize');
    } catch (error) {
      toast({
        title: "Error",
        description: error.response?.data?.detail || "Failed to fetch stock data",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  // Train LSTM model
  const trainModel = async () => {
    if (!stockData) {
      toast({
        title: "Error",
        description: "Please fetch stock data first",
        variant: "destructive",
      });
      return;
    }

    setTraining(true);
    try {
      const response = await axios.post(`${API}/train-model`, {
        symbol: stockSymbol.toUpperCase(),
        epochs: epochs,
        batch_size: 32,
        sequence_length: 60
      });

      toast({
        title: "Success",
        description: "Model trained successfully!",
      });
      
      // Refresh trained models list
      await fetchTrainedModels();
      setActiveTab('predict');
    } catch (error) {
      toast({
        title: "Training Failed",
        description: error.response?.data?.detail || "Failed to train model",
        variant: "destructive",
      });
    } finally {
      setTraining(false);
    }
  };

  // Generate predictions
  const generatePredictions = async () => {
    if (!stockData) {
      toast({
        title: "Error",
        description: "Please fetch stock data first",
        variant: "destructive",
      });
      return;
    }

    setPredicting(true);
    try {
      const response = await axios.post(`${API}/predict`, {
        symbol: stockSymbol.toUpperCase(),
        days: predictionDays
      });

      setPredictions(response.data);
      toast({
        title: "Success",
        description: `Generated ${predictionDays} day predictions for ${stockSymbol.toUpperCase()}`,
      });
    } catch (error) {
      toast({
        title: "Prediction Failed",
        description: error.response?.data?.detail || "Failed to generate predictions",
        variant: "destructive",
      });
    } finally {
      setPredicting(false);
    }
  };

  // Fetch trained models
  const fetchTrainedModels = async () => {
    try {
      const response = await axios.get(`${API}/models`);
      setTrainedModels(response.data);
    } catch (error) {
      console.error('Failed to fetch trained models:', error);
    }
  };

  // Download predictions as CSV
  const downloadPredictions = async () => {
    if (!predictions) return;

    try {
      const response = await axios.get(`${API}/export-predictions/${stockSymbol.toUpperCase()}`, {
        responseType: 'blob'
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `${stockSymbol.toUpperCase()}_predictions.csv`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      
      toast({
        title: "Success",
        description: "Predictions downloaded successfully!",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to download predictions",
        variant: "destructive",
      });
    }
  };

  // Prepare chart data
  const getChartData = () => {
    if (!stockData) return [];
    
    const historicalData = stockData.data.slice(-100).map(item => ({
      date: new Date(item.date).toLocaleDateString(),
      price: item.close,
      type: 'historical'
    }));

    if (predictions) {
      const predictionData = predictions.predictions.map(item => ({
        date: new Date(item.date).toLocaleDateString(),
        price: item.predicted_price,
        type: 'prediction'
      }));
      return [...historicalData, ...predictionData];
    }

    return historicalData;
  };

  useEffect(() => {
    fetchTrainedModels();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <div className="container mx-auto p-6">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-slate-800 mb-2 flex items-center justify-center gap-2">
            <TrendingUp className="h-8 w-8 text-blue-600" />
            Stock Price Predictor
          </h1>
          <p className="text-slate-600 text-lg">AI-powered LSTM stock price prediction using machine learning</p>
        </div>

        {/* Controls */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Configuration
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="space-y-2">
                <Label htmlFor="symbol">Stock Symbol</Label>
                <Input
                  id="symbol"
                  placeholder="e.g., AAPL, TSLA, GOOGL"
                  value={stockSymbol}
                  onChange={(e) => setStockSymbol(e.target.value.toUpperCase())}
                  className="font-mono"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="period">Time Period</Label>
                <Select value={period} onValueChange={setPeriod}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1y">1 Year</SelectItem>
                    <SelectItem value="2y">2 Years</SelectItem>
                    <SelectItem value="5y">5 Years</SelectItem>
                    <SelectItem value="max">Max</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="epochs">Training Epochs</Label>
                <Input
                  id="epochs"
                  type="number"
                  min="10"
                  max="200"
                  value={epochs}
                  onChange={(e) => setEpochs(parseInt(e.target.value))}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="days">Prediction Days</Label>
                <Input
                  id="days"
                  type="number"
                  min="1"
                  max="30"
                  value={predictionDays}
                  onChange={(e) => setPredictionDays(parseInt(e.target.value))}
                />
              </div>
            </div>
            <div className="flex gap-2 mt-4">
              <Button onClick={fetchStockData} disabled={loading} className="bg-blue-600 hover:bg-blue-700">
                {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                Fetch Data
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Main Content */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="data">Data</TabsTrigger>
            <TabsTrigger value="visualize">Visualize</TabsTrigger>
            <TabsTrigger value="train">Train Model</TabsTrigger>
            <TabsTrigger value="predict">Predict</TabsTrigger>
          </TabsList>

          {/* Data Tab */}
          <TabsContent value="data">
            <Card>
              <CardHeader>
                <CardTitle>Stock Data Summary</CardTitle>
              </CardHeader>
              <CardContent>
                {stockData ? (
                  <div className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="text-center p-4 bg-blue-50 rounded-lg">
                        <div className="text-2xl font-bold text-blue-600">{stockData.symbol}</div>
                        <div className="text-sm text-slate-600">Symbol</div>
                      </div>
                      <div className="text-center p-4 bg-green-50 rounded-lg">
                        <div className="text-2xl font-bold text-green-600">{stockData.data.length}</div>
                        <div className="text-sm text-slate-600">Data Points</div>
                      </div>
                      <div className="text-center p-4 bg-purple-50 rounded-lg">
                        <div className="text-2xl font-bold text-purple-600">
                          ${stockData.data[stockData.data.length - 1]?.close.toFixed(2)}
                        </div>
                        <div className="text-sm text-slate-600">Latest Price</div>
                      </div>
                    </div>
                    <div className="text-sm text-slate-600">
                      Data range: {stockData.data[0]?.date} to {stockData.data[stockData.data.length - 1]?.date}
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-slate-500">
                    No data loaded. Please fetch stock data first.
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Visualize Tab */}
          <TabsContent value="visualize">
            <Card>
              <CardHeader>
                <CardTitle>Price Chart</CardTitle>
              </CardHeader>
              <CardContent>
                {stockData ? (
                  <div className="h-96">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={getChartData()}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis />
                        <Tooltip formatter={(value) => [`$${value.toFixed(2)}`, 'Price']} />
                        <Legend />
                        <Line 
                          type="monotone" 
                          dataKey="price" 
                          stroke="#2563eb" 
                          strokeWidth={2}
                          dot={false}
                          name="Stock Price"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="text-center py-8 text-slate-500">
                    No data to visualize. Please fetch stock data first.
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Train Model Tab */}
          <TabsContent value="train">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5" />
                  LSTM Model Training
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <h3 className="font-semibold text-blue-800 mb-2">Model Architecture</h3>
                    <ul className="text-sm text-blue-700 space-y-1">
                      <li>• 3-layer LSTM with dropout regularization</li>
                      <li>• Sequence length: 60 days</li>
                      <li>• Optimizer: Adam with learning rate 0.001</li>
                      <li>• Loss function: Mean Squared Error</li>
                    </ul>
                  </div>
                  
                  <Button 
                    onClick={trainModel} 
                    disabled={training || !stockData}
                    className="w-full bg-green-600 hover:bg-green-700"
                    size="lg"
                  >
                    {training ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Training Model... (This may take a few minutes)
                      </>
                    ) : (
                      <>
                        <Brain className="mr-2 h-4 w-4" />
                        Train LSTM Model
                      </>
                    )}
                  </Button>

                  {/* Trained Models */}
                  <div>
                    <h3 className="font-semibold mb-3">Trained Models</h3>
                    {trainedModels.length > 0 ? (
                      <div className="space-y-2">
                        {trainedModels.map((model) => (
                          <div key={model.id} className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                            <div>
                              <Badge variant="secondary">{model.symbol}</Badge>
                              <span className="ml-2 text-sm text-slate-600">
                                MSE: {model.metrics.test_mse.toFixed(4)}
                              </span>
                            </div>
                            <div className="text-xs text-slate-500">
                              {new Date(model.created_at).toLocaleDateString()}
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center py-4 text-slate-500">
                        No trained models yet.
                      </div>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Predict Tab */}
          <TabsContent value="predict">
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Target className="h-5 w-5" />
                    Generate Predictions
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <Button 
                    onClick={generatePredictions} 
                    disabled={predicting || !stockData}
                    className="w-full bg-purple-600 hover:bg-purple-700"
                    size="lg"
                  >
                    {predicting ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Generating Predictions...
                      </>
                    ) : (
                      <>
                        <Target className="mr-2 h-4 w-4" />
                        Generate {predictionDays} Day Predictions
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>

              {/* Predictions Results */}
              {predictions && (
                <div className="space-y-6">
                  {/* Chart with Predictions */}
                  <Card>
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <CardTitle>Predictions Chart</CardTitle>
                        <Button onClick={downloadPredictions} variant="outline" size="sm">
                          <Download className="mr-2 h-4 w-4" />
                          Download CSV
                        </Button>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="h-96">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={getChartData()}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="date" />
                            <YAxis />
                            <Tooltip formatter={(value) => [`$${value.toFixed(2)}`, 'Price']} />
                            <Legend />
                            <Line 
                              type="monotone" 
                              dataKey="price" 
                              stroke="#2563eb" 
                              strokeWidth={2}
                              dot={false}
                              name="Historical + Predictions"
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Predictions Table */}
                  <Card>
                    <CardHeader>
                      <CardTitle>Prediction Results</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                          <thead>
                            <tr className="border-b">
                              <th className="text-left p-2">Day</th>
                              <th className="text-left p-2">Date</th>
                              <th className="text-left p-2">Predicted Price</th>
                            </tr>
                          </thead>
                          <tbody>
                            {predictions.predictions.map((pred, index) => (
                              <tr key={index} className="border-b">
                                <td className="p-2">{pred.day}</td>
                                <td className="p-2">{pred.date}</td>
                                <td className="p-2 font-mono">${pred.predicted_price.toFixed(2)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Model Metrics */}
                  {predictions.model_metrics && (
                    <Card>
                      <CardHeader>
                        <CardTitle>Model Performance</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          <div className="text-center p-3 bg-blue-50 rounded-lg">
                            <div className="text-lg font-bold text-blue-600">
                              {predictions.model_metrics.test_mse.toFixed(4)}
                            </div>
                            <div className="text-xs text-slate-600">Test MSE</div>
                          </div>
                          <div className="text-center p-3 bg-green-50 rounded-lg">
                            <div className="text-lg font-bold text-green-600">
                              {predictions.model_metrics.test_mae.toFixed(2)}
                            </div>
                            <div className="text-xs text-slate-600">Test MAE</div>
                          </div>
                          <div className="text-center p-3 bg-purple-50 rounded-lg">
                            <div className="text-lg font-bold text-purple-600">
                              {predictions.model_metrics.final_loss.toFixed(4)}
                            </div>
                            <div className="text-xs text-slate-600">Final Loss</div>
                          </div>
                          <div className="text-center p-3 bg-orange-50 rounded-lg">
                            <div className="text-lg font-bold text-orange-600">
                              {predictions.model_metrics.final_val_loss.toFixed(4)}
                            </div>
                            <div className="text-xs text-slate-600">Validation Loss</div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </div>
              )}
            </div>
          </TabsContent>
        </Tabs>
      </div>
      <Toaster />
    </div>
  );
}

export default App;