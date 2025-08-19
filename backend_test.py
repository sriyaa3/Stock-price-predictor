import requests
import sys
import time
import json
from datetime import datetime

class StockPredictorAPITester:
    def __init__(self, base_url="https://stock-predictor-5.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name, success, message="", response_data=None):
        """Log test results"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name}: PASSED - {message}")
        else:
            print(f"❌ {name}: FAILED - {message}")
        
        self.test_results.append({
            "test": name,
            "success": success,
            "message": message,
            "response_data": response_data
        })

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=30):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)
            
            success = response.status_code == expected_status
            
            if success:
                try:
                    response_json = response.json()
                    self.log_test(name, True, f"Status: {response.status_code}", response_json)
                    return True, response_json
                except:
                    self.log_test(name, True, f"Status: {response.status_code} (Non-JSON response)")
                    return True, {}
            else:
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                except:
                    error_detail = response.text[:200]
                self.log_test(name, False, f"Expected {expected_status}, got {response.status_code}: {error_detail}")
                return False, {}

        except requests.exceptions.Timeout:
            self.log_test(name, False, f"Request timed out after {timeout} seconds")
            return False, {}
        except Exception as e:
            self.log_test(name, False, f"Error: {str(e)}")
            return False, {}

    def test_api_root(self):
        """Test API root endpoint"""
        return self.run_test("API Root", "GET", "", 200)

    def test_fetch_stock_data(self, symbol="AAPL", period="2y"):
        """Test fetching stock data"""
        data = {"symbol": symbol, "period": period}
        return self.run_test(
            f"Fetch Stock Data ({symbol})", 
            "POST", 
            "fetch-stock", 
            200, 
            data,
            timeout=60  # Longer timeout for data fetching
        )

    def test_get_stock_data(self, symbol="AAPL"):
        """Test retrieving stored stock data"""
        return self.run_test(
            f"Get Stock Data ({symbol})", 
            "GET", 
            f"stock-data/{symbol}", 
            200
        )

    def test_train_model(self, symbol="AAPL", epochs=20):
        """Test model training with reduced epochs for faster testing"""
        data = {
            "symbol": symbol,
            "epochs": epochs,
            "batch_size": 32,
            "sequence_length": 60
        }
        return self.run_test(
            f"Train Model ({symbol})", 
            "POST", 
            "train-model", 
            200, 
            data,
            timeout=300  # 5 minutes for training
        )

    def test_generate_predictions(self, symbol="AAPL", days=5):
        """Test generating predictions"""
        data = {"symbol": symbol, "days": days}
        return self.run_test(
            f"Generate Predictions ({symbol})", 
            "POST", 
            "predict", 
            200, 
            data,
            timeout=60
        )

    def test_get_models(self):
        """Test getting trained models list"""
        return self.run_test("Get Models List", "GET", "models", 200)

    def test_get_predictions(self, symbol="AAPL"):
        """Test getting stored predictions"""
        return self.run_test(
            f"Get Predictions ({symbol})", 
            "GET", 
            f"predictions/{symbol}", 
            200
        )

    def test_export_predictions(self, symbol="AAPL"):
        """Test CSV export functionality"""
        url = f"{self.api_url}/export-predictions/{symbol}"
        print(f"\n🔍 Testing Export Predictions CSV ({symbol})...")
        print(f"   URL: {url}")
        
        try:
            response = requests.get(url, timeout=30)
            success = response.status_code == 200
            
            if success:
                # Check if response is CSV format
                content_type = response.headers.get('content-type', '')
                if 'csv' in content_type.lower() or len(response.content) > 0:
                    self.log_test(f"Export Predictions CSV ({symbol})", True, f"CSV downloaded, size: {len(response.content)} bytes")
                    return True, {"csv_size": len(response.content)}
                else:
                    self.log_test(f"Export Predictions CSV ({symbol})", False, "Response is not CSV format")
                    return False, {}
            else:
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                except:
                    error_detail = response.text[:200]
                self.log_test(f"Export Predictions CSV ({symbol})", False, f"Status {response.status_code}: {error_detail}")
                return False, {}
                
        except Exception as e:
            self.log_test(f"Export Predictions CSV ({symbol})", False, f"Error: {str(e)}")
            return False, {}

    def test_invalid_symbol(self):
        """Test error handling with invalid stock symbol"""
        data = {"symbol": "INVALID123", "period": "1y"}
        success, response = self.run_test(
            "Invalid Symbol Error Handling", 
            "POST", 
            "fetch-stock", 
            404,  # Expecting 404 for invalid symbol
            data,
            timeout=30
        )
        return success, response

    def run_comprehensive_test(self):
        """Run all tests in sequence"""
        print("🚀 Starting Stock Price Predictor API Tests")
        print(f"📍 Testing against: {self.base_url}")
        print("=" * 60)

        # Test 1: API Root
        self.test_api_root()

        # Test 2: Fetch stock data for AAPL
        print("\n📊 Testing Stock Data Operations...")
        fetch_success, _ = self.test_fetch_stock_data("AAPL", "2y")
        
        if fetch_success:
            # Test 3: Get stored stock data
            self.test_get_stock_data("AAPL")
            
            # Test 4: Train model (with reduced epochs for speed)
            print("\n🧠 Testing Model Training...")
            train_success, _ = self.test_train_model("AAPL", epochs=20)
            
            if train_success:
                # Wait a moment for model to be fully stored
                time.sleep(2)
                
                # Test 5: Generate predictions
                print("\n🔮 Testing Predictions...")
                predict_success, _ = self.test_generate_predictions("AAPL", days=5)
                
                if predict_success:
                    # Test 6: Get predictions
                    self.test_get_predictions("AAPL")
                    
                    # Test 7: Export predictions as CSV
                    self.test_export_predictions("AAPL")
                
                # Test 8: Get models list
                self.test_get_models()
            else:
                print("⚠️  Skipping prediction tests due to training failure")
        else:
            print("⚠️  Skipping subsequent tests due to data fetch failure")

        # Test 9: Test different stock symbol
        print("\n📈 Testing with Different Stock Symbol...")
        tsla_success, _ = self.test_fetch_stock_data("TSLA", "1y")
        if tsla_success:
            self.test_get_stock_data("TSLA")

        # Test 10: Error handling
        print("\n🚨 Testing Error Handling...")
        self.test_invalid_symbol()

        # Print final results
        self.print_summary()

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        if self.tests_run - self.tests_passed > 0:
            print("\n❌ FAILED TESTS:")
            for result in self.test_results:
                if not result['success']:
                    print(f"   • {result['test']}: {result['message']}")
        
        print("\n" + "=" * 60)
        return self.tests_passed == self.tests_run

def main():
    """Main test execution"""
    tester = StockPredictorAPITester()
    
    try:
        success = tester.run_comprehensive_test()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n💥 Unexpected error during testing: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())