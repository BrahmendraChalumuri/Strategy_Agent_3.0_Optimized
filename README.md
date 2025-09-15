# Strategy Agent 3.0 - Optimized Recommendation Engine

A high-performance AI-powered recommendation system that generates cross-sell and up-sell opportunities for customers using advanced machine learning, natural language processing, and real-time API integration.

## 🚀 Features

### Core Capabilities
- **AI-Powered Ingredient Matching**: Uses sentence transformers and Perplexity AI for intelligent product recommendations
- **Customer Classification**: Automatically categorizes customers (CHG Own Sales, Distributor, Small Customer) based on business metrics
- **Cross-Sell Analysis**: Identifies products customers haven't purchased but should consider based on ingredient compatibility
- **Performance Optimization**: Implements caching, parallel processing, and batch operations for maximum efficiency
- **Real-time API Integration**: FastAPI-based REST API with automatic startup generation and background processing

### Technical Highlights
- **Vector Embeddings**: Persistent caching of product embeddings for fast similarity calculations
- **Async Processing**: Parallel API calls with rate limiting and exponential backoff
- **Batch Operations**: Optimized pandas operations and vectorized calculations
- **Memory Management**: Efficient lookup dictionaries and pre-computed data structures
- **PDF Report Generation**: Automated comprehensive analysis reports

## 📁 Project Structure

```
Strategy_Agent_3.0/
├── 📊 data/                          # Customer and product data
│   ├── customer.csv                  # Customer information
│   ├── customer_catalogue_enhanced.csv # Customer product catalogues
│   ├── products.csv                  # Product database
│   ├── sales_enhanced.csv           # Sales transaction data
│   └── stores.csv                   # Store information
├── 🧠 cache/                        # ML model and embedding cache
│   └── *.pkl                        # Pickled embeddings and models
├── 📋 recommendations/              # Generated recommendation files
│   └── recommendations_*.json       # JSON recommendation outputs
├── 📄 reports/                      # PDF analysis reports
│   └── analysis_report_*.pdf        # Generated PDF reports
├── 🔧 main_optimized.py             # Core recommendation engine
├── 🌐 api_endpoint_optimized.py     # FastAPI REST API server
├── 📊 pdf_report_generator.py       # PDF report generation
├── ⚙️ rate_limit_config.py          # API rate limiting configuration
├── 🔑 api_keys.env.template         # API keys template
└── 📦 requirements_optimized.txt    # Python dependencies
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Strategy_Agent_3.0
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_optimized.txt
   ```

3. **Configure API keys**
   ```bash
   cp api_keys.env.template api_keys.env
   # Edit api_keys.env and add your Perplexity API key
   ```

4. **Verify data files**
   Ensure all CSV files are present in the `data/` directory:
   - `customer.csv`
   - `customer_catalogue_enhanced.csv`
   - `products.csv`
   - `sales_enhanced.csv`
   - `stores.csv`

## 🚀 Quick Start

### Option 1: API Server (Recommended)
Start the FastAPI server for web-based access:

```bash
python api_endpoint_optimized.py
```

The server will:
- Initialize the recommendation engine
- Automatically generate recommendations for all customers
- Start on `http://localhost:8002`

**API Documentation:**
- Swagger UI: `http://localhost:8002/docs`
- ReDoc: `http://localhost:8002/redoc`
- Health Check: `http://localhost:8002/health`

### Option 2: Direct Python Usage
Use the recommendation engine directly in Python:

```python
from main_optimized import OptimizedRecommendationEngine
import asyncio

async def main():
    # Initialize engine
    engine = OptimizedRecommendationEngine()
    
    # Generate recommendations for a customer
    recommendations, rejected, already_purchased, classification = await engine.generate_recommendations("C001")
    
    # Display results
    engine.display_recommendations(recommendations, rejected, already_purchased, "C001", classification)

# Run the analysis
asyncio.run(main())
```

## 📊 API Endpoints

### Customer Management
- `GET /customers` - List all available customers
- `GET /recommendations` - List all existing recommendation files

### Recommendations
- `GET /recommendations/{customer_id}/json` - Get JSON recommendations for a customer
- `GET /recommendations/{customer_id}/download` - Download PDF report for a customer
- `GET /recommendations/all` - Get all customer recommendations as JSON objects array
- `GET /recommendations/download/all` - Download combined PDF with all customer reports
- `POST /regenerate_recommendations/{customer_id}` - Regenerate recommendations for a specific customer (returns recommendation data, deletes old files only after successful generation)
- `POST /regenerate_recommendations` - Regenerate recommendations for all customers (returns array of all recommendation data, deletes old files immediately after each customer's successful generation)

### System Monitoring
- `GET /health` - Health check with performance metrics
- `GET /performance-stats` - Detailed performance statistics

## 🧠 How It Works

### 1. Data Processing
- Loads customer, product, and sales data using optimized pandas operations
- Creates lookup dictionaries for O(1) access to customer and product information
- Pre-computes customer classifications based on store count and purchase volume

### 2. AI-Powered Analysis
- Generates vector embeddings for all products using sentence transformers
- Caches embeddings for fast similarity calculations
- Uses Perplexity AI to validate ingredient compatibility between products

### 3. Recommendation Generation
- Analyzes customer catalogue items and their ingredients
- Finds similar products using cosine similarity (threshold: 0.7)
- Validates recommendations through AI analysis
- Categorizes results as accepted, rejected, or already purchased

### 4. Performance Optimization
- Parallel API calls with rate limiting (30 requests/minute)
- Batch similarity calculations using PyTorch tensors
- Persistent caching of embeddings and API responses
- Memory-efficient data structures and vectorized operations

## 📈 Performance Features

### Caching System
- **Product Embeddings**: Cached in `cache/product_embeddings.pkl`
- **Ingredient Embeddings**: Individual cache files for each ingredient
- **API Responses**: Intelligent caching to minimize API calls

### Rate Limiting
- Configurable rate limits (default: 30 requests/minute)
- Exponential backoff for failed requests
- Concurrent request management with semaphores

### Batch Processing
- Vectorized similarity calculations
- Parallel data loading using ThreadPoolExecutor
- Batch API calls for multiple ingredients

## 📊 Customer Classification

The system automatically classifies customers into three categories:

### CHG Own Sales Customer
- **Criteria**: >50 stores OR >200,000 units sold
- **Strategy**: Premium attention, comprehensive recommendations

### Distributor Customer
- **Criteria**: 25-50 stores OR 50,000-200,000 units sold
- **Strategy**: Standard recommendations, medium-scale focus

### Small Customer
- **Criteria**: <25 stores AND <50,000 units sold
- **Strategy**: Basic recommendations, growth-focused approach

## 📄 Output Formats

### JSON Recommendations
Comprehensive JSON files containing:
- Customer classification and metrics
- Accepted cross-sell recommendations
- Rejected recommendations with AI reasoning
- Already purchased items
- Performance statistics

### PDF Reports
Professional PDF reports featuring:
- Executive summary with key metrics
- Customer classification analysis
- Detailed cross-sell recommendations
- Strategic implications and next steps

## 🔧 Configuration

### Rate Limiting
Configure API rate limits in `rate_limit_config.py`:
```python
def get_rate_limit_config():
    return {
        "max_requests_per_minute": 30,
        "max_concurrent_requests": 5
    }
```

### API Keys
Set your Perplexity API key in `api_keys.env`:
```
PERPLEXITY_API_KEY=your_api_key_here
```

## 📊 Performance Monitoring

The system tracks comprehensive performance metrics:
- Embedding cache hit/miss rates
- API call statistics
- Similarity calculation counts
- Processing times
- Memory usage

Access performance stats via:
- API endpoint: `GET /performance-stats`
- Health check: `GET /health`

## 🛡️ Error Handling

### Robust Error Management
- Graceful API failure handling with fallback strategies
- Comprehensive logging with structured output
- Automatic retry mechanisms with exponential backoff
- Input validation and data integrity checks

### Logging
- Structured logging with timestamps and severity levels
- Performance tracking and bottleneck identification
- API call monitoring and rate limit management

## 🔄 Data Flow

1. **Startup**: Load data, create embeddings, initialize caches
2. **Customer Analysis**: Classify customer, retrieve purchase history
3. **Ingredient Processing**: Extract and clean ingredient lists
4. **Similarity Matching**: Find similar products using vector embeddings
5. **AI Validation**: Use Perplexity AI to validate recommendations
6. **Result Categorization**: Sort into accepted/rejected/already purchased
7. **Report Generation**: Create JSON and PDF outputs

## 🚀 Advanced Features

### Parallel Processing
- Concurrent API calls for multiple ingredients
- ThreadPoolExecutor for data loading
- Async/await patterns throughout the system

### Memory Optimization
- Pre-computed lookup dictionaries
- Efficient pandas operations
- Vectorized calculations with NumPy/PyTorch

### Scalability
- Configurable rate limits and concurrency
- Persistent caching for large datasets
- Modular architecture for easy extension

## 📝 Dependencies

### Core Libraries
- **pandas**: Data manipulation and analysis
- **fastapi**: High-performance web framework
- **sentence-transformers**: AI-powered text embeddings
- **torch**: Deep learning framework
- **aiohttp**: Async HTTP client
- **reportlab**: PDF generation

### Optional Dependencies
- **pytest**: Testing framework
- **httpx**: HTTP client for testing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the API documentation at `/docs`
- Review the performance stats at `/performance-stats`
- Monitor system health at `/health`

## 🔮 Future Enhancements

- Real-time recommendation updates
- Machine learning model improvements
- Advanced customer segmentation
- Integration with external CRM systems
- Mobile application support
- Advanced analytics dashboard

---

**Strategy Agent 3.0** - Empowering data-driven customer recommendations through AI and advanced analytics.
