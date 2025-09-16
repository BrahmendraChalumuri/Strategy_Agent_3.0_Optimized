# Strategy Agent Chatbot API

A comprehensive FastAPI-based REST API for the Strategy Agent Chatbot system, providing endpoints for chatbot conversations, database queries, and system management.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_chatbot.txt
```

### 2. Configure Environment
```bash
# Copy the template
cp chatbot.env.template chatbot.env

# Edit chatbot.env with your configuration
# Required variables:
# - DATABASE_URL
# - OPENAI_API_KEY
# - PERPLEXITY_API_KEY (optional)
# - WEAVIATE_URL (optional)
# - WEAVIATE_API_KEY (optional)
```

### 3. Initialize System
```bash
# Initialize database and vector store
python chatbot_init.py
```

### 4. Start API Server
```bash
# Start the API server directly
python chatbot_api.py

# Or using uvicorn directly
uvicorn chatbot_api:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Access API
- **API Server**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📚 API Endpoints

### Core Endpoints

#### `GET /`
Root endpoint with API information
```json
{
  "message": "Strategy Agent Chatbot API",
  "version": "1.0.0",
  "status": "running",
  "docs": "/docs",
  "health": "/health"
}
```

#### `GET /health`
Health check endpoint
```json
{
  "status": "healthy",
  "timestamp": "2025-01-16T12:00:00",
  "database_connected": true,
  "vector_store_connected": true,
  "chatbot_ready": true
}
```

#### `POST /chat`
Chat with the chatbot
```json
// Request
{
  "message": "What products does customer C001 buy?",
  "customer_id": "C001",
  "session_id": "session_123"
}

// Response
{
  "response": "Customer C001 (Starbucks) purchases various beverage products...",
  "session_id": "session_123",
  "timestamp": "2025-01-16T12:00:00",
  "customer_context": {
    "customerid": "C001",
    "customername": "Starbucks",
    "customertype": "Licensed",
    "country": "International",
    "region": "Middle East",
    "totalstores": 99
  }
}
```

### Data Endpoints

#### `GET /customers`
Get list of all customers
```json
[
  {
    "customer_id": "C001",
    "customer_name": "Starbucks",
    "customer_type": "Licensed",
    "country": "International",
    "region": "Middle East",
    "total_stores": 99
  }
]
```

#### `GET /products`
Get list of products with optional filtering
```bash
# Get all products (limit 100)
GET /products

# Get products by category
GET /products?category=Biscuits&limit=50
```

#### `GET /recommendations/{customer_id}`
Get recommendations for a specific customer
```json
{
  "customer_id": "C001",
  "customer_name": "Starbucks",
  "catalogue_items": 25,
  "total_sales": 150,
  "catalogue": [...],
  "recent_sales": [...]
}
```

### System Management

#### `POST /initialize`
Initialize the system (runs chatbot_init.py in background)
```json
{
  "message": "System initialization started in background",
  "status": "initializing"
}
```

#### `GET /status`
Get system status and initialization info
```json
{
  "initialized": true,
  "database_tables": ["customers", "products", "customer_catalogue", "sales"],
  "vector_store_collections": ["PDFReports"],
  "last_initialization": null
}
```

#### `GET /search`
Search documents in vector store
```bash
GET /search?query=Starbucks recommendations&limit=5
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Yes | - |
| `OPENAI_API_KEY` | OpenAI API key for LLM | Yes | - |
| `PERPLEXITY_API_KEY` | Perplexity API key | No | - |
| `WEAVIATE_URL` | Weaviate cluster URL | No | - |
| `WEAVIATE_API_KEY` | Weaviate API key | No | - |
| `API_HOST` | API server host | No | 0.0.0.0 |
| `API_PORT` | API server port | No | 8000 |
| `API_RELOAD` | Enable auto-reload | No | false |

### Database Schema

The API expects the following database tables:
- `customers` - Customer information
- `products` - Product catalog
- `customer_catalogue` - Customer-specific product catalogues
- `sales` - Sales transaction data

## 🧪 Testing

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Chat with bot
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?"}'

# Get customers
curl http://localhost:8000/customers

# Get products
curl http://localhost:8000/products?limit=10
```

### Using Python requests

```python
import requests

# Chat with bot
response = requests.post("http://localhost:8000/chat", json={
    "message": "What products does customer C001 buy?",
    "customer_id": "C001"
})
print(response.json())

# Get customers
customers = requests.get("http://localhost:8000/customers")
print(customers.json())
```

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements_chatbot.txt .
RUN pip install -r requirements_chatbot.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "chatbot_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Configuration

```bash
# Set production environment variables
export API_HOST=0.0.0.0
export API_PORT=8000
export API_RELOAD=false

# Use production ASGI server
pip install gunicorn
gunicorn chatbot_api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 🔍 Monitoring and Logging

The API includes comprehensive logging and monitoring:

- **Structured Logging**: All requests and responses are logged
- **Health Checks**: Monitor system status and dependencies
- **Error Handling**: Graceful error handling with detailed error messages
- **Performance Tracking**: Built-in performance metrics

## 🛠️ Development

### Code Structure

```
chatbot_api.py          # Main API application with startup functionality
requirements_chatbot.txt # All dependencies (chatbot + API)
README_API.md          # This documentation
```

### Adding New Endpoints

1. Define Pydantic models for request/response
2. Create endpoint function with proper error handling
3. Add dependency injection for database/chatbot access
4. Update documentation

### Error Handling

The API includes comprehensive error handling:
- HTTP exceptions with proper status codes
- Database connection errors
- Chatbot initialization errors
- Vector store connection errors

## 📊 API Response Formats

All API responses follow a consistent format:

### Success Response
```json
{
  "data": {...},
  "timestamp": "2025-01-16T12:00:00"
}
```

### Error Response
```json
{
  "error": "Error message",
  "detail": "Detailed error information",
  "status_code": 500,
  "timestamp": "2025-01-16T12:00:00"
}
```

## 🔐 Security Considerations

- **CORS**: Configured for cross-origin requests (adjust for production)
- **Input Validation**: All inputs validated using Pydantic models
- **SQL Injection**: Protected using SQLAlchemy ORM
- **Rate Limiting**: Consider implementing rate limiting for production
- **Authentication**: Add authentication/authorization as needed

## 📈 Performance

- **Async Processing**: Built on FastAPI for high performance
- **Connection Pooling**: Database connection pooling
- **Caching**: Vector store and embedding caching
- **Background Tasks**: Long-running tasks run in background

## 🆘 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check `DATABASE_URL` in chatbot.env
   - Ensure PostgreSQL is running
   - Verify database credentials

2. **Chatbot Not Initialized**
   - Run `python chatbot_init.py` first
   - Check OpenAI API key
   - Verify database tables exist

3. **Vector Store Not Available**
   - Check Weaviate connection
   - Verify FAISS cache exists
   - API will work without vector store

4. **API Server Won't Start**
   - Check port availability
   - Verify all dependencies installed
   - Check environment variables

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python start_chatbot_api.py
```

## 📞 Support

For issues and questions:
1. Check the logs for error messages
2. Verify environment configuration
3. Test individual components
4. Check API documentation at `/docs`
