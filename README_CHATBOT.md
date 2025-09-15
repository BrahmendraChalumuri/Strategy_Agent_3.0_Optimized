# Hybrid Chatbot System for Cross-Sell & Up-Sell Recommendations

A sophisticated AI-powered chatbot that combines structured database queries with unstructured PDF report analysis using LangChain, PostgreSQL, and vector databases.

## 🚀 Features

### Hybrid Intelligence
- **Structured Data Queries**: Natural language to SQL translation for customer, product, and sales data
- **PDF Knowledge Base**: Weaviate vector-based search through cross-sell and up-sell recommendation reports
- **Conversation Memory**: Maintains context across multiple queries
- **Intelligent Routing**: Automatically determines whether to query database or PDF reports

### Database Integration
- **PostgreSQL**: Robust relational database for structured data
- **Auto Schema Creation**: Automatically creates tables and relationships
- **CSV Data Ingestion**: Seamlessly loads existing project data
- **Sample Data Generation**: Creates realistic sales data for testing

### AI-Powered Analysis
- **LangChain Agents**: Zero-shot reasoning for complex queries
- **OpenAI Integration**: GPT-3.5-turbo for natural language understanding
- **Weaviate Vector Store**: Enterprise-grade vector database for PDF content search
- **Context-Aware Responses**: Combines database and PDF insights

## 📁 Project Structure

```
Strategy_Agent_3.0_Optimized/
├── chatbot.py                 # Main chatbot system
├── requirements_chatbot.txt   # Python dependencies
├── chatbot.env.template      # Environment configuration template
├── setup_database.sql        # Database setup instructions
├── README_CHATBOT.md         # This file
├── data/                     # CSV data files (existing)
│   ├── customer.csv
│   ├── products.csv
│   └── customer_catalogue_enhanced.csv
└── reports/                  # PDF reports (existing)
    ├── analysis_report_*.pdf
    └── combined_*.pdf
```

## 🛠️ Installation & Setup

### 1. Prerequisites

- Python 3.8 or higher
- PostgreSQL 12 or higher
- Weaviate (local Docker or cloud instance)
- OpenAI API key

### 2. Install Dependencies

```bash
pip install -r requirements_chatbot.txt
```

### 3. Database Setup

#### PostgreSQL Setup

##### Option A: Use existing PostgreSQL user
```bash
# Create database (as postgres user)
createdb strategy_agent_db

# Update your environment file
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/strategy_agent_db
```

##### Option B: Create dedicated user
```bash
# Run setup_database.sql as postgres superuser
psql -U postgres -f setup_database.sql

# Update your environment file
DATABASE_URL=postgresql://strategy_agent_user:your_password@localhost:5432/strategy_agent_db
```

#### Weaviate Setup

##### Option A: Local Docker (Recommended for development)
```bash
# Run Weaviate with Docker
docker run -p 8080:8080 -p 50051:50051 \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
  -e DEFAULT_VECTORIZER_MODULE='none' \
  -e CLUSTER_HOSTNAME='node1' \
  semitechnologies/weaviate:latest

# Verify Weaviate is running
curl http://localhost:8080/v1/meta
```

##### Option B: Weaviate Cloud Services (Recommended for production)
1. Sign up at [Weaviate Cloud Services](https://console.weaviate.cloud/)
2. Create a new cluster
3. Get your cluster URL and API key
4. Update your environment file:
```env
WEAVIATE_URL=https://your-cluster.weaviate.network
WEAVIATE_API_KEY=your-weaviate-api-key
```

### 4. Environment Configuration

```bash
# Copy template and configure
cp chatbot.env.template .env

# Edit .env with your credentials
nano .env
```

Required environment variables:
```env
OPENAI_API_KEY=your-openai-api-key-here
DATABASE_URL=postgresql://username:password@host:port/database_name
WEAVIATE_URL=http://localhost:8080
# Optional for Weaviate Cloud Services:
# WEAVIATE_API_KEY=your-weaviate-api-key
```

### 5. Run the Chatbot

```bash
python chatbot.py
```

## 💬 Example Queries

### Database Queries
- "Which customer has the most stores?"
- "Show me all biscuit products under $35"
- "What were the top 5 products by sales in January 2024?"
- "How many sales did Starbucks make last month?"
- "Which store bought the most biscuits?"

### PDF Report Queries
- "Show me cross-sell recommendations for Starbucks"
- "What customer classification analysis is available?"
- "Find up-sell opportunities in the latest reports"
- "What are the key insights from the portfolio analysis?"

### Hybrid Queries
- "Compare Starbucks sales data with their cross-sell recommendations"
- "Show me products that appear in both sales data and PDF recommendations"
- "Which customers have both high sales and good cross-sell opportunities?"

## 🏗️ Architecture

### Database Schema

```sql
-- Customers
customers(CustomerID, CustomerName, CustomerType, Country, Region, TotalStores)

-- Products  
products(ProductID, Name, Category, SubCategory, Price, Tags)

-- Customer Catalogue
customer_catalogue(CatalogueID, CustomerID, ProductName, Category, Description, Ingredients, Calories)

-- Sales
sales(SaleID, CustomerID, StoreID, ProductID, PlantID, Quantity, UnitPrice, TotalAmount, SaleDate, DeliveryDate, Status)
```

### System Components

1. **Database Layer**: PostgreSQL with SQLAlchemy ORM
2. **Vector Store**: Weaviate for PDF embeddings and semantic search
3. **LangChain Agents**: Zero-shot reasoning with tool selection
4. **Memory System**: ConversationBufferMemory for context
5. **Interactive Interface**: Command-line chat interface

### Data Flow

```
User Query → LangChain Agent → Tool Selection → Database/PDF Search → Response Generation
```

## 🔧 Configuration

### Rate Limiting
The system includes built-in rate limiting for OpenAI API calls to prevent exceeding quotas.

### Memory Management
- Conversation memory persists for the session
- Vector store loads all PDF reports at startup
- Database connections use connection pooling

### Performance Optimization
- Weaviate vector store for fast similarity search and enterprise scalability
- SQL indexes for database performance
- Chunked PDF processing for large documents
- Persistent vector storage with metadata filtering

## 🧪 Testing

### Manual Testing
```bash
# Start the chatbot
python chatbot.py

# Test database queries
"Show me all customers"

# Test PDF queries  
"Find Starbucks recommendations"

# Test hybrid queries
"Compare customer sales with recommendations"
```

### Automated Testing
```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests (when implemented)
pytest tests/
```

## 🚨 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check PostgreSQL is running
   - Verify DATABASE_URL format
   - Ensure database exists

2. **Weaviate Connection Failed**
   - Check Weaviate is running (Docker or cloud)
   - Verify WEAVIATE_URL is correct
   - Check WEAVIATE_API_KEY if using cloud services
   - Test connection: `curl http://localhost:8080/v1/meta`

3. **OpenAI API Errors**
   - Verify OPENAI_API_KEY is correct
   - Check API quota and billing
   - Monitor rate limiting

4. **PDF Loading Issues**
   - Ensure reports/ directory exists
   - Check PDF file permissions
   - Verify PDF files are not corrupted

5. **Memory Issues**
   - Reduce PDF chunk size
   - Limit number of concurrent queries
   - Monitor system memory usage

### Debug Mode
```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python chatbot.py
```

## 📈 Performance Metrics

The system tracks:
- Database query response times
- Vector search performance
- OpenAI API usage
- Memory consumption
- Conversation context size

## 🔮 Future Enhancements

- **Web Interface**: Flask/FastAPI web UI
- **Multi-Modal**: Support for images and charts
- **Real-Time Updates**: WebSocket for live data
- **Advanced Analytics**: Custom dashboard
- **API Endpoints**: RESTful API for integration
- **Caching**: Redis for improved performance
- **Monitoring**: Prometheus metrics and alerts

## 📄 License

This project extends the existing Strategy Agent 3.0 system and follows the same licensing terms.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the logs for error details
- Ensure all dependencies are installed
- Verify environment configuration

---

**Built with ❤️ using LangChain, PostgreSQL, and OpenAI**
