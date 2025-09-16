#!/usr/bin/env python3
"""
Chatbot API Server
==================

FastAPI-based REST API for the Strategy Agent Chatbot system.
Provides endpoints for chatbot conversations, database queries, and system management.

Usage:
    python chatbot_api.py

Endpoints:
    - POST /chat - Chat with the bot
    - GET /health - Health check
    - GET /customers - List customers
    - GET /products - List products
    - GET /recommendations/{customer_id} - Get recommendations
    - POST /initialize - Initialize system
"""

import os
import sys
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Database imports
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# LangChain imports
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool

# Vector database and embeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Weaviate, FAISS
import weaviate

# Environment and utilities
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv('chatbot.env')

# Global variables for chatbot components
chatbot_agent = None
vector_store = None
db = None

# Pydantic models for API requests/responses
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to send to the chatbot")
    customer_id: Optional[str] = Field(None, description="Customer ID for context")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Chatbot response")
    session_id: str = Field(..., description="Session ID")
    timestamp: datetime = Field(default_factory=datetime.now)
    customer_context: Optional[Dict[str, Any]] = Field(None, description="Customer context if provided")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now)
    database_connected: bool = Field(..., description="Database connection status")
    vector_store_connected: bool = Field(..., description="Vector store connection status")
    chatbot_ready: bool = Field(..., description="Chatbot initialization status")

class CustomerResponse(BaseModel):
    customer_id: str
    customer_name: str
    customer_type: str
    country: str
    region: str
    total_stores: int

class ProductResponse(BaseModel):
    product_id: int
    name: str
    category: str
    subcategory: str
    price: float
    tags: str

class RecommendationRequest(BaseModel):
    customer_id: str = Field(..., description="Customer ID to get recommendations for")

class SystemStatus(BaseModel):
    initialized: bool
    database_tables: List[str]
    vector_store_collections: List[str]
    last_initialization: Optional[datetime]

# Initialize FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    # Startup
    logger.info("🚀 Starting Chatbot API Server...")
    try:
        await initialize_chatbot_components()
        logger.info("✅ Chatbot API Server started successfully")
    except Exception as e:
        logger.error(f"❌ Failed to start Chatbot API Server: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Chatbot API Server...")

app = FastAPI(
    title="Strategy Agent Chatbot API",
    description="AI-powered chatbot for customer analysis and recommendations",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get database connection
def get_database():
    """Get database connection"""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    return db

# Dependency to get chatbot agent
def get_chatbot_agent():
    """Get chatbot agent"""
    if chatbot_agent is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    return chatbot_agent

# Dependency to get vector store
def get_vector_store():
    """Get vector store"""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    return vector_store

async def initialize_chatbot_components():
    """Initialize chatbot components"""
    global chatbot_agent, vector_store, db
    
    try:
        # Database configuration
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            raise ValueError("DATABASE_URL not found in chatbot.env")
        
        # Initialize database
        engine = create_engine(database_url)
        db = SQLDatabase(engine)
        logger.info("✅ Database connected successfully")
        
        # OpenAI configuration
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in chatbot.env")
        
        # Initialize LLM
        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-3.5-turbo",
            temperature=0.1
        )
        
        # Initialize SQL toolkit
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        
        # Initialize memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize agent
        chatbot_agent = initialize_agent(
            tools=toolkit.get_tools(),
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True
        )
        
        logger.info("✅ Chatbot agent initialized successfully")
        
        # Initialize vector store (optional)
        try:
            await initialize_vector_store()
        except Exception as e:
            logger.warning(f"⚠️ Vector store initialization failed: {str(e)}")
            logger.info("📝 Continuing without vector store - chatbot will work with database queries only")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize chatbot components: {str(e)}")
        raise

async def initialize_vector_store():
    """Initialize vector store for PDF reports"""
    global vector_store
    
    try:
        # Weaviate configuration
        weaviate_url = os.getenv('WEAVIATE_URL')
        weaviate_api_key = os.getenv('WEAVIATE_API_KEY')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY required for vector store")
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Try to connect to Weaviate
        if weaviate_url and weaviate_api_key:
            # For Weaviate Cloud Services
            weaviate_client = weaviate.connect_to_weaviate_cloud(
                cluster_url=weaviate_url,
                auth_credentials=weaviate.AuthApiKey(api_key=weaviate_api_key)
            )
        else:
            # For local Weaviate instance
            weaviate_client = weaviate.connect_to_local(
                host="localhost",
                port=8080
            )
        
        # Test connection
        weaviate_client.get_meta()
        
        # Try to get existing collection
        try:
            collection = weaviate_client.collections.get("PDFReports")
            if collection.aggregate().total_count > 0:
                vector_store = Weaviate(
                    client=weaviate_client,
                    index_name="PDFReports",
                    text_key="content",
                    embedding=embeddings
                )
                logger.info("✅ Connected to existing Weaviate vector store")
            else:
                logger.info("📝 Weaviate collection exists but is empty")
                weaviate_client.close()
        except:
            logger.info("📝 Weaviate collection doesn't exist, using FAISS fallback")
            weaviate_client.close()
            
            # Fallback to FAISS
            try:
                vector_store = FAISS.load_local("cache/faiss_vector_store", embeddings)
                logger.info("✅ Loaded FAISS vector store from cache")
            except:
                logger.info("📝 No FAISS cache found, vector store not available")
                
    except Exception as e:
        logger.warning(f"⚠️ Vector store setup failed: {str(e)}")
        # Continue without vector store

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Strategy Agent Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        database_connected = False
        if db:
            try:
                db.run("SELECT 1")
                database_connected = True
            except:
                pass
        
        # Check vector store connection
        vector_store_connected = vector_store is not None
        
        # Check chatbot readiness
        chatbot_ready = chatbot_agent is not None
        
        status = "healthy" if all([database_connected, chatbot_ready]) else "degraded"
        
        return HealthResponse(
            status=status,
            database_connected=database_connected,
            vector_store_connected=vector_store_connected,
            chatbot_ready=chatbot_ready
        )
    except Exception as e:
        logger.error(f"❌ Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    agent=Depends(get_chatbot_agent),
    database=Depends(get_database)
):
    """Chat with the chatbot"""
    try:
        # Add customer context if provided
        context_message = request.message
        if request.customer_id:
            # Get customer information
            try:
                customer_info = database.run(f"""
                    SELECT customername, customertype, country, region, totalstores 
                    FROM customers 
                    WHERE customerid = '{request.customer_id}'
                """)
                if customer_info:
                    context_message = f"Customer Context: {customer_info}\n\nUser Question: {request.message}"
            except Exception as e:
                logger.warning(f"⚠️ Failed to get customer context: {str(e)}")
        
        # Get response from chatbot
        response = agent.run(context_message)
        
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get customer context for response
        customer_context = None
        if request.customer_id:
            try:
                customer_context = database.run(f"""
                    SELECT * FROM customers 
                    WHERE customerid = '{request.customer_id}'
                """)
            except:
                pass
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            customer_context=customer_context
        )
        
    except Exception as e:
        logger.error(f"❌ Chat request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat request failed: {str(e)}")

@app.get("/customers", response_model=List[CustomerResponse])
async def get_customers(database=Depends(get_database)):
    """Get list of all customers"""
    try:
        result = database.run("""
            SELECT customerid, customername, customertype, country, region, totalstores 
            FROM customers 
            ORDER BY customername
        """)
        
        customers = []
        for row in result:
            customers.append(CustomerResponse(
                customer_id=row[0],
                customer_name=row[1],
                customer_type=row[2],
                country=row[3],
                region=row[4],
                total_stores=row[5]
            ))
        
        return customers
        
    except Exception as e:
        logger.error(f"❌ Failed to get customers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get customers: {str(e)}")

@app.get("/products", response_model=List[ProductResponse])
async def get_products(
    limit: int = 100,
    category: Optional[str] = None,
    database=Depends(get_database)
):
    """Get list of products with optional filtering"""
    try:
        query = """
            SELECT productid, name, category, subcategory, price, tags 
            FROM products 
        """
        
        if category:
            query += f" WHERE category = '{category}'"
        
        query += f" ORDER BY name LIMIT {limit}"
        
        result = database.run(query)
        
        products = []
        for row in result:
            products.append(ProductResponse(
                product_id=row[0],
                name=row[1],
                category=row[2],
                subcategory=row[3],
                price=row[4],
                tags=row[5]
            ))
        
        return products
        
    except Exception as e:
        logger.error(f"❌ Failed to get products: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get products: {str(e)}")

@app.get("/recommendations/{customer_id}")
async def get_recommendations(
    customer_id: str,
    database=Depends(get_database)
):
    """Get recommendations for a specific customer"""
    try:
        # Check if customer exists
        customer_check = database.run(f"""
            SELECT customername FROM customers WHERE customerid = '{customer_id}'
        """)
        
        if not customer_check:
            raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
        
        # Get customer catalogue
        catalogue = database.run(f"""
            SELECT customercatalogueitemid, productname, product_category, 
                   description, ingredients, quantityrequired
            FROM customer_catalogue 
            WHERE customerid = '{customer_id}'
        """)
        
        # Get sales data
        sales = database.run(f"""
            SELECT productid, quantity, totalamount, saledate
            FROM sales 
            WHERE customerid = '{customer_id}'
            ORDER BY saledate DESC
        """)
        
        return {
            "customer_id": customer_id,
            "customer_name": customer_check[0][0],
            "catalogue_items": len(catalogue),
            "total_sales": len(sales),
            "catalogue": catalogue,
            "recent_sales": sales[:10]  # Last 10 sales
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")

@app.post("/initialize")
async def initialize_system(background_tasks: BackgroundTasks):
    """Initialize the system (run chatbot_init.py)"""
    try:
        # Add initialization task to background
        background_tasks.add_task(run_initialization)
        
        return {
            "message": "System initialization started in background",
            "status": "initializing"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to start initialization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start initialization: {str(e)}")

async def run_initialization():
    """Run the initialization script in background"""
    try:
        import subprocess
        result = subprocess.run([sys.executable, "chatbot_init.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ System initialization completed successfully")
        else:
            logger.error(f"❌ System initialization failed: {result.stderr}")
            
    except Exception as e:
        logger.error(f"❌ Background initialization failed: {str(e)}")

@app.get("/status", response_model=SystemStatus)
async def get_system_status(database=Depends(get_database)):
    """Get system status and initialization info"""
    try:
        # Get database tables
        tables_result = database.run("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        database_tables = [row[0] for row in tables_result] if tables_result else []
        
        # Check vector store collections
        vector_store_collections = []
        if vector_store:
            try:
                if hasattr(vector_store, 'client'):
                    # Weaviate
                    collections = vector_store.client.collections.list_all()
                    vector_store_collections = [col.name for col in collections]
                else:
                    # FAISS
                    vector_store_collections = ["FAISS"]
            except:
                pass
        
        return SystemStatus(
            initialized=len(database_tables) > 0,
            database_tables=database_tables,
            vector_store_collections=vector_store_collections,
            last_initialization=None  # Could be tracked in a file
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get system status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

@app.get("/search")
async def search_documents(
    query: str,
    limit: int = 5,
    vector_store_dep=Depends(get_vector_store)
):
    """Search documents in vector store"""
    try:
        if not vector_store_dep:
            raise HTTPException(status_code=503, detail="Vector store not available")
        
        # Perform similarity search
        docs = vector_store_dep.similarity_search(query, k=limit)
        
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": doc.metadata.get("source", "unknown")
            })
        
        return {
            "query": query,
            "results": results,
            "total_found": len(results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"❌ Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# Startup and utility functions
def check_environment():
    """Check if required environment files exist"""
    from pathlib import Path
    env_file = Path("chatbot.env")
    if not env_file.exists():
        logger.error("❌ chatbot.env file not found!")
        logger.info("💡 Please create chatbot.env file with required configuration")
        logger.info("📝 You can copy from chatbot.env.template")
        return False
    
    logger.info("✅ Environment file found")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import sqlalchemy
        import langchain
        logger.info("✅ Required dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.info("💡 Please install dependencies: pip install -r requirements_chatbot.txt")
        return False

def start_api_server():
    """Start the API server with proper configuration"""
    try:
        # Get configuration
        host = os.getenv("API_HOST", "0.0.0.0")
        port = int(os.getenv("API_PORT", "8000"))
        reload = os.getenv("API_RELOAD", "false").lower() == "true"
        
        logger.info(f"🚀 Starting Chatbot API Server...")
        logger.info(f"🌐 Server will be available at: http://{host}:{port}")
        logger.info(f"📚 API Documentation: http://{host}:{port}/docs")
        logger.info(f"🔍 Health Check: http://{host}:{port}/health")
        logger.info("=" * 60)
        
        # Start the server
        import uvicorn
        uvicorn.run(
            "chatbot_api:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start server: {str(e)}")
        sys.exit(1)

def main():
    """Main function for startup"""
    print("🤖 Strategy Agent Chatbot API")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Start server
    start_api_server()

if __name__ == "__main__":
    main()
