#!/usr/bin/env python3
"""
Hybrid Chatbot System for Cross-Sell and Up-Sell Recommendations

This system combines:
1. PostgreSQL database for structured data queries
2. Vector database (FAISS) for PDF report knowledge base
3. LangChain agents for natural language processing
4. Interactive chatbot interface

Author: AI Assistant
Date: 2025
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Database imports
import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# LangChain imports
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool

# Vector database and embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain.schema import Document
import weaviate
from weaviate.classes.config import Configure

# Environment and utilities
from dotenv import load_dotenv
import glob
import json

# Load environment variables
load_dotenv('chatbot.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HybridChatbotSystem:
    """
    Hybrid chatbot system that combines structured database queries 
    with unstructured PDF report knowledge base using LangChain agents.
    """
    
    def __init__(self):
        """Initialize the hybrid chatbot system"""
        self.db_engine = None
        self.db_chain = None
        self.vector_store = None
        self.agent = None
        self.memory = None
        
        # Configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.database_url = os.getenv('DATABASE_URL')
        self.weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
        self.weaviate_api_key = os.getenv('WEAVIATE_API_KEY')  # Optional for cloud Weaviate
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        if not self.database_url:
            raise ValueError("DATABASE_URL not found in environment variables")
        
        logger.info("🚀 Initializing Hybrid Chatbot System...")
        
        # Initialize components
        self._setup_database()
        self._setup_vector_store()
        self._setup_agent()
        
        logger.info("✅ Hybrid Chatbot System initialized successfully!")
    
    def _setup_database(self):
        """Setup PostgreSQL database connection and create tables"""
        try:
            logger.info("📊 Setting up PostgreSQL database...")
            
            # Create SQLAlchemy engine
            self.db_engine = create_engine(self.database_url)
            
            # Create database tables
            self._create_database_schema()
            
            # Ingest CSV data
            self._ingest_csv_data()
            
            # Setup LangChain SQL database
            self._setup_sql_chain()
            
            logger.info("✅ Database setup completed")
            
        except Exception as e:
            logger.error(f"❌ Database setup failed: {str(e)}")
            raise
    
    def _create_database_schema(self):
        """Create database tables with proper schemas"""
        schema_sql = """
        -- Drop existing tables if they exist (in correct order due to foreign keys)
        DROP TABLE IF EXISTS sales CASCADE;
        DROP TABLE IF EXISTS customer_catalogue CASCADE;
        DROP TABLE IF EXISTS customers CASCADE;
        DROP TABLE IF EXISTS products CASCADE;
        
        -- Create customers table
        CREATE TABLE customers (
            CustomerID VARCHAR(10) PRIMARY KEY,
            CustomerName VARCHAR(100) NOT NULL,
            CustomerType VARCHAR(50),
            Country VARCHAR(50),
            Region VARCHAR(50),
            TotalStores INTEGER
        );
        
        -- Create products table
        CREATE TABLE products (
            ProductID INTEGER PRIMARY KEY,
            Name VARCHAR(200) NOT NULL,
            Category VARCHAR(50),
            SubCategory VARCHAR(50),
            Price DECIMAL(10,2),
            Tags TEXT
        );
        
        -- Create customer_catalogue table
        CREATE TABLE customer_catalogue (
            CatalogueID VARCHAR(20) PRIMARY KEY,
            CustomerID VARCHAR(10) REFERENCES customers(CustomerID),
            ProductName VARCHAR(200) NOT NULL,
            Category VARCHAR(50),
            Description TEXT,
            Ingredients TEXT,
            Calories INTEGER,
            QuantityRequired INTEGER DEFAULT 1
        );
        
        -- Create sales table
        CREATE TABLE sales (
            SaleID SERIAL PRIMARY KEY,
            CustomerID VARCHAR(10) REFERENCES customers(CustomerID),
            StoreID VARCHAR(20),
            ProductID INTEGER REFERENCES products(ProductID),
            PlantID VARCHAR(20),
            Quantity INTEGER,
            UnitPrice DECIMAL(10,2),
            TotalAmount DECIMAL(12,2),
            SaleDate DATE,
            DeliveryDate DATE,
            Status VARCHAR(20)
        );
        
        -- Create indexes for better performance
        CREATE INDEX idx_sales_customer_date ON sales(CustomerID, SaleDate);
        CREATE INDEX idx_sales_product ON sales(ProductID);
        CREATE INDEX idx_catalogue_customer ON customer_catalogue(CustomerID);
        """
        
        try:
            with self.db_engine.connect() as conn:
                conn.execute(text(schema_sql))
                conn.commit()
            logger.info("✅ Database schema created successfully")
        except SQLAlchemyError as e:
            logger.error(f"❌ Schema creation failed: {str(e)}")
            raise
    
    def _ingest_csv_data(self):
        """Ingest CSV data into PostgreSQL tables"""
        logger.info("📥 Ingesting CSV data into PostgreSQL...")
        
        try:
            # Load and process customer data
            self._load_customers()
            
            # Load and process products data
            self._load_products()
            
            # Load and process customer catalogue data
            self._load_customer_catalogue()
            
            # Generate and load sales data (since we don't have sales CSV)
            self._generate_sample_sales_data()
            
            logger.info("✅ CSV data ingestion completed")
            
        except Exception as e:
            logger.error(f"❌ CSV data ingestion failed: {str(e)}")
            raise
    
    def _load_customers(self):
        """Load customer data from CSV"""
        try:
            df = pd.read_csv('data/customer.csv')
            
            # Clean column names (remove all quotes if present and convert to lowercase)
            df.columns = df.columns.str.replace('"', '').str.lower()
            
            # Clean and prepare data
            df['totalstores'] = pd.to_numeric(df['totalstores'], errors='coerce').fillna(0).astype(int)
            
            # Insert into database
            df.to_sql('customers', self.db_engine, if_exists='append', index=False)
            logger.info(f"✅ Loaded {len(df)} customers")
            
        except Exception as e:
            logger.error(f"❌ Failed to load customers: {str(e)}")
            raise
    
    def _load_products(self):
        """Load products data from CSV"""
        try:
            df = pd.read_csv('data/products.csv')
            
            # Clean column names (remove all quotes if present and convert to lowercase)
            df.columns = df.columns.str.replace('"', '').str.lower()
            
            # Clean and prepare data
            df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0)
            df['productid'] = pd.to_numeric(df['productid'], errors='coerce').fillna(0).astype(int)
            
            # Insert into database
            df.to_sql('products', self.db_engine, if_exists='append', index=False)
            logger.info(f"✅ Loaded {len(df)} products")
            
        except Exception as e:
            logger.error(f"❌ Failed to load products: {str(e)}")
            raise
    
    def _load_customer_catalogue(self):
        """Load customer catalogue data from CSV"""
        try:
            df = pd.read_csv('data/customer_catalogue_enhanced.csv')
            
            # Clean column names (remove all quotes if present and convert to lowercase)
            df.columns = df.columns.str.replace('"', '').str.lower()
            
            # Add additional columns for our schema
            df['calories'] = np.random.randint(100, 500, len(df))  # Mock calories data
            df['quantityrequired'] = df.get('quantityrequired', 1)
            
            # Rename columns to match our schema
            df = df.rename(columns={
                'customercatalogueitemid': 'catalogueid'
            })
            
            # Select and reorder columns
            catalogue_columns = [
                'catalogueid', 'customerid', 'productname', 'product category',
                'description', 'ingredients', 'calories', 'quantityrequired'
            ]
            
            df_catalogue = df[catalogue_columns].copy()
            df_catalogue = df_catalogue.rename(columns={
                'product category': 'category'
            })
            
            # Insert into database
            df_catalogue.to_sql('customer_catalogue', self.db_engine, if_exists='append', index=False)
            logger.info(f"✅ Loaded {len(df_catalogue)} catalogue items")
            
        except Exception as e:
            logger.error(f"❌ Failed to load customer catalogue: {str(e)}")
            raise
    
    def _generate_sample_sales_data(self):
        """Generate sample sales data since we don't have sales CSV"""
        try:
            logger.info("📊 Generating sample sales data...")
            
            # Get customer and product data
            customers_df = pd.read_sql("SELECT CustomerID FROM customers", self.db_engine)
            products_df = pd.read_sql("SELECT ProductID, Price FROM products", self.db_engine)
            
            # Generate sample sales data
            sales_data = []
            
            for customer_id in customers_df['CustomerID']:
                # Generate 10-50 sales per customer
                num_sales = np.random.randint(10, 51)
                
                for _ in range(num_sales):
                    product = products_df.sample(1).iloc[0]
                    
                    sale_date = pd.date_range(
                        start='2023-01-01',
                        end='2024-12-31',
                        periods=1000
                    )[np.random.randint(0, 1000)]
                    
                    quantity = np.random.randint(1, 100)
                    unit_price = float(product['Price'])
                    total_amount = quantity * unit_price
                    
                    sales_data.append({
                        'CustomerID': customer_id,
                        'StoreID': f'ST{np.random.randint(1, 100):03d}',
                        'ProductID': int(product['ProductID']),
                        'PlantID': f'PL{np.random.randint(1, 10):02d}',
                        'Quantity': quantity,
                        'UnitPrice': unit_price,
                        'TotalAmount': total_amount,
                        'SaleDate': sale_date,
                        'DeliveryDate': sale_date + pd.Timedelta(days=np.random.randint(1, 7)),
                        'Status': np.random.choice(['Completed', 'Pending', 'Shipped'])
                    })
            
            # Create DataFrame and insert
            sales_df = pd.DataFrame(sales_data)
            sales_df.to_sql('sales', self.db_engine, if_exists='append', index=False)
            
            logger.info(f"✅ Generated {len(sales_df)} sample sales records")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate sales data: {str(e)}")
            raise
    
    def _setup_sql_chain(self):
        """Setup LangChain SQL database chain"""
        try:
            logger.info("🔗 Setting up SQL database chain...")
            
            # Create SQL database wrapper
            db = SQLDatabase(self.db_engine)
            
            # Initialize LLM
            llm = ChatOpenAI(
                temperature=0,
                model_name="gpt-3.5-turbo",
                openai_api_key=self.openai_api_key
            )
            
            # Create SQL toolkit
            toolkit = SQLDatabaseToolkit(db=db, llm=llm)
            
            # Create database chain
            self.db_chain = toolkit.get_tools()
            
            logger.info("✅ SQL database chain setup completed")
            
        except Exception as e:
            logger.error(f"❌ SQL chain setup failed: {str(e)}")
            raise
    
    def _setup_vector_store(self):
        """Setup Weaviate vector store for PDF reports"""
        try:
            logger.info("📚 Setting up Weaviate vector store for PDF reports...")
            
            # Initialize Weaviate client
            client_kwargs = {
                "url": self.weaviate_url,
            }
            
            # Add API key if provided (for Weaviate Cloud Services)
            if self.weaviate_api_key:
                client_kwargs["auth_client_secret"] = weaviate.AuthApiKey(api_key=self.weaviate_api_key)
            
            weaviate_client = weaviate.Client(**client_kwargs)
            
            # Test connection
            try:
                weaviate_client.is_ready()
                logger.info("✅ Connected to Weaviate successfully")
            except Exception as e:
                logger.error(f"❌ Failed to connect to Weaviate: {str(e)}")
                raise
            
            # Initialize embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            
            # Load PDF reports
            pdf_files = glob.glob("reports/*.pdf")
            
            if not pdf_files:
                logger.warning("⚠️ No PDF reports found in reports/ directory")
                # Create empty vector store with dummy document
                dummy_doc = Document(page_content="No PDF reports available", metadata={"source": "empty"})
                self.vector_store = Weaviate.from_documents(
                    [dummy_doc], 
                    embeddings, 
                    client=weaviate_client,
                    index_name="PDFReports"
                )
                return
            
            logger.info(f"📄 Found {len(pdf_files)} PDF reports")
            
            # Load and split documents
            documents = []
            for pdf_file in pdf_files:
                try:
                    loader = PyPDFLoader(pdf_file)
                    pages = loader.load()
                    
                    # Add metadata
                    for page in pages:
                        page.metadata['source'] = pdf_file
                        page.metadata['filename'] = os.path.basename(pdf_file)
                        page.metadata['file_type'] = 'pdf_report'
                        page.metadata['ingestion_date'] = datetime.now().isoformat()
                    
                    documents.extend(pages)
                    logger.info(f"✅ Loaded {len(pages)} pages from {os.path.basename(pdf_file)}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {pdf_file}: {str(e)}")
                    continue
            
            if not documents:
                logger.warning("⚠️ No documents loaded, creating empty vector store")
                dummy_doc = Document(page_content="No PDF reports available", metadata={"source": "empty"})
                self.vector_store = Weaviate.from_documents(
                    [dummy_doc], 
                    embeddings, 
                    client=weaviate_client,
                    index_name="PDFReports"
                )
                return
            
            # Split documents
            text_splitter = CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            split_docs = text_splitter.split_documents(documents)
            
            logger.info(f"📝 Split into {len(split_docs)} chunks")
            
            # Create Weaviate vector store
            self.vector_store = Weaviate.from_documents(
                split_docs, 
                embeddings, 
                client=weaviate_client,
                index_name="PDFReports",
                text_key="content"
            )
            
            logger.info("✅ Weaviate vector store setup completed")
            
        except Exception as e:
            logger.error(f"❌ Weaviate vector store setup failed: {str(e)}")
            raise
    
    def _setup_agent(self):
        """Setup hybrid agent with database and PDF tools"""
        try:
            logger.info("🤖 Setting up hybrid agent...")
            
            # Initialize LLM
            llm = ChatOpenAI(
                temperature=0,
                model_name="gpt-3.5-turbo",
                openai_api_key=self.openai_api_key
            )
            
            # Create PDF retriever tool
            pdf_retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 4}
            )
            
            # Create custom PDF tool
            class PDFReportTool(BaseTool):
                name = "pdf_reports"
                description = """
                Use this tool to search through PDF reports for cross-sell and up-sell recommendations.
                Input should be a search query about recommendations, customer analysis, or report content.
                Examples:
                - "cross-sell recommendations for Starbucks"
                - "customer classification analysis"
                - "up-sell opportunities"
                """
                
                def _run(self, query: str) -> str:
                    try:
                        docs = pdf_retriever.get_relevant_documents(query)
                        if not docs:
                            return "No relevant information found in PDF reports."
                        
                        result = "PDF Report Information:\n\n"
                        for i, doc in enumerate(docs, 1):
                            result += f"Source {i}: {doc.metadata.get('filename', 'Unknown')}\n"
                            result += f"Content: {doc.page_content[:500]}...\n\n"
                        
                        return result
                        
                    except Exception as e:
                        return f"Error searching PDF reports: {str(e)}"
                
                async def _arun(self, query: str) -> str:
                    return self._run(query)
            
            # Create all tools
            tools = self.db_chain + [PDFReportTool()]
            
            # Setup memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Initialize agent
            self.agent = initialize_agent(
                tools=tools,
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5
            )
            
            logger.info("✅ Hybrid agent setup completed")
            
        except Exception as e:
            logger.error(f"❌ Agent setup failed: {str(e)}")
            raise
    
    def query(self, question: str) -> str:
        """Query the hybrid chatbot system"""
        try:
            logger.info(f"❓ Processing query: {question}")
            
            # Run the agent
            response = self.agent.run(question)
            
            logger.info("✅ Query processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"❌ Query processing failed: {str(e)}")
            return f"Sorry, I encountered an error processing your query: {str(e)}"
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            stats = {}
            
            with self.db_engine.connect() as conn:
                # Get table counts
                tables = ['customers', 'products', 'customer_catalogue', 'sales']
                for table in tables:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.scalar()
                    stats[table] = count
                
                # Get recent sales
                result = conn.execute(text("""
                    SELECT COUNT(*) as recent_sales 
                    FROM sales 
                    WHERE SaleDate >= CURRENT_DATE - INTERVAL '30 days'
                """))
                stats['recent_sales_30_days'] = result.scalar()
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get database stats: {str(e)}")
            return {}
    
    def interactive_chat(self):
        """Start interactive chatbot session"""
        print("\n" + "="*80)
        print("🤖 HYBRID CHATBOT SYSTEM FOR CROSS-SELL & UP-SELL RECOMMENDATIONS")
        print("="*80)
        print("Ask questions about:")
        print("• Customer data and sales (e.g., 'Which customer has the most stores?')")
        print("• Product information (e.g., 'Show me all biscuit products')")
        print("• Sales analysis (e.g., 'What were the top products in January 2024?')")
        print("• PDF reports (e.g., 'Show me cross-sell recommendations for Starbucks')")
        print("• Combined queries (e.g., 'Compare Starbucks sales with their recommendations')")
        print("\nType 'quit', 'exit', or 'bye' to end the session")
        print("Type 'stats' to see database statistics")
        print("="*80)
        
        # Show initial stats
        stats = self.get_database_stats()
        if stats:
            print(f"\n📊 Database Statistics:")
            for table, count in stats.items():
                print(f"   {table}: {count:,} records")
        
        while True:
            try:
                # Get user input
                user_input = input("\n💬 You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n👋 Thank you for using the Hybrid Chatbot System!")
                    break
                
                # Check for stats command
                if user_input.lower() == 'stats':
                    stats = self.get_database_stats()
                    if stats:
                        print(f"\n📊 Current Database Statistics:")
                        for table, count in stats.items():
                            print(f"   {table}: {count:,} records")
                    continue
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Process query
                print("\n🤔 Processing your query...")
                response = self.query(user_input)
                
                # Display response
                print(f"\n🤖 Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                logger.error(f"Interactive chat error: {str(e)}")

def main():
    """Main function to run the hybrid chatbot system"""
    try:
        # Check if required files exist
        required_files = [
            'data/customer.csv',
            'data/products.csv', 
            'data/customer_catalogue_enhanced.csv'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            print(f"❌ Missing required files: {missing_files}")
            print("Please ensure all CSV files are in the data/ directory")
            return
        
        # Initialize chatbot system
        chatbot = HybridChatbotSystem()
        
        # Start interactive chat
        chatbot.interactive_chat()
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {str(e)}")
        print(f"❌ Error: {str(e)}")
        print("\nPlease check:")
        print("1. Environment variables (OPENAI_API_KEY, DATABASE_URL)")
        print("2. Required CSV files in data/ directory")
        print("3. PostgreSQL database is running and accessible")

if __name__ == "__main__":
    main()
