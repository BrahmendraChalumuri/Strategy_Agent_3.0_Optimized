#!/usr/bin/env python3
"""
Chatbot Initialization Script
============================

This script handles the one-time setup of:
1. Database tables and data
2. Vector store and embeddings
3. Weaviate configuration

Run this script once to initialize the system, then use chatbot.py for conversations.

Usage:
    python chatbot_init.py
"""

import os
import sys
import pandas as pd
import glob
import logging
from datetime import datetime
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import weaviate
from langchain_community.vectorstores import Weaviate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatbotInitializer:
    """Handles one-time initialization of database and vector store"""
    
    def __init__(self):
        """Initialize the chatbot initializer"""
        logger.info("🚀 Starting Chatbot Initialization...")
        
        # Load environment variables
        load_dotenv('chatbot.env')
        
        # Database configuration
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL not found in chatbot.env")
        
        # OpenAI configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in chatbot.env")
        
        # Weaviate configuration
        self.weaviate_url = os.getenv('WEAVIATE_URL')
        self.weaviate_api_key = os.getenv('WEAVIATE_API_KEY')
        
        # Initialize database engine
        self.db_engine = create_engine(self.database_url)
        
        logger.info("✅ Configuration loaded successfully")
    
    def initialize_database(self):
        """Initialize database tables and data"""
        try:
            logger.info("📊 Initializing PostgreSQL database...")
            
            # Check if database is already initialized
            if self._is_database_initialized():
                logger.info("✅ Database already initialized, skipping...")
                return True
            
            # Create database tables
            self._create_database_schema()
            
            # Ingest CSV data
            self._ingest_csv_data()
            
            logger.info("✅ Database initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {str(e)}")
            return False
    
    def initialize_vector_store(self):
        """Initialize Weaviate vector store"""
        try:
            logger.info("📚 Initializing Weaviate vector store...")
            
            # Check if vector store is already initialized
            if self._is_vector_store_initialized():
                logger.info("✅ Vector store already initialized, skipping...")
                return True
            
            # Setup vector store
            self._setup_vector_store()
            
            logger.info("✅ Vector store initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Vector store initialization failed: {str(e)}")
            return False
    
    def _is_database_initialized(self):
        """Check if database is already initialized"""
        try:
            with self.db_engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('customers', 'products', 'customer_catalogue', 'sales')
                """))
                existing_tables = [row[0] for row in result]
                
                if len(existing_tables) == 4:
                    # Check if tables have data
                    for table in existing_tables:
                        count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        count = count_result.scalar()
                        if count == 0:
                            logger.info(f"📝 Table {table} exists but is empty, will reinitialize")
                            return False
                    
                    logger.info("✅ Database tables exist and contain data")
                    return True
                else:
                    logger.info(f"📝 Found {len(existing_tables)}/4 tables, will initialize")
                    return False
                    
        except Exception as e:
            logger.warning(f"⚠️ Error checking database status: {str(e)}")
            return False
    
    def _is_vector_store_initialized(self):
        """Check if vector store is already initialized"""
        try:
            if not self.weaviate_url:
                logger.info("📝 No Weaviate URL configured, skipping vector store check")
                return True
            
            # Try to connect to Weaviate
            if self.weaviate_api_key:
                weaviate_client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=self.weaviate_url,
                    auth_credentials=weaviate.AuthApiKey(api_key=self.weaviate_api_key)
                )
            else:
                weaviate_client = weaviate.connect_to_local(
                    host="localhost",
                    port=8080
                )
            
            # Check if PDFReports collection exists and has data
            try:
                collection = weaviate_client.collections.get("PDFReports")
                if collection.aggregate().total_count > 0:
                    logger.info("✅ Vector store exists and contains data")
                    weaviate_client.close()
                    return True
                else:
                    logger.info("📝 Vector store exists but is empty, will reinitialize")
                    weaviate_client.close()
                    return False
            except:
                logger.info("📝 Vector store collection doesn't exist, will create")
                weaviate_client.close()
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Error checking vector store status: {str(e)}")
            return False
    
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
            customerid VARCHAR(10) PRIMARY KEY,
            customername VARCHAR(100) NOT NULL,
            customertype VARCHAR(50),
            country VARCHAR(50),
            region VARCHAR(50),
            totalstores INTEGER
        );
        
        -- Create products table
        CREATE TABLE products (
            productid INTEGER PRIMARY KEY,
            name VARCHAR(200) NOT NULL,
            category VARCHAR(50),
            subcategory VARCHAR(50),
            price DECIMAL(10,2),
            tags TEXT
        );
        
        -- Create customer_catalogue table
        CREATE TABLE customer_catalogue (
            customercatalogueitemid VARCHAR(20) PRIMARY KEY,
            customerid VARCHAR(10) REFERENCES customers(customerid),
            productname VARCHAR(200) NOT NULL,
            product_category VARCHAR(50),
            description TEXT,
            ingredients TEXT,
            quantityrequired INTEGER DEFAULT 1
        );
        
        -- Create sales table
        CREATE TABLE sales (
            saleid VARCHAR(10) PRIMARY KEY,
            customerid VARCHAR(10) REFERENCES customers(customerid),
            storeid VARCHAR(10),
            productid INTEGER REFERENCES products(productid),
            plantid INTEGER,
            quantity INTEGER NOT NULL,
            unitprice DECIMAL(10,2),
            totalamount DECIMAL(10,2),
            saledate DATE,
            deliverydate DATE,
            status VARCHAR(50)
        );
        """
        
        with self.db_engine.connect() as conn:
            conn.execute(text(schema_sql))
            conn.commit()
        
        logger.info("✅ Database schema created successfully")
    
    def _ingest_csv_data(self):
        """Load and process CSV data into database"""
        try:
            logger.info("📄 Starting CSV data ingestion...")
            
            # Load and process customer data
            self._load_customers()
            
            # Load and process products data
            self._load_products()
            
            # Load and process customer catalogue data
            self._load_customer_catalogue()
            
            # Load actual sales data
            self._load_sales_data()
            
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
            df_catalogue = pd.read_csv('data/customer_catalogue_enhanced.csv')
            
            # Clean column names (remove all quotes if present and convert to lowercase)
            df_catalogue.columns = df_catalogue.columns.str.replace('"', '').str.lower()
            
            # Fix the "product category" column name to match database schema
            df_catalogue = df_catalogue.rename(columns={'product category': 'product_category'})
            
            # Clean and prepare data
            df_catalogue['quantityrequired'] = pd.to_numeric(df_catalogue['quantityrequired'], errors='coerce').fillna(1).astype(int)
            
            # Insert into database
            df_catalogue.to_sql('customer_catalogue', self.db_engine, if_exists='append', index=False)
            logger.info(f"✅ Loaded {len(df_catalogue)} customer catalogue items")
            
        except Exception as e:
            logger.error(f"❌ Failed to load customer catalogue: {str(e)}")
            raise
    
    def _load_sales_data(self):
        """Load actual sales data from CSV"""
        try:
            df_sales = pd.read_csv('data/sales_enhanced.csv')
            
            # Clean column names (remove all quotes if present and convert to lowercase)
            df_sales.columns = df_sales.columns.str.replace('"', '').str.lower()
            
            # Clean and prepare data
            df_sales['quantity'] = pd.to_numeric(df_sales['quantity'], errors='coerce').fillna(0).astype(int)
            df_sales['unitprice'] = pd.to_numeric(df_sales['unitprice'], errors='coerce').fillna(0.0)
            df_sales['totalamount'] = pd.to_numeric(df_sales['totalamount'], errors='coerce').fillna(0.0)
            df_sales['plantid'] = pd.to_numeric(df_sales['plantid'], errors='coerce').fillna(0).astype(int)
            df_sales['productid'] = pd.to_numeric(df_sales['productid'], errors='coerce').fillna(0).astype(int)
            
            # Convert date columns
            df_sales['saledate'] = pd.to_datetime(df_sales['saledate'], format='%d-%m-%Y', errors='coerce')
            df_sales['deliverydate'] = pd.to_datetime(df_sales['deliverydate'], format='%d-%m-%Y', errors='coerce')
            
            # Insert into database
            df_sales.to_sql('sales', self.db_engine, if_exists='append', index=False)
            logger.info(f"✅ Loaded {len(df_sales)} sales records")
            
        except Exception as e:
            logger.error(f"❌ Failed to load sales data: {str(e)}")
            raise
    
    def _setup_vector_store(self):
        """Setup Weaviate vector store for PDF reports"""
        try:
            logger.info("📚 Setting up Weaviate vector store for PDF reports...")
            
            # Initialize Weaviate client (v4 API)
            try:
                if self.weaviate_api_key:
                    # For Weaviate Cloud Services
                    weaviate_client = weaviate.connect_to_weaviate_cloud(
                        cluster_url=self.weaviate_url,
                        auth_credentials=weaviate.AuthApiKey(api_key=self.weaviate_api_key)
                    )
                else:
                    # For local Weaviate instance
                    weaviate_client = weaviate.connect_to_local(
                        host="localhost",
                        port=8080
                    )
                
                # Test connection
                weaviate_client.get_meta()
                logger.info("✅ Connected to Weaviate successfully")
                
            except Exception as e:
                logger.error(f"❌ Weaviate connection failed: {str(e)}")
                raise Exception(f"Failed to connect to Weaviate: {str(e)}")
            
            # Initialize embeddings and create vector store
            try:
                embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
                
                # Load PDF reports
                pdf_files = glob.glob("reports/*.pdf")
                
                if not pdf_files:
                    logger.warning("⚠️ No PDF reports found in reports/ directory")
                    # Create empty vector store with dummy document
                    dummy_doc = Document(page_content="No PDF reports available", metadata={"source": "empty"})
                    vector_store = Weaviate.from_documents(
                        [dummy_doc], 
                        embeddings, 
                        client=weaviate_client,
                        index_name="PDFReports"
                    )
                    logger.info("✅ Created empty Weaviate vector store")
                    weaviate_client.close()
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
                    vector_store = Weaviate.from_documents(
                        [dummy_doc], 
                        embeddings, 
                        client=weaviate_client,
                        index_name="PDFReports"
                    )
                    weaviate_client.close()
                    return
                
                # Split documents
                text_splitter = CharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                split_docs = text_splitter.split_documents(documents)
                
                logger.info(f"📝 Split into {len(split_docs)} chunks")
                
                # Create Weaviate vector store using v4 compatible method
                # First, check if collection exists and delete it if it does
                try:
                    existing_collection = weaviate_client.collections.get("PDFReports")
                    if existing_collection:
                        logger.info("📝 Deleting existing PDFReports collection...")
                        weaviate_client.collections.delete("PDFReports")
                        logger.info("✅ Deleted existing collection")
                except:
                    # Collection doesn't exist, which is fine
                    pass
                
                # Create new vector store
                vector_store = Weaviate.from_documents(
                    split_docs, 
                    embeddings, 
                    client=weaviate_client,
                    index_name="PDFReports",
                    text_key="content"
                )
                logger.info("✅ Weaviate vector store created successfully")
                
                weaviate_client.close()
                
            except Exception as e:
                logger.error(f"❌ Vector store creation failed: {str(e)}")
                weaviate_client.close()
                raise Exception(f"Failed to create Weaviate vector store: {str(e)}")
            
        except Exception as e:
            logger.error(f"❌ Weaviate vector store setup failed: {str(e)}")
            raise Exception(f"Weaviate vector store setup failed: {str(e)}")
    
    def run_initialization(self):
        """Run the complete initialization process"""
        try:
            logger.info("🚀 Starting complete initialization process...")
            
            # Initialize database
            db_success = self.initialize_database()
            
            # Initialize vector store
            vector_success = self.initialize_vector_store()
            
            if db_success and vector_success:
                logger.info("🎉 Initialization completed successfully!")
                logger.info("💡 You can now run 'python chatbot.py' to start chatting")
                return True
            else:
                logger.error("❌ Initialization failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Initialization process failed: {str(e)}")
            return False

def main():
    """Main function to run initialization"""
    try:
        print("🤖 Chatbot Initialization Script")
        print("=" * 50)
        print("This script will initialize:")
        print("• Database tables and data")
        print("• Vector store and embeddings")
        print("• Weaviate configuration")
        print("=" * 50)
        
        # Create initializer
        initializer = ChatbotInitializer()
        
        # Run initialization
        success = initializer.run_initialization()
        
        if success:
            print("\n🎉 Initialization completed successfully!")
            print("💡 You can now run 'python chatbot.py' to start chatting")
            sys.exit(0)
        else:
            print("\n❌ Initialization failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Initialization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Initialization failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
