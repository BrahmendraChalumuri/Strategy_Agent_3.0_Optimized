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
            ProductID INTEGER REFERENCES products(ProductID),
            Quantity INTEGER NOT NULL,
            SaleDate DATE DEFAULT CURRENT_DATE,
            TotalAmount DECIMAL(10,2)
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
            df_catalogue = pd.read_csv('data/customer_catalogue_enhanced.csv')
            
            # Clean column names
            df_catalogue.columns = df_catalogue.columns.str.replace('"', '').str.lower()
            
            # Clean and prepare data
            df_catalogue['calories'] = pd.to_numeric(df_catalogue['calories'], errors='coerce').fillna(0).astype(int)
            df_catalogue['quantityrequired'] = pd.to_numeric(df_catalogue['quantityrequired'], errors='coerce').fillna(1).astype(int)
            
            # Insert into database
            df_catalogue.to_sql('customer_catalogue', self.db_engine, if_exists='append', index=False)
            logger.info(f"✅ Loaded {len(df_catalogue)} customer catalogue items")
            
        except Exception as e:
            logger.error(f"❌ Failed to load customer catalogue: {str(e)}")
            raise
    
    def _generate_sample_sales_data(self):
        """Generate sample sales data"""
        try:
            # Get customer and product data
            customers_df = pd.read_sql('SELECT CustomerID FROM customers', self.db_engine)
            products_df = pd.read_sql('SELECT ProductID, Price FROM products', self.db_engine)
            
            # Generate sample sales data
            sales_data = []
            for _, customer in customers_df.iterrows():
                # Random number of sales per customer (1-10)
                num_sales = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).sample(1).iloc[0]
                
                for _ in range(num_sales):
                    # Random product
                    product = products_df.sample(1).iloc[0]
                    
                    # Random quantity (1-50)
                    quantity = pd.Series(range(1, 51)).sample(1).iloc[0]
                    
                    # Calculate total amount
                    total_amount = product['price'] * quantity
                    
                    sales_data.append({
                        'CustomerID': customer['CustomerID'],
                        'ProductID': product['ProductID'],
                        'Quantity': quantity,
                        'TotalAmount': total_amount
                    })
            
            # Create sales DataFrame
            sales_df = pd.DataFrame(sales_data)
            
            # Insert into database
            sales_df.to_sql('sales', self.db_engine, if_exists='append', index=False)
            logger.info(f"✅ Generated {len(sales_df)} sample sales records")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate sales data: {str(e)}")
            raise
    
    def _setup_vector_store(self):
        """Setup Weaviate vector store for PDF reports"""
        try:
            logger.info("📚 Setting up Weaviate vector store for PDF reports...")
            
            # Try to initialize Weaviate client (v4 API)
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
                logger.warning(f"⚠️ Weaviate connection failed: {str(e)}")
                logger.info("📝 Continuing without vector store - chatbot will work with database queries only")
                return
            
            # Initialize embeddings and create vector store
            try:
                embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
                
                # Load PDF reports
                pdf_files = glob.glob("reports/*.pdf")
                
                if not pdf_files:
                    logger.warning("⚠️ No PDF reports found in reports/ directory")
                    # Create empty vector store with dummy document
                    dummy_doc = Document(page_content="No PDF reports available", metadata={"source": "empty"})
                    try:
                        vector_store = Weaviate.from_documents(
                            [dummy_doc], 
                            embeddings, 
                            client=weaviate_client,
                            index_name="PDFReports"
                        )
                        logger.info("✅ Created empty vector store")
                    except Exception:
                        # Fallback to FAISS
                        from langchain_community.vectorstores import FAISS
                        vector_store = FAISS.from_documents([dummy_doc], embeddings)
                        logger.info("✅ Created empty FAISS vector store")
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
                    try:
                        vector_store = Weaviate.from_documents(
                            [dummy_doc], 
                            embeddings, 
                            client=weaviate_client,
                            index_name="PDFReports"
                        )
                    except Exception:
                        # Fallback to FAISS
                        from langchain_community.vectorstores import FAISS
                        vector_store = FAISS.from_documents([dummy_doc], embeddings)
                    return
                
                # Split documents
                text_splitter = CharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                split_docs = text_splitter.split_documents(documents)
                
                logger.info(f"📝 Split into {len(split_docs)} chunks")
                
                # Create Weaviate vector store using v4 compatible method
                try:
                    vector_store = Weaviate.from_documents(
                        split_docs, 
                        embeddings, 
                        client=weaviate_client,
                        index_name="PDFReports",
                        text_key="content"
                    )
                    logger.info("✅ Weaviate vector store created successfully")
                except Exception as weaviate_error:
                    logger.warning(f"⚠️ Weaviate v4 compatibility issue: {str(weaviate_error)}")
                    logger.info("📝 Using FAISS fallback...")
                    # Fallback: Create a simple in-memory vector store
                    from langchain_community.vectorstores import FAISS
                    vector_store = FAISS.from_documents(split_docs, embeddings)
                    logger.info("✅ FAISS vector store created successfully")
                
                weaviate_client.close()
                
            except Exception as e:
                logger.warning(f"⚠️ Vector store creation failed: {str(e)}")
                logger.info("📝 Continuing without vector store - chatbot will work with database queries only")
            
        except Exception as e:
            logger.warning(f"⚠️ Weaviate vector store setup failed: {str(e)}")
            logger.info("📝 Continuing without vector store - chatbot will work with database queries only")
    
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
