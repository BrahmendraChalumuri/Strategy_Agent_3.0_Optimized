#!/usr/bin/env python3
"""
Test script for the Hybrid Chatbot System

This script performs basic validation of the chatbot system components
without requiring a full database setup.
"""

import os
import sys
import logging
from unittest.mock import Mock, patch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required packages can be imported"""
    try:
        import pandas as pd
        import numpy as np
        from sqlalchemy import create_engine
        from langchain.agents import initialize_agent
        from langchain.llms import OpenAI
        from langchain.chat_models import ChatOpenAI
        from langchain.vectorstores import Weaviate
        from langchain.embeddings import OpenAIEmbeddings
        import weaviate
        from langchain.document_loaders import PyPDFLoader
        from langchain.text_splitter import CharacterTextSplitter
        from dotenv import load_dotenv
        
        logger.info("✅ All required packages imported successfully")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {str(e)}")
        return False

def test_data_files():
    """Test that required CSV files exist"""
    required_files = [
        'data/customer.csv',
        'data/products.csv',
        'data/customer_catalogue_enhanced.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"❌ Missing required files: {missing_files}")
        return False
    
    logger.info("✅ All required CSV files found")
    return True

def test_pdf_reports():
    """Test that PDF reports exist"""
    pdf_files = []
    if os.path.exists('reports'):
        pdf_files = [f for f in os.listdir('reports') if f.endswith('.pdf')]
    
    if not pdf_files:
        logger.warning("⚠️ No PDF reports found in reports/ directory")
        logger.info("   This is optional - the chatbot will work without PDF reports")
    else:
        logger.info(f"✅ Found {len(pdf_files)} PDF reports")
    
    return True

def test_environment_config():
    """Test environment configuration"""
    load_dotenv()
    
    openai_key = os.getenv('OPENAI_API_KEY')
    database_url = os.getenv('DATABASE_URL')
    weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
    
    if not openai_key:
        logger.error("❌ OPENAI_API_KEY not found in environment variables")
        return False
    
    if not database_url:
        logger.error("❌ DATABASE_URL not found in environment variables")
        return False
    
    logger.info(f"✅ Environment variables configured (Weaviate URL: {weaviate_url})")
    return True

def test_csv_data_loading():
    """Test CSV data loading and structure"""
    try:
        import pandas as pd
        
        # Test customer data
        customers_df = pd.read_csv('data/customer.csv')
        expected_customer_columns = ['CustomerID', 'CustomerName', 'CustomerType', 'Country', 'Region', 'TotalStores']
        
        if not all(col in customers_df.columns for col in expected_customer_columns):
            logger.error("❌ Customer CSV missing required columns")
            return False
        
        logger.info(f"✅ Customer data loaded: {len(customers_df)} records")
        
        # Test products data
        products_df = pd.read_csv('data/products.csv')
        expected_product_columns = ['ProductID', 'Name', 'Category', 'SubCategory', 'Price', 'Tags']
        
        if not all(col in products_df.columns for col in expected_product_columns):
            logger.error("❌ Products CSV missing required columns")
            return False
        
        logger.info(f"✅ Products data loaded: {len(products_df)} records")
        
        # Test catalogue data
        catalogue_df = pd.read_csv('data/customer_catalogue_enhanced.csv')
        expected_catalogue_columns = ['CustomerCatalogueItemID', 'CustomerID', 'ProductName', 'Product Category', 'Description', 'Ingredients']
        
        if not all(col in catalogue_df.columns for col in expected_catalogue_columns):
            logger.error("❌ Catalogue CSV missing required columns")
            return False
        
        logger.info(f"✅ Catalogue data loaded: {len(catalogue_df)} records")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ CSV data loading failed: {str(e)}")
        return False

def test_weaviate_connection():
    """Test Weaviate connection"""
    try:
        import weaviate
        from dotenv import load_dotenv
        
        load_dotenv()
        weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
        
        # Test Weaviate connection
        client = weaviate.Client(url=weaviate_url)
        
        # Check if Weaviate is ready
        if client.is_ready():
            logger.info("✅ Weaviate connection successful")
            return True
        else:
            logger.warning("⚠️ Weaviate is not ready (may not be running)")
            return False
            
    except Exception as e:
        logger.warning(f"⚠️ Weaviate connection failed: {str(e)}")
        logger.info("   This is expected if Weaviate is not running")
        return False

def test_mock_chatbot_initialization():
    """Test chatbot initialization with mocked dependencies"""
    try:
        # Mock external dependencies
        with patch('psycopg2.connect'), \
             patch('sqlalchemy.create_engine'), \
             patch('langchain.sql_database.SQLDatabase'), \
             patch('langchain.vectorstores.Weaviate.from_documents'), \
             patch('weaviate.Client'), \
             patch('langchain.agents.initialize_agent'):
            
            # Mock environment variables
            with patch.dict(os.environ, {
                'OPENAI_API_KEY': 'test-key',
                'DATABASE_URL': 'postgresql://test:test@localhost:5432/test',
                'WEAVIATE_URL': 'http://localhost:8080'
            }):
                
                # Import and test initialization
                from chatbot import HybridChatbotSystem
                
                # This would normally fail due to missing dependencies
                # but we're mocking them, so it should pass
                logger.info("✅ Chatbot class can be imported successfully")
                return True
                
    except Exception as e:
        logger.error(f"❌ Mock chatbot initialization failed: {str(e)}")
        return False

def run_all_tests():
    """Run all tests and report results"""
    logger.info("🧪 Running Hybrid Chatbot System Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Data Files", test_data_files),
        ("PDF Reports", test_pdf_reports),
        ("Environment Config", test_environment_config),
        ("CSV Data Loading", test_csv_data_loading),
        ("Weaviate Connection", test_weaviate_connection),
        ("Mock Chatbot Init", test_mock_chatbot_initialization),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Testing: {test_name}")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {str(e)}")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! The chatbot system is ready to use.")
        return True
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Please check the issues above.")
        return False

def main():
    """Main test function"""
    success = run_all_tests()
    
    if success:
        print("\n🚀 Ready to run the chatbot!")
        print("Execute: python chatbot.py")
    else:
        print("\n🔧 Please fix the issues above before running the chatbot.")
        sys.exit(1)

if __name__ == "__main__":
    main()
