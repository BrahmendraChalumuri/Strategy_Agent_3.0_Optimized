#!/usr/bin/env python3
"""
Script to check the FAISS vector store data
"""

import os
import sys
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv('chatbot.env')

def check_faiss_vector_store():
    """Check the FAISS vector store data"""
    try:
        # Get OpenAI API key
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            print("❌ OPENAI_API_KEY not found in chatbot.env")
            return
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Try to load FAISS vector store
        try:
            vector_store = FAISS.load_local("cache/faiss_vector_store", embeddings)
            print("✅ FAISS vector store loaded successfully")
            
            # Get basic info
            print(f"📊 Vector store info:")
            print(f"   - Index size: {vector_store.index.ntotal}")
            print(f"   - Vector dimension: {vector_store.index.d}")
            
            # Test search
            print("\n🔍 Testing search functionality:")
            test_queries = [
                "customer analysis",
                "recommendations", 
                "C001",
                "products",
                "sales data"
            ]
            
            for query in test_queries:
                print(f"\n📝 Query: '{query}'")
                try:
                    docs = vector_store.similarity_search(query, k=2)
                    print(f"   Found {len(docs)} results:")
                    for i, doc in enumerate(docs, 1):
                        print(f"   {i}. Content preview: {doc.page_content[:100]}...")
                        print(f"      Source: {doc.metadata.get('source', 'unknown')}")
                except Exception as e:
                    print(f"   ❌ Search failed: {str(e)}")
            
        except FileNotFoundError:
            print("❌ FAISS vector store not found at cache/faiss_vector_store")
            print("💡 Run 'python chatbot_init.py' to initialize the vector store")
        except Exception as e:
            print(f"❌ Failed to load FAISS vector store: {str(e)}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    print("🔍 Checking FAISS Vector Store Data")
    print("=" * 50)
    check_faiss_vector_store()
