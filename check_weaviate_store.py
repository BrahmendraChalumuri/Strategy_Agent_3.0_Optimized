#!/usr/bin/env python3
"""
Script to check the Weaviate vector store data
"""

import os
import sys
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Weaviate
import weaviate

# Load environment variables
load_dotenv('chatbot.env')

def check_weaviate_vector_store():
    """Check the Weaviate vector store data"""
    try:
        # Get configuration
        weaviate_url = os.getenv('WEAVIATE_URL')
        weaviate_api_key = os.getenv('WEAVIATE_API_KEY')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not openai_api_key:
            print("❌ OPENAI_API_KEY not found in chatbot.env")
            return
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Try to connect to Weaviate
        try:
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
            
            print("✅ Connected to Weaviate successfully")
            
            # Test connection
            meta = weaviate_client.get_meta()
            print(f"📊 Weaviate version: {meta.get('version', 'unknown')}")
            
            # Check if PDFReports collection exists
            try:
                collection = weaviate_client.collections.get("PDFReports")
                total_count = collection.aggregate.over_all(total_count=True).total_count
                print(f"📊 PDFReports collection found with {total_count} documents")
                
                if total_count > 0:
                    # Test search functionality using native Weaviate v4 API
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
                            response = collection.query.near_text(
                                query=query,
                                limit=2,
                                return_metadata=weaviate.classes.MetadataQuery(distance=True)
                            )
                            
                            print(f"   Found {len(response.objects)} results:")
                            for i, obj in enumerate(response.objects, 1):
                                content = obj.properties.get("content", "")
                                source = obj.properties.get("source", "unknown")
                                distance = obj.metadata.distance if obj.metadata else "unknown"
                                print(f"   {i}. Content preview: {content[:100]}...")
                                print(f"      Source: {source}")
                                print(f"      Distance: {distance}")
                        except Exception as e:
                            print(f"   ❌ Search failed: {str(e)}")
                else:
                    print("⚠️ PDFReports collection is empty")
                    
            except Exception as e:
                print(f"❌ PDFReports collection not found: {str(e)}")
                print("💡 Run 'python chatbot_init.py' to initialize the vector store")
            
            weaviate_client.close()
            
        except Exception as e:
            print(f"❌ Failed to connect to Weaviate: {str(e)}")
            print("💡 Make sure Weaviate is running on localhost:8080 or configure WEAVIATE_URL and WEAVIATE_API_KEY")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    print("🔍 Checking Weaviate Vector Store Data")
    print("=" * 50)
    check_weaviate_vector_store()
