#!/usr/bin/env python3
"""
Chatbot Conversation System
==========================

This script handles conversations with the chatbot after initialization.
It assumes the database and vector store are already set up by chatbot_init.py.

Usage:
    python chatbot_conversation.py
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

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
from langchain_community.vectorstores import Weaviate
import weaviate

# Environment and utilities
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatbotConversation:
    """Handles chatbot conversations after initialization"""
    
    def __init__(self):
        """Initialize the conversation chatbot"""
        logger.info("🤖 Starting Chatbot Conversation System...")
        
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
        
        # Initialize components
        self.db_engine = create_engine(self.database_url)
        self.vector_store = None
        self.agent = None
        
        # Setup components
        self._setup_sql_chain()
        self._setup_vector_store()
        self._setup_agent()
        
        logger.info("✅ Chatbot conversation system ready!")
    
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
        """Setup Weaviate vector store connection"""
        try:
            logger.info("📚 Connecting to existing Weaviate vector store...")
            
            # Try to connect to existing Weaviate instance
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
                
                # Initialize embeddings
                embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
                
                # Connect to existing vector store
                try:
                    self.vector_store = Weaviate(
                        client=weaviate_client,
                        index_name="PDFReports",
                        text_key="content",
                        embedding=embeddings
                    )
                    logger.info("✅ Connected to existing vector store")
                except Exception as e:
                    logger.warning(f"⚠️ Could not connect to existing vector store: {str(e)}")
                    logger.info("📝 Continuing without vector store - chatbot will work with database queries only")
                    self.vector_store = None
                
            except Exception as e:
                logger.warning(f"⚠️ Weaviate connection failed: {str(e)}")
                logger.info("📝 Continuing without vector store - chatbot will work with database queries only")
                self.vector_store = None
            
        except Exception as e:
            logger.warning(f"⚠️ Vector store setup failed: {str(e)}")
            logger.info("📝 Continuing without vector store - chatbot will work with database queries only")
            self.vector_store = None
    
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
            
            # Create PDF retriever tool (if vector store is available)
            pdf_retriever = None
            if self.vector_store is not None:
                pdf_retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": 4}
                )
            
            # Create custom PDF tool
            class PDFReportTool(BaseTool):
                name: str = "pdf_reports"
                description: str = """
                Use this tool to search through PDF reports for cross-sell and up-sell recommendations.
                Input should be a search query about recommendations, customer analysis, or report content.
                Examples:
                - "cross-sell recommendations for Starbucks"
                - "customer classification analysis"
                - "up-sell opportunities"
                """
                
                def _run(self, query: str) -> str:
                    try:
                        if pdf_retriever is None:
                            return "PDF reports are not available. Vector store setup failed or no PDF reports found."
                        
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
                max_iterations=10,  # Increased from 5 to 10
                max_execution_time=60  # Add execution time limit
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
            
            # Provide helpful error message with common solutions
            error_msg = f"Sorry, I encountered an error processing your query: {str(e)}\n\n"
            error_msg += "💡 **Common solutions:**\n"
            error_msg += "• Try rephrasing your question\n"
            error_msg += "• Check if you're asking about customers, products, sales, or stores\n"
            error_msg += "• For store counts, try: 'Which customer has the most stores?'\n"
            error_msg += "• For sales data, try: 'What are the top selling products?'\n"
            
            return error_msg
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            stats = {}
            
            with self.db_engine.connect() as conn:
                # Get table counts
                tables = ['customers', 'products', 'customer_catalogue', 'sales']
                for table in tables:
                    try:
                        result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        count = result.scalar()
                        stats[table] = count
                    except Exception as e:
                        stats[table] = f"Error: {str(e)}"
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting database stats: {str(e)}")
            return {"error": str(e)}
    
    def start_conversation(self):
        """Start interactive conversation"""
        try:
            print("🤖 Chatbot Conversation System")
            print("=" * 50)
            print("Type 'quit' or 'exit' to end the conversation")
            print("Type 'stats' to see database statistics")
            print("Type 'help' for example queries")
            print("=" * 50)
            
            # Get database stats
            stats = self.get_database_stats()
            print(f"📊 Database Status: {stats}")
            print()
            
            while True:
                try:
                    # Get user input
                    user_input = input("💬 You: ").strip()
                    
                    # Check for exit commands
                    if user_input.lower() in ['quit', 'exit', 'bye']:
                        print("👋 Goodbye! Thanks for chatting!")
                        break
                    
                    # Check for help
                    if user_input.lower() == 'help':
                        print("\n💡 Example queries:")
                        print("• Which customer has the most stores?")
                        print("• What are the top selling products?")
                        print("• Show me cross-sell recommendations for Starbucks")
                        print("• What is the total revenue by customer?")
                        print("• Which products are most popular?")
                        print()
                        continue
                    
                    # Check for stats
                    if user_input.lower() == 'stats':
                        stats = self.get_database_stats()
                        print(f"\n📊 Database Statistics: {stats}\n")
                        continue
                    
                    # Skip empty input
                    if not user_input:
                        continue
                    
                    # Process query
                    print("🤔 Processing your query...")
                    response = self.query(user_input)
                    print(f"🤖 Bot: {response}\n")
                    
                except KeyboardInterrupt:
                    print("\n👋 Goodbye! Thanks for chatting!")
                    break
                except Exception as e:
                    print(f"❌ Error: {str(e)}\n")
                    
        except Exception as e:
            logger.error(f"❌ Conversation failed: {str(e)}")
            print(f"❌ Conversation failed: {str(e)}")

def main():
    """Main function to start conversation"""
    try:
        # Check if database is initialized
        print("🔍 Checking if system is initialized...")
        
        # Load environment variables
        load_dotenv('chatbot.env')
        database_url = os.getenv('DATABASE_URL')
        
        if not database_url:
            print("❌ DATABASE_URL not found in chatbot.env")
            print("💡 Please run 'python chatbot_init.py' first to initialize the system")
            sys.exit(1)
        
        # Test database connection
        try:
            engine = create_engine(database_url)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM customers"))
                count = result.scalar()
                if count == 0:
                    print("❌ Database is not initialized")
                    print("💡 Please run 'python chatbot_init.py' first to initialize the system")
                    sys.exit(1)
                else:
                    print(f"✅ Database is initialized with {count} customers")
        except Exception as e:
            print(f"❌ Database connection failed: {str(e)}")
            print("💡 Please run 'python chatbot_init.py' first to initialize the system")
            sys.exit(1)
        
        # Start conversation
        chatbot = ChatbotConversation()
        chatbot.start_conversation()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Failed to start conversation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
