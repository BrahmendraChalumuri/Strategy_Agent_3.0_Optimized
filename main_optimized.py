import pandas as pd
import requests
import os
import asyncio
import aiohttp
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import json
from datetime import datetime
import time
import pickle
import hashlib
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter for Perplexity API calls with exponential backoff"""
    
    def __init__(self, max_requests_per_minute: int = 30, max_concurrent: int = 5):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_concurrent = max_concurrent
        self.request_times = []
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.last_request_time = 0
        
    async def acquire(self):
        """Acquire permission to make a request"""
        await self.semaphore.acquire()
        
        # Rate limiting logic
        current_time = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # If we're at the rate limit, wait
        if len(self.request_times) >= self.max_requests_per_minute:
            wait_time = 60 - (current_time - self.request_times[0]) + 1
            logger.warning(f"⏳ Rate limit reached, waiting {wait_time:.1f}s...")
            await asyncio.sleep(wait_time)
            # Clean up old requests after waiting
            current_time = time.time()
            self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0.1, 0.5)
        await asyncio.sleep(jitter)
        
        self.request_times.append(current_time)
        
    def release(self):
        """Release the semaphore"""
        self.semaphore.release()

# Import rate limit configuration
from rate_limit_config import get_rate_limit_config

# Get rate limit configuration
rate_config = get_rate_limit_config()

# Global rate limiter instance
rate_limiter = RateLimiter(
    max_requests_per_minute=rate_config["max_requests_per_minute"], 
    max_concurrent=rate_config["max_concurrent_requests"]
)

# Load environment variables
load_dotenv('api_keys.env')

class OptimizedRecommendationEngine:
    def __init__(self):
        logger.info("🚀 Initializing Optimized Recommendation Engine...")
        
        # Load data with optimized pandas operations
        self._load_data_optimized()
        
        # Load sentence transformer model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Initialize caching system
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load or create embeddings with persistence
        self._load_or_create_embeddings()
        
        # OpenAI GPT API configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_url = "https://api.openai.com/v1/chat/completions"
        
        # Performance tracking
        self.performance_stats = {
            'embedding_cache_hits': 0,
            'embedding_cache_misses': 0,
            'api_calls_made': 0,
            'api_calls_parallel': 0,
            'similarity_calculations': 0
        }
        
        logger.info("✅ Optimized Recommendation Engine initialized successfully")
    
    def _load_data_optimized(self):
        """Load data with optimized pandas operations"""
        logger.info("📊 Loading data with optimized operations...")
        
        # Load all CSV files in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                'customers': executor.submit(pd.read_csv, "data/customer.csv"),
                'catalogue': executor.submit(pd.read_csv, "data/customer_catalogue_enhanced.csv"),
                'products': executor.submit(pd.read_csv, "data/products.csv"),
                'sales': executor.submit(pd.read_csv, "data/sales_enhanced.csv"),
                'stores': executor.submit(pd.read_csv, "data/stores.csv")
            }
            
            # Wait for all to complete
            for name, future in futures.items():
                setattr(self, name, future.result())
                logger.info(f"✅ Loaded {name}: {len(getattr(self, name))} rows")
        
        # Pre-compute lookup dictionaries for O(1) access
        self._create_lookup_dicts()
    
    def _create_lookup_dicts(self):
        """Create lookup dictionaries for O(1) access instead of iterrows()"""
        logger.info("🔍 Creating optimized lookup dictionaries...")
        
        # Product lookup by name - handle duplicates by using ProductID as key
        self.product_lookup = {}
        for _, row in self.products.iterrows():
            product_name = row['Name']
            product_id = row['ProductID']
            # Use ProductID as key to handle duplicate names
            self.product_lookup[product_id] = {
                'Name': product_name,
                'ProductID': product_id,
                'Category': row['Category'],
                'SubCategory': row['SubCategory'],
                'Price': row['Price'],
                'Tags': row['Tags']
            }
        
        # Also create a name-based lookup for similarity matching (handle duplicates)
        self.product_name_lookup = {}
        for _, row in self.products.iterrows():
            product_name = row['Name']
            product_id = row['ProductID']
            if product_name not in self.product_name_lookup:
                self.product_name_lookup[product_name] = []
            self.product_name_lookup[product_name].append(product_id)
        
        # Customer lookup
        self.customer_lookup = self.customers.set_index('CustomerID').to_dict('index')
        
        # Sales lookup by customer
        self.sales_by_customer = self.sales.groupby('CustomerID')
        
        # Stores lookup by customer
        self.stores_by_customer = self.stores.groupby('CustomerID')
        
        # Catalogue lookup by customer
        self.catalogue_by_customer = self.catalogue.groupby('CustomerID')
        
        logger.info("✅ Lookup dictionaries created")
    
    def _get_cache_path(self, cache_type: str, identifier: str) -> str:
        """Generate cache file path"""
        hash_id = hashlib.md5(identifier.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{cache_type}_{hash_id}.pkl")
    
    def _load_or_create_embeddings(self):
        """Load embeddings from cache or create new ones with persistence"""
        logger.info("🧠 Loading or creating embeddings with persistence...")
        
        # Check for existing product embeddings cache
        product_cache_path = os.path.join(self.cache_dir, "product_embeddings.pkl")
        
        if os.path.exists(product_cache_path):
            logger.info("📁 Loading product embeddings from cache...")
            with open(product_cache_path, 'rb') as f:
                self.product_embeddings = pickle.load(f)
            logger.info(f"✅ Loaded {len(self.product_embeddings)} product embeddings from cache")
        else:
            logger.info("🔄 Creating new product embeddings...")
            self._create_product_embeddings()
            # Save to cache
            with open(product_cache_path, 'wb') as f:
                pickle.dump(self.product_embeddings, f)
            logger.info(f"💾 Saved {len(self.product_embeddings)} product embeddings to cache")
        
        # Initialize ingredient cache
        self.ingredient_cache = {}
        
        # Create product embedding matrix for batch operations
        self._create_embedding_matrix()
    
    def _create_product_embeddings(self):
        """Create product embeddings with batch processing"""
        product_names = self.products['Name'].tolist()
        
        # Batch encode all products at once
        logger.info(f"🔄 Encoding {len(product_names)} products in batch...")
        embeddings = self.model.encode(product_names, convert_to_tensor=True, show_progress_bar=True)
        
        # Create dictionary
        self.product_embeddings = dict(zip(product_names, embeddings))
    
    def _create_embedding_matrix(self):
        """Create embedding matrix for efficient batch similarity calculations"""
        logger.info("📊 Creating embedding matrix for batch operations...")
        
        # Stack all product embeddings into a matrix
        self.product_embedding_matrix = torch.stack(list(self.product_embeddings.values()))
        self.product_names = list(self.product_embeddings.keys())
        
        logger.info(f"✅ Created embedding matrix: {self.product_embedding_matrix.shape}")
    
    async def query_openai_api_async(self, session: aiohttp.ClientSession, 
                                       ingredient: str, product_name: str, 
                                       catalogue_item_name: str, catalogue_category: str, 
                                       catalogue_description: str, catalogue_ingredients: str) -> Tuple[bool, str]:
        """Async OpenAI GPT API query with rate limiting and exponential backoff"""
        if not self.openai_api_key:
            logger.info(f"🔍 Potential match (API key missing): {ingredient} → {product_name}")
            return True, "API key missing - defaulting to True"
        
        # Acquire rate limiter permission
        await rate_limiter.acquire()
        
        try:
            # Construct the prompt
            prompt = f"""Analyze if the product "{product_name}" could realistically be used as an actual ingredient in making "{catalogue_item_name}".

Consider:
1. Culinary compatibility and cooking methods
2. Whether the suggested product fits the food category and preparation style
3. Real-world kitchen usage, not just ingredient name matching
4. Food science and baking/cooking logic

Catalogue Item Details:
- Product Name: {catalogue_item_name}
- Product Category: {catalogue_category}
- Description: {catalogue_description}
- Current Ingredients: {catalogue_ingredients}

Potential Ingredient Match: {ingredient}
Suggested Product: {product_name}

Example: "Biscuit Dough" in a chocolate chip cookie recipe refers to cookie dough, NOT actual biscuit dough (which is for biscuits, not cookies).

Please answer with ONLY "YES" or "NO" followed by brief reasoning focusing on real culinary compatibility (max 50 words)."""
            
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0.1
            }
            
            # Reduced timeout for better performance
            timeout = aiohttp.ClientTimeout(total=15)
            
            async with session.post(self.openai_url, headers=headers, json=data, timeout=timeout) as response:
                if response.status == 200:
                    result = await response.json()
                    ai_response = result['choices'][0]['message']['content'].strip()
                    
                    self.performance_stats['api_calls_made'] += 1
                    
                    if ai_response.upper().startswith('YES'):
                        logger.info(f"      ✅ AI Confirmed: {ingredient} → {product_name}")
                        return True, ai_response
                    elif ai_response.upper().startswith('NO'):
                        logger.info(f"      ❌ AI Rejected: {ingredient} → {product_name}")
                        return False, ai_response
                    else:
                        logger.warning(f"      ⚠️  AI Response unclear: {ai_response}")
                        return False, ai_response
                elif response.status == 429:
                    # Rate limit exceeded - implement exponential backoff
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"      ⚠️  Rate limit exceeded (429), waiting {retry_after}s...")
                    await asyncio.sleep(retry_after + random.uniform(1, 5))  # Add jitter
                    # Retry the request once
                    async with session.post(self.openai_url, headers=headers, json=data, timeout=timeout) as retry_response:
                        if retry_response.status == 200:
                            result = await retry_response.json()
                            ai_response = result['choices'][0]['message']['content'].strip()
                            self.performance_stats['api_calls_made'] += 1
                            
                            if ai_response.upper().startswith('YES'):
                                logger.info(f"      ✅ AI Confirmed (retry): {ingredient} → {product_name}")
                                return True, ai_response
                            elif ai_response.upper().startswith('NO'):
                                logger.info(f"      ❌ AI Rejected (retry): {ingredient} → {product_name}")
                                return False, ai_response
                            else:
                                logger.warning(f"      ⚠️  AI Response unclear (retry): {ai_response}")
                                return False, ai_response
                        else:
                            logger.warning(f"      ⚠️  Retry failed: {retry_response.status}")
                            return True, f"Retry failed {retry_response.status} - Defaulting to True"
                else:
                    logger.warning(f"      ⚠️  API Error: {response.status}")
                    return True, f"API Error {response.status} - Defaulting to True"
                    
        except asyncio.TimeoutError:
            logger.warning(f"      ⚠️  API Timeout: {ingredient} → {product_name}")
            return True, "API Timeout - Defaulting to True"
        except Exception as e:
            logger.warning(f"      ⚠️  API Exception: {str(e)}")
            return True, f"Exception: {str(e)}"
        finally:
            # Always release the rate limiter
            rate_limiter.release()
    
    def classify_customer_type(self, customer_id: str) -> Dict:
        """Optimized customer classification using lookup dictionaries"""
        # Use pre-computed lookup instead of filtering
        if customer_id not in self.customer_lookup:
            raise ValueError(f"Customer {customer_id} not found")
        
        # Get sales data using groupby (much faster than filtering)
        if customer_id in self.sales_by_customer.groups:
            cust_sales = self.sales_by_customer.get_group(customer_id)
            total_quantity = cust_sales['Quantity'].sum()
        else:
            total_quantity = 0
        
        # Get stores count using groupby
        if customer_id in self.stores_by_customer.groups:
            number_of_stores = len(self.stores_by_customer.get_group(customer_id))
        else:
            number_of_stores = 0
        
        logger.info(f"    📊 Customer Analysis:")
        logger.info(f"       Total Quantity Sold: {total_quantity:,}")
        logger.info(f"       Number of Stores: {number_of_stores}")
        
        # Classify customer
        if number_of_stores > 50 or total_quantity > 200000:
            customer_type = "CHG Own Sales Customer"
            logger.info(f"       Customer Type: {customer_type} (Large Scale)")
        elif (25 <= number_of_stores <= 50) or (50000 < total_quantity <= 200000):
            customer_type = "Distributor Customer"
            logger.info(f"       Customer Type: {customer_type} (Medium Scale)")
        else:
            customer_type = "Small Customer"
            logger.info(f"       Customer Type: {customer_type} (Small Scale)")
        
        return {
            "CustomerType": customer_type,
            "TotalQuantitySold": int(total_quantity),
            "NumberOfStores": int(number_of_stores),
            "ClassificationCriteria": {
                "StoresGreaterThan50": number_of_stores > 50,
                "QuantityGreaterThan200K": total_quantity > 200000,
                "StoresBetween25And50": 25 <= number_of_stores <= 50,
                "QuantityBetween50KAnd200K": 50000 < total_quantity <= 200000
            }
        }
    
    def get_customer_data(self, customer_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Optimized customer data retrieval using lookup dictionaries"""
        # Use pre-computed groupby instead of filtering
        if customer_id in self.catalogue_by_customer.groups:
            cust_catalogue = self.catalogue_by_customer.get_group(customer_id)
        else:
            cust_catalogue = pd.DataFrame()
        
        if customer_id in self.sales_by_customer.groups:
            cust_sales = self.sales_by_customer.get_group(customer_id)
            purchased_product_ids = cust_sales['ProductID'].unique()
        else:
            cust_sales = pd.DataFrame()
            purchased_product_ids = np.array([])
        
        # Filter unsold products using vectorized operations
        unsold_products = self.products[~self.products['ProductID'].isin(purchased_product_ids)]
        
        logger.info(f"    📊 Customer has purchased {len(purchased_product_ids)} unique products")
        logger.info(f"    🆕 {len(unsold_products)} products available for cross-sell")
        
        return cust_catalogue, cust_sales, unsold_products
    
    async def analyze_cross_sell_optimized(self, ingredients: List[str], unsold_products: pd.DataFrame, 
                                   cust_sales: pd.DataFrame, item_id: str) -> Tuple[List, List, List]:
        """Highly optimized cross-sell analysis with batch operations and parallel processing"""
        cross_sell_items = []
        rejected_items = []
        already_purchased_items = []
        
        # Process ingredients efficiently
        processed_ingredients = []
        for ingredient_list in ingredients:
            ingredient_parts = ingredient_list.split(',')
            for part in ingredient_parts:
                cleaned_ingredient = part.strip()
                if cleaned_ingredient and cleaned_ingredient.lower() not in ['water', 'salt', 'sugar']:
                    processed_ingredients.append(cleaned_ingredient)
        
        if not processed_ingredients:
            return cross_sell_items, rejected_items, already_purchased_items
        
        logger.info(f"    🔍 Processing {len(processed_ingredients)} ingredients: {processed_ingredients}")
        
        # Get purchased product IDs
        purchased_product_ids = cust_sales['ProductID'].unique() if not cust_sales.empty else np.array([])
        
        # Batch process all ingredients at once
        ingredient_embeddings = []
        valid_ingredients = []
        
        for ingredient in processed_ingredients:
            # Check cache first
            cache_path = self._get_cache_path("ingredient", ingredient)
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    embedding = pickle.load(f)
                self.performance_stats['embedding_cache_hits'] += 1
            else:
                embedding = self.model.encode(ingredient, convert_to_tensor=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump(embedding, f)
                self.performance_stats['embedding_cache_misses'] += 1
            
            ingredient_embeddings.append(embedding)
            valid_ingredients.append(ingredient)
        
        if not ingredient_embeddings:
            return cross_sell_items, rejected_items, already_purchased_items
        
        # Batch similarity calculation - much faster than individual calls
        ingredient_matrix = torch.stack(ingredient_embeddings)
        similarities = util.cos_sim(ingredient_matrix, self.product_embedding_matrix)
        
        self.performance_stats['similarity_calculations'] += len(ingredients) * len(self.product_names)
        
        # Find matches above threshold
        matches = []
        for i, ingredient in enumerate(valid_ingredients):
            ingredient_similarities = similarities[i]
            high_similarity_indices = torch.where(ingredient_similarities > 0.7)[0]
            
            for idx in high_similarity_indices:
                product_name = self.product_names[idx]
                similarity = ingredient_similarities[idx].item()
                
                logger.info(f"      🎯 Match found: '{ingredient}' → '{product_name}' (Sim: {similarity:.3f})")
                
                # Get product info from lookup - handle duplicate names
                if product_name in self.product_name_lookup:
                    # Get the first product ID for this name (or could choose based on criteria)
                    product_id = self.product_name_lookup[product_name][0]
                    product_info = self.product_lookup[product_id]
                    
                    matches.append({
                        'ingredient': ingredient,
                        'product_name': product_name,
                        'product_id': product_id,
                        'similarity': similarity,
                        'product_info': product_info
                    })
        
        # Process matches with parallel API calls
        if matches:
            await self._process_matches_parallel(matches, item_id, purchased_product_ids, 
                                               cross_sell_items, rejected_items, already_purchased_items)
        
        return cross_sell_items, rejected_items, already_purchased_items
    
    async def _process_matches_parallel(self, matches: List[Dict], item_id: str, 
                                      purchased_product_ids: np.ndarray,
                                      cross_sell_items: List, rejected_items: List, 
                                      already_purchased_items: List):
        """Process matches with parallel API calls"""
        logger.info(f"    🚀 Processing {len(matches)} matches with parallel API calls...")
        
        # Get catalogue item details once
        catalogue_item_details = self.catalogue[self.catalogue['CustomerCatalogueItemID'] == item_id]
        if catalogue_item_details.empty:
            logger.warning(f"    ⚠️  No catalogue details found for item {item_id}")
            return
        
        catalogue_item_name = catalogue_item_details['ProductName'].iloc[0]
        catalogue_category = catalogue_item_details['Product Category'].iloc[0]
        catalogue_description = catalogue_item_details['Description'].iloc[0]
        catalogue_ingredients = catalogue_item_details['Ingredients'].iloc[0]
        
        # Create async session with reduced connection pooling for rate limiting
        connector = aiohttp.TCPConnector(limit=5, limit_per_host=3)
        timeout = aiohttp.ClientTimeout(total=15)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create tasks for parallel API calls
            tasks = []
            for match in matches:
                task = self.query_openai_api_async(
                    session, match['ingredient'], match['product_name'],
                    catalogue_item_name, catalogue_category, catalogue_description, catalogue_ingredients
                )
                tasks.append((match, task))
            
            # Process results as they complete
            for match, task in tasks:
                try:
                    confirmed, reasoning = await task
                    
                    if confirmed:
                        product_id = match['product_id']
                        
                        if product_id not in purchased_product_ids:
                            logger.info(f"        ✅ Verified: Customer hasn't purchased {match['product_name']} (ID: {product_id})")
                            cross_sell_items.append({
                                "Ingredient": match['ingredient'],
                                "SuggestedProduct": match['product_name'],
                                "ProductID": product_id,
                                "Similarity": round(match['similarity'], 3),
                                "Category": match['product_info']['Category'],
                                "Price": match['product_info']['Price'],
                                "AIReasoning": reasoning,
                                "Status": "Accepted"
                            })
                        else:
                            logger.info(f"        ❌ Skipped: Customer has already purchased {match['product_name']} (ID: {product_id})")
                            already_purchased_items.append({
                                "Ingredient": match['ingredient'],
                                "SuggestedProduct": match['product_name'],
                                "ProductID": product_id,
                                "Similarity": round(match['similarity'], 3),
                                "Category": match['product_info']['Category'],
                                "Price": match['product_info']['Price'],
                                "AIReasoning": reasoning,
                                "Status": "Already Purchased"
                            })
                    else:
                        logger.info(f"        ❌ Rejected by AI: {match['ingredient']} → {match['product_name']} (Reason: {reasoning})")
                        rejected_items.append({
                            "Ingredient": match['ingredient'],
                            "SuggestedProduct": match['product_name'],
                            "ProductID": match['product_id'],
                            "Similarity": round(match['similarity'], 3),
                            "Category": match['product_info']['Category'],
                            "Price": match['product_info']['Price'],
                            "AIReasoning": reasoning,
                            "Status": "Rejected"
                        })
                        
                except Exception as e:
                    logger.error(f"        ❌ Error processing match: {str(e)}")
                    # Default to accepted on error
                    cross_sell_items.append({
                        "Ingredient": match['ingredient'],
                        "SuggestedProduct": match['product_name'],
                        "ProductID": match['product_id'],
                        "Similarity": round(match['similarity'], 3),
                        "Category": match['product_info']['Category'],
                        "Price": match['product_info']['Price'],
                        "AIReasoning": f"Error: {str(e)}",
                        "Status": "Accepted"
                    })
        
        self.performance_stats['api_calls_parallel'] += len(matches)
        logger.info(f"    ✅ Processed {len(matches)} matches with parallel API calls")
    
    async def generate_recommendations(self, customer_id: str):
        """Generate comprehensive recommendations with performance tracking"""
        start_time = time.time()
        logger.info(f"\n🔍 Analyzing recommendations for CustomerID: {customer_id}")
        
        # Get customer info using lookup
        if customer_id not in self.customer_lookup:
            logger.error(f"❌ Customer {customer_id} not found!")
            return [], [], [], {}
        
        customer_info = self.customer_lookup[customer_id]
        customer_name = customer_info['CustomerName']
        logger.info(f"📊 Customer: {customer_name}")
        
        # Store customer info
        self.current_customer_name = customer_name
        self.current_customer_id = customer_id
        
        # Classify customer type
        customer_classification = self.classify_customer_type(customer_id)
        
        # Get customer-specific data
        cust_catalogue, cust_sales, unsold_products = self.get_customer_data(customer_id)
        
        if cust_catalogue.empty:
            logger.error(f"❌ No catalogue items found for customer {customer_id}")
            return [], [], [], customer_classification
        
        logger.info(f"📋 Found {len(cust_catalogue)} catalogue items")
        logger.info(f"🆕 {len(unsold_products)} products available for cross-sell")
        
        recommendations = []
        rejected_recommendations = []
        already_purchased_recommendations = []
        items_with_recommendations = 0
        
        # Process catalogue items efficiently
        for _, row in cust_catalogue.iterrows():
            item_id = row['CustomerCatalogueItemID']
            item_name = row.get('ProductName', 'Unnamed Product')
            qty_required = row['QuantityRequired']
            ingredients = str(row['Ingredients']).split(";")
            
            logger.info(f"\n🔍 Analyzing: {item_name} (ID: {item_id})")
            
            # Cross-sell analysis (optimized)
            cross_sell_items, rejected_items, already_purchased_items = await self.analyze_cross_sell_optimized(
                ingredients, unsold_products, cust_sales, item_id
            )
            
            # Add to recommendations if there are actual recommendations
            if cross_sell_items:
                recommendation = {
                    "CustomerCatalogueItemID": item_id,
                    "ProductName": item_name,
                    "QuantityRequired": qty_required,
                    "Ingredients": ingredients,
                    "CrossSell": cross_sell_items
                }
                recommendations.append(recommendation)
                items_with_recommendations += 1
            
            # Add rejected items
            if rejected_items:
                rejected_recommendation = {
                    "CustomerCatalogueItemID": item_id,
                    "ProductName": item_name,
                    "QuantityRequired": qty_required,
                    "Ingredients": ingredients,
                    "RejectedCrossSell": rejected_items
                }
                rejected_recommendations.append(rejected_recommendation)
            
            # Add already purchased items
            if already_purchased_items:
                already_purchased_recommendation = {
                    "CustomerCatalogueItemID": item_id,
                    "ProductName": item_name,
                    "QuantityRequired": qty_required,
                    "Ingredients": ingredients,
                    "AlreadyPurchasedCrossSell": already_purchased_items
                }
                already_purchased_recommendations.append(already_purchased_recommendation)
        
        # Performance summary
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"✅ Found recommendations for {items_with_recommendations} out of {len(cust_catalogue)} catalogue items")
        logger.info(f"❌ Found {len(rejected_recommendations)} items with rejected cross-sell opportunities")
        logger.info(f"🔄 Found {len(already_purchased_recommendations)} items with already purchased cross-sell opportunities")
        logger.info(f"⏱️  Total processing time: {processing_time:.2f} seconds")
        
        # Log performance stats
        logger.info(f"📊 Performance Stats:")
        logger.info(f"   Embedding cache hits: {self.performance_stats['embedding_cache_hits']}")
        logger.info(f"   Embedding cache misses: {self.performance_stats['embedding_cache_misses']}")
        logger.info(f"   API calls made: {self.performance_stats['api_calls_made']}")
        logger.info(f"   API calls parallel: {self.performance_stats['api_calls_parallel']}")
        logger.info(f"   Similarity calculations: {self.performance_stats['similarity_calculations']}")
        
        return recommendations, rejected_recommendations, already_purchased_recommendations, customer_classification
    
    def display_recommendations(self, recommendations, rejected_recommendations, already_purchased_recommendations, 
                              customer_id, customer_classification=None):
        """Display formatted recommendations with performance info"""
        logger.info(f"\n{'='*80}")
        logger.info(f"📈 OPTIMIZED RECOMMENDATION REPORT")
        logger.info(f"{'='*80}")
        
        if not recommendations and not rejected_recommendations and not already_purchased_recommendations:
            logger.info("❌ No recommendations found")
            return
        
        total_cross_sell = 0
        
        for rec in recommendations:
            logger.info(f"\n🧾 Catalogue Item: {rec['ProductName']}")
            logger.info(f"   ID: {rec['CustomerCatalogueItemID']}")
            logger.info(f"   Required Quantity: {rec['QuantityRequired']}")
            
            # Cross-sell recommendations
            if 'CrossSell' in rec and rec['CrossSell']:
                logger.info(f"   🔁 Cross-Sell Opportunities ({len(rec['CrossSell'])}):")
                for cross in rec['CrossSell']:
                    logger.info(f"      • {cross['Ingredient']} → {cross['SuggestedProduct']}")
                    logger.info(f"        Product ID: {cross['ProductID']}")
                    logger.info(f"        Category: {cross['Category']}")
                    logger.info(f"        Price: ${cross['Price']}")
                    logger.info(f"        Similarity Score: {cross['Similarity']}")
                    if 'AIReasoning' in cross:
                        logger.info(f"        AI Reasoning: {cross['AIReasoning']}")
                    total_cross_sell += 1
            else:
                logger.info(f"   🔁 No cross-sell opportunities")
        
        # Display rejected recommendations
        if rejected_recommendations:
            logger.info(f"\n{'='*80}")
            logger.info(f"❌ REJECTED RECOMMENDATIONS")
            logger.info(f"{'='*80}")
            
            total_rejected = 0
            for rec in rejected_recommendations:
                logger.info(f"\n🧾 Catalogue Item: {rec['ProductName']}")
                logger.info(f"   ID: {rec['CustomerCatalogueItemID']}")
                logger.info(f"   Required Quantity: {rec['QuantityRequired']}")
                
                if 'RejectedCrossSell' in rec and rec['RejectedCrossSell']:
                    logger.info(f"   ❌ Rejected Cross-Sell Opportunities ({len(rec['RejectedCrossSell'])}):")
                    for rejected in rec['RejectedCrossSell']:
                        logger.info(f"      • {rejected['Ingredient']} → {rejected['SuggestedProduct']}")
                        logger.info(f"        Product ID: {rejected['ProductID']}")
                        logger.info(f"        Category: {rejected['Category']}")
                        logger.info(f"        Price: ${rejected['Price']}")
                        logger.info(f"        Similarity Score: {rejected['Similarity']}")
                        logger.info(f"        AI Reasoning: {rejected['AIReasoning']}")
                        total_rejected += 1
        
        # Display already purchased recommendations
        if already_purchased_recommendations:
            logger.info(f"\n{'='*80}")
            logger.info(f"🔄 ALREADY PURCHASED RECOMMENDATIONS")
            logger.info(f"{'='*80}")
            
            total_already_purchased = 0
            for rec in already_purchased_recommendations:
                logger.info(f"\n🧾 Catalogue Item: {rec['ProductName']}")
                logger.info(f"   ID: {rec['CustomerCatalogueItemID']}")
                logger.info(f"   Required Quantity: {rec['QuantityRequired']}")
                
                if 'AlreadyPurchasedCrossSell' in rec and rec['AlreadyPurchasedCrossSell']:
                    logger.info(f"   🔄 Already Purchased Cross-Sell Opportunities ({len(rec['AlreadyPurchasedCrossSell'])}):")
                    for already_purchased in rec['AlreadyPurchasedCrossSell']:
                        logger.info(f"      • {already_purchased['Ingredient']} → {already_purchased['SuggestedProduct']}")
                        logger.info(f"        Product ID: {already_purchased['ProductID']}")
                        logger.info(f"        Category: {already_purchased['Category']}")
                        logger.info(f"        Price: ${already_purchased['Price']}")
                        logger.info(f"        Similarity Score: {already_purchased['Similarity']}")
                        logger.info(f"        AI Reasoning: {already_purchased['AIReasoning']}")
                        total_already_purchased += 1
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total Cross-Sell Opportunities: {total_cross_sell}")
        logger.info(f"Total Rejected Cross-Sell Opportunities: {total_rejected}")
        logger.info(f"Total Already Purchased Cross-Sell Opportunities: {total_already_purchased}")
        logger.info(f"Total Recommendations: {total_cross_sell}")
        
        # Save recommendations to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recommendations/recommendations_{customer_id}_{timestamp}.json"
        
        # Combine all recommendations
        all_recommendations = {
            "CustomerInfo": {
                "CustomerID": self.current_customer_id,
                "CustomerName": self.current_customer_name
            },
            "CustomerClassification": customer_classification,
            "AcceptedRecommendations": recommendations,
            "RejectedRecommendations": rejected_recommendations,
            "AlreadyPurchasedRecommendations": already_purchased_recommendations,
            "Summary": {
                "TotalCrossSell": total_cross_sell,
                "TotalRejected": total_rejected,
                "TotalAlreadyPurchased": total_already_purchased,
                "TotalRecommendations": total_cross_sell
            },
            "PerformanceStats": self.performance_stats
        }
        
        with open(filename, 'w') as f:
            json.dump(all_recommendations, f, indent=2, default=str)
        
        logger.info(f"\n💾 Recommendations saved to: {filename}")
        
        # Generate PDF report automatically
        try:
            from pdf_report_generator_optimized import PDFReportGenerator
            generator = PDFReportGenerator()
            pdf_filename = f"reports/analysis_report_{customer_id}_{timestamp}.pdf"
            success = generator.generate_pdf_report(filename, pdf_filename)
            
            if success:
                logger.info(f"📄 PDF analysis report generated: {pdf_filename}")
            else:
                logger.error("❌ Failed to generate PDF report")
        except ImportError:
            logger.warning("⚠️  PDF report generation skipped (pdf_report_generator_optimized.py not found)")
        except Exception as e:
            logger.error(f"❌ Error generating PDF report: {str(e)}")

# CLI interface removed - this module is now used only as a library for the API
