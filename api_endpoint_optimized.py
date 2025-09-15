from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from enum import Enum
import os
import json
import glob
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime
import numpy as np
import asyncio
import aiohttp
import time
import shutil
from contextlib import asynccontextmanager

# Import the optimized recommendation engine
from main_optimized import OptimizedRecommendationEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global storage for recommendations (in production, use Redis or database)
recommendations_storage = {}

# Initialize the optimized recommendation engine
engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    global engine
    
    # Startup
    try:
        engine = OptimizedRecommendationEngine()
        logger.info("✅ Optimized recommendation engine initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize recommendation engine: {str(e)}")
        engine = None
        yield
        return
    
    # Check and load existing recommendations or generate new ones
    if engine:
        logger.info("🚀 Checking for existing recommendations...")
        
        # Get all available customers
        customer_ids = list(engine.customer_lookup.keys())
        logger.info(f"📋 Found {len(customer_ids)} customers: {customer_ids}")
        
        # Check each customer for existing files
        existing_count = 0
        generated_count = 0
        failed_count = 0
        
        for customer_id in customer_ids:
            try:
                # Check if recommendations already exist for this customer
                # Extract just the value part from the enum (e.g., "C001" from CustomerID.C001)
                customer_id_str = str(customer_id).split('.')[-1] if '.' in str(customer_id) else str(customer_id)
                existing_files = check_existing_recommendations(customer_id_str)
                
                if existing_files and existing_files['json_file'] and os.path.exists(existing_files['json_file']):
                    # Load existing recommendations
                    logger.info(f"📁 Loading existing recommendations for {customer_id}...")
                    
                    with open(existing_files['json_file'], 'r') as f:
                        existing_data = json.load(f)
                    
                    # Store in memory for quick access
                    recommendations_storage[customer_id_str] = existing_data
                    existing_count += 1
                    
                    logger.info(f"✅ Loaded existing recommendations for {customer_id}")
                else:
                    # Generate new recommendations
                    logger.info(f"🔄 No existing files found, generating new recommendations for {customer_id}...")
                    start_time = time.time()
                    
                    # Generate recommendations
                    result = await engine.generate_recommendations(customer_id)
                    
                    if result is None:
                        logger.error(f"❌ Failed to generate recommendations for {customer_id}")
                        failed_count += 1
                        continue
                    
                    recommendations, rejected_recommendations, already_purchased_recommendations, customer_classification = result
                    
                    processing_time = time.time() - start_time
                    
                    # Create timestamp for file naming
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Prepare data
                    recommendations_data = {
                        "customer_id": customer_id,
                        "timestamp": timestamp,
                        "processing_time": processing_time,
                        "CustomerInfo": {
                            "CustomerID": customer_id,
                            "CustomerName": get_customer_name(customer_id)
                        },
                        "CustomerClassification": customer_classification,
                        "AcceptedRecommendations": recommendations,
                        "RejectedRecommendations": rejected_recommendations,
                        "AlreadyPurchasedRecommendations": already_purchased_recommendations,
                        "Summary": {
                            "TotalCrossSell": len(recommendations),
                            "TotalRejected": len(rejected_recommendations),
                            "TotalAlreadyPurchased": len(already_purchased_recommendations),
                            "TotalRecommendations": len(recommendations)
                        },
                        "PerformanceStats": engine.performance_stats
                    }
                    
                    # Save JSON file
                    await save_recommendations_to_file(customer_id_str, recommendations_data)
                    
                    # Generate PDF report
                    await generate_pdf_report(customer_id_str, recommendations_data)
                    
                    # Store in memory
                    recommendations_storage[customer_id_str] = recommendations_data
                    
                    generated_count += 1
                    logger.info(f"✅ Successfully generated new recommendations for {customer_id} in {processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ Error processing customer {customer_id}: {str(e)}")
                failed_count += 1
        
        logger.info(f"🎉 Startup completed: {existing_count} existing files loaded, {generated_count} new files generated, {failed_count} failed")
        
        # Generate combined report if we have individual reports
        if generated_count > 0 or existing_count > 0:
            try:
                logger.info("📊 Generating combined portfolio report...")
                await generate_combined_portfolio_report()
                logger.info("✅ Combined portfolio report generated successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to generate combined report: {str(e)}")
    
    yield
    
    # Shutdown (if needed)
    logger.info("🛑 Shutting down API server...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Strategy Agent 3.0 Optimized API",
    description="High-performance API for generating customer recommendations with automatic startup generation",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("✅ CORS middleware configured")

# Enum for customer IDs
class CustomerID(str, Enum):
    C001 = "C001"
    C002 = "C002"
    C003 = "C003"

# Pydantic models
class RegenerateRequest(BaseModel):
    force_regenerate: bool = True
    
    class Config:
        json_schema_extra = {
            "example": {
                "force_regenerate": True
            }
        }

class RecommendationResponse(BaseModel):
    success: bool
    message: str
    customer_id: str
    timestamp: str
    processing_time: float
    performance_stats: Dict[str, Any]
    data: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseModel):
    success: bool
    message: str
    error: str
    timestamp: str

@app.get("/customers", response_model=Dict[str, Any])
async def get_available_customers():
    """
    Get list of available customers and their information
    
    Returns information about all available customers in the system.
    """
    if not engine:
        raise HTTPException(
            status_code=500,
            detail="Recommendation engine not available"
        )
    
    try:
        customers_info = []
        for customer_id, customer_info in engine.customer_lookup.items():
            customers_info.append({
                "customer_id": customer_id,
                "customer_name": customer_info['CustomerName'],
                "customer_type": customer_info['CustomerType'],
                "country": customer_info['Country'],
                "region": customer_info['Region'],
                "total_stores": customer_info['TotalStores']
            })
        
        return {
            "success": True,
            "customers": customers_info,
            "total_customers": len(customers_info)
        }
        
    except Exception as e:
        logger.error(f"❌ Error retrieving customers: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving customers: {str(e)}"
        )

@app.get("/recommendations/{customer_id}/json", response_model=Dict[str, Any])
async def get_recommendations_json(customer_id: CustomerID):
    """
    Get recommendations in JSON format for a specific customer
    
    - **customer_id**: Customer ID (C001, C002, or C003)
    
    Returns the complete recommendations data in JSON format.
    """
    try:
        # First check if customer is in memory storage
        # Extract just the value part from the enum (e.g., "C001" from CustomerID.C001)
        customer_id_str = str(customer_id).split('.')[-1] if '.' in str(customer_id) else str(customer_id)
        if customer_id_str in recommendations_storage:
            recommendations_data = recommendations_storage[customer_id_str]
            
            # Try to load from file if available
            timestamp = recommendations_data.get("timestamp", "")
            json_file_path = f"recommendations/recommendations_{customer_id_str}_{timestamp}.json"
            
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as f:
                    file_data = json.load(f)
                    return convert_numpy_types(file_data)
            else:
                # Return from memory storage
                return convert_numpy_types({
                    "CustomerInfo": recommendations_data["CustomerInfo"],
                    "CustomerClassification": recommendations_data["CustomerClassification"],
                    "AcceptedRecommendations": recommendations_data["AcceptedRecommendations"],
                    "RejectedRecommendations": recommendations_data["RejectedRecommendations"],
                    "AlreadyPurchasedRecommendations": recommendations_data["AlreadyPurchasedRecommendations"],
                    "Summary": recommendations_data["Summary"],
                    "PerformanceStats": recommendations_data.get("PerformanceStats", {}),
                    "GeneratedAt": recommendations_data["timestamp"],
                    "ProcessingTime": recommendations_data.get("processing_time", 0)
                })
        
        # If not in memory, check for existing files
        existing_files = check_existing_recommendations(customer_id)
        
        if existing_files and existing_files['json_file']:
            logger.info(f"📁 Loading existing JSON recommendations for {customer_id}")
            
            with open(existing_files['json_file'], 'r') as f:
                file_data = json.load(f)
            
            # Store in memory for quick access
            recommendations_storage[customer_id_str] = file_data
            
            return convert_numpy_types(file_data)
        
        # No recommendations found
        raise HTTPException(
            status_code=404,
            detail=f"No recommendations found for customer {customer_id}. Please regenerate recommendations first using POST /regenerate_recommendations/{customer_id}"
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving JSON recommendations for {customer_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving recommendations: {str(e)}"
        )

@app.get("/recommendations/{customer_id}/download")
async def download_report(customer_id: CustomerID):
    """
    Download the PDF report for a specific customer
    
    - **customer_id**: Customer ID (C001, C002, or C003)
    
    Returns the PDF report file for download.
    """
    try:
        # First check if customer is in memory storage
        # Extract just the value part from the enum (e.g., "C001" from CustomerID.C001)
        customer_id_str = str(customer_id).split('.')[-1] if '.' in str(customer_id) else str(customer_id)
        if customer_id_str in recommendations_storage:
            recommendations_data = recommendations_storage[customer_id_str]
            timestamp = recommendations_data.get("timestamp", "")
            
            # Look for the PDF file
            pdf_file_path = f"reports/analysis_report_{customer_id_str}_{timestamp}.pdf"
            
            if os.path.exists(pdf_file_path):
                return FileResponse(
                    path=pdf_file_path,
                    media_type='application/pdf',
                    filename=f"analysis_report_{customer_id}_{timestamp}.pdf"
                )
        
        # If not in memory or PDF not found, check for existing files
        existing_files = check_existing_recommendations(customer_id)
        
        if existing_files and existing_files['pdf_file'] and os.path.exists(existing_files['pdf_file']):
            logger.info(f"📁 Loading existing PDF report for {customer_id}")
            
            return FileResponse(
                path=existing_files['pdf_file'],
                media_type='application/pdf',
                filename=f"analysis_report_{customer_id}_{existing_files['timestamp']}.pdf"
            )
        
        # Try to find any existing PDF for this customer
        # Extract just the value part from the enum (e.g., "C001" from CustomerID.C001)
        customer_id_str = str(customer_id).split('.')[-1] if '.' in str(customer_id) else str(customer_id)
        pdf_pattern = f"reports/analysis_report_{customer_id_str}_*.pdf"
        pdf_files = glob.glob(pdf_pattern)
        
        if pdf_files:
            # Get the most recent file
            pdf_file_path = max(pdf_files, key=os.path.getctime)
            filename = os.path.basename(pdf_file_path)
            
            logger.info(f"📁 Found existing PDF report: {filename}")
            
            return FileResponse(
                path=pdf_file_path,
                media_type='application/pdf',
                filename=filename
            )
        
        # If no PDF found, try to generate it if we have recommendations
        if customer_id in recommendations_storage:
            recommendations_data = recommendations_storage[customer_id]
            pdf_success = await generate_pdf_report(customer_id, recommendations_data)
            
            if pdf_success:
                # Try again after generation
                pdf_files = glob.glob(pdf_pattern)
                if pdf_files:
                    pdf_file_path = max(pdf_files, key=os.path.getctime)
                    filename = os.path.basename(pdf_file_path)
                    
                    return FileResponse(
                        path=pdf_file_path,
                        media_type='application/pdf',
                        filename=filename
                    )
            
            raise HTTPException(
                status_code=500,
                detail=f"PDF report generation failed for customer {customer_id}. Check if pdf_report_generator.py exists and reportlab is installed."
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"No recommendations found for customer {customer_id}. Please regenerate recommendations first using POST /regenerate_recommendations/{customer_id}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error downloading report for {customer_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading report: {str(e)}"
        )

@app.get("/recommendations", response_model=Dict[str, Any])
async def list_existing_recommendations():
    """
    List all existing recommendations files
    
    Returns information about all existing recommendation files for all customers.
    """
    try:
        recommendations_info = []
        
        # Get all JSON files in recommendations directory
        json_pattern = "recommendations/recommendations_*.json"
        json_files = glob.glob(json_pattern)
        
        for json_file in json_files:
            try:
                # Extract customer ID and timestamp from filename
                filename = os.path.basename(json_file)
                parts = filename.replace("recommendations_", "").replace(".json", "").split("_")
                
                if len(parts) >= 2:
                    customer_id = parts[0]
                    timestamp = "_".join(parts[1:])
                    
                    # Check if corresponding PDF exists
                    pdf_file = f"reports/analysis_report_{customer_id}_{timestamp}.pdf"
                    pdf_exists = os.path.exists(pdf_file)
                    
                    # Get file stats
                    json_stats = os.stat(json_file)
                    
                    recommendations_info.append({
                        "customer_id": customer_id,
                        "timestamp": timestamp,
                        "json_file": json_file,
                        "pdf_file": pdf_file if pdf_exists else None,
                        "pdf_exists": pdf_exists,
                        "file_size_bytes": json_stats.st_size,
                        "created_at": datetime.fromtimestamp(json_stats.st_ctime).isoformat(),
                        "modified_at": datetime.fromtimestamp(json_stats.st_mtime).isoformat()
                    })
                    
            except Exception as e:
                logger.warning(f"⚠️  Error processing file {json_file}: {str(e)}")
                continue
        
        # Sort by creation time (newest first)
        recommendations_info.sort(key=lambda x: x["created_at"], reverse=True)
        
        return {
            "success": True,
            "total_recommendations": len(recommendations_info),
            "recommendations": recommendations_info
        }
        
    except Exception as e:
        logger.error(f"❌ Error listing recommendations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing recommendations: {str(e)}"
        )

@app.get("/recommendations/all", response_model=Dict[str, Any])
async def get_all_recommendations_json():
    """
    Get all customer recommendations as JSON objects array
    
    Returns an array containing the complete recommendation data for all customers.
    """
    try:
        all_recommendations = []
        
        # Get all JSON files in recommendations directory
        json_pattern = "recommendations/recommendations_*.json"
        json_files = glob.glob(json_pattern)
        
        if not json_files:
            return {
                "success": True,
                "message": "No recommendation files found",
                "total_customers": 0,
                "recommendations": []
            }
        
        for json_file in json_files:
            try:
                # Load JSON data
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                
                # Convert numpy types and add to array
                converted_data = convert_numpy_types(json_data)
                all_recommendations.append(converted_data)
                
            except Exception as e:
                logger.warning(f"⚠️  Error loading file {json_file}: {str(e)}")
                continue
        
        # Sort by customer ID for consistent ordering
        all_recommendations.sort(key=lambda x: x.get('CustomerInfo', {}).get('CustomerID', ''))
        
        logger.info(f"✅ Loaded {len(all_recommendations)} customer recommendation files")
        
        return {
            "success": True,
            "message": f"Successfully loaded {len(all_recommendations)} customer recommendations",
            "total_customers": len(all_recommendations),
            "recommendations": all_recommendations
        }
        
    except Exception as e:
        logger.error(f"❌ Error loading all recommendations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error loading all recommendations: {str(e)}"
        )

@app.get("/recommendations/download/all")
async def download_all_reports_combined():
    """
    Download a combined PDF containing all customer analysis reports
    
    Returns a single PDF file that contains all customer analysis reports merged together.
    """
    try:
        # Get all PDF files in reports directory
        pdf_pattern = "reports/analysis_report_*.pdf"
        pdf_files = glob.glob(pdf_pattern)
        
        if not pdf_files:
            raise HTTPException(
                status_code=404,
                detail="No PDF reports found. Please generate recommendations first."
            )
        
        # Sort by creation time to get the most recent files for each customer
        pdf_files.sort(key=os.path.getctime, reverse=True)
        
        # Group by customer ID to get the latest report for each customer
        customer_pdfs = {}
        for pdf_file in pdf_files:
            filename = os.path.basename(pdf_file)
            # Extract customer ID from filename: analysis_report_C001_timestamp.pdf
            parts = filename.replace("analysis_report_", "").replace(".pdf", "").split("_")
            if len(parts) >= 2:
                customer_id = parts[0]
                if customer_id not in customer_pdfs:
                    customer_pdfs[customer_id] = pdf_file
        
        if not customer_pdfs:
            raise HTTPException(
                status_code=404,
                detail="No valid PDF reports found for customers."
            )
        
        # Create combined PDF
        combined_pdf_path = await create_combined_pdf(list(customer_pdfs.values()))
        
        if not combined_pdf_path or not os.path.exists(combined_pdf_path):
            raise HTTPException(
                status_code=500,
                detail="Failed to create combined PDF report."
            )
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"all_customers_analysis_report_{timestamp}.pdf"
        
        logger.info(f"✅ Created combined PDF with {len(customer_pdfs)} customer reports")
        
        return FileResponse(
            path=combined_pdf_path,
            media_type='application/pdf',
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error creating combined PDF: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating combined PDF: {str(e)}"
        )

@app.post("/regenerate_recommendations", response_model=Dict[str, Any])
async def regenerate_all_recommendations(request: RegenerateRequest, background_tasks: BackgroundTasks):
    """
    Regenerate recommendations for all customers
    
    - **force_regenerate**: Whether to force regeneration (default: True)
    
    Generates new recommendations for each customer, then immediately deletes old files for that customer after successful generation.
    Returns an array containing the complete recommendation data for all customers.
    """
    if not engine:
        raise HTTPException(
            status_code=500,
            detail="Optimized recommendation engine not available"
        )
    
    try:
        logger.info("🔄 Starting regeneration of recommendations for all customers...")
        
        # Get all available customers
        customer_ids = list(engine.customer_lookup.keys())
        logger.info(f"📋 Regenerating for {len(customer_ids)} customers: {customer_ids}")
        
        # Generate recommendations for all customers first (before deleting old files)
        success_count = 0
        failed_count = 0
        results = {}
        
        for customer_id in customer_ids:
            try:
                logger.info(f"🔄 Regenerating recommendations for {customer_id}...")
                start_time = time.time()
                
                # Generate recommendations
                result = await engine.generate_recommendations(customer_id)
                
                if result is None:
                    logger.error(f"❌ Failed to generate recommendations for {customer_id}")
                    failed_count += 1
                    results[customer_id] = {"success": False, "error": "Failed to generate recommendations"}
                    continue
                
                recommendations, rejected_recommendations, already_purchased_recommendations, customer_classification = result
                
                processing_time = time.time() - start_time
                
                # Create timestamp for file naming
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Prepare data
                recommendations_data = {
                "customer_id": customer_id,
                    "timestamp": timestamp,
                    "processing_time": processing_time,
                    "CustomerInfo": {
                        "CustomerID": customer_id,
                        "CustomerName": get_customer_name(customer_id)
                    },
                    "CustomerClassification": customer_classification,
                    "AcceptedRecommendations": recommendations,
                    "RejectedRecommendations": rejected_recommendations,
                    "AlreadyPurchasedRecommendations": already_purchased_recommendations,
                    "Summary": {
                        "TotalCrossSell": len(recommendations),
                        "TotalRejected": len(rejected_recommendations),
                        "TotalAlreadyPurchased": len(already_purchased_recommendations),
                        "TotalRecommendations": len(recommendations)
                    },
                    "PerformanceStats": engine.performance_stats
                }
                
                # Save JSON file
                await save_recommendations_to_file(customer_id, recommendations_data)
                
                # Generate PDF report
                await generate_pdf_report(customer_id, recommendations_data)
                
                # Store in memory
                recommendations_storage[str(customer_id)] = recommendations_data
                
                # Extract customer ID string for deletion
                customer_id_str = str(customer_id).split('.')[-1] if '.' in str(customer_id) else str(customer_id)
                
                # Immediately delete old files for this customer after successful generation
                logger.info(f"🗑️  Deleting old recommendation files for customer {customer_id}...")
                await clear_old_customer_recommendation_files(customer_id, timestamp)
                
                success_count += 1
                results[customer_id] = {
                    "success": True,
                    "processing_time": processing_time,
                    "recommendations_count": len(recommendations),
                    "timestamp": timestamp
                }
                
                logger.info(f"✅ Successfully regenerated recommendations for {customer_id} in {processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ Error regenerating recommendations for {customer_id}: {str(e)}")
                failed_count += 1
                results[customer_id] = {"success": False, "error": str(e)}
        
        logger.info(f"🎉 Regeneration completed: {success_count} successful, {failed_count} failed")
        logger.info(f"🗑️  Old files were deleted immediately after each customer's successful generation")
        
        # Return array of all recommendation data (similar to GET /recommendations/all)
        all_recommendations = []
        for customer_id_str, data in recommendations_storage.items():
            converted_data = convert_numpy_types(data)
            all_recommendations.append(converted_data)
        
        # Sort by customer ID for consistent ordering
        all_recommendations.sort(key=lambda x: x.get('CustomerInfo', {}).get('CustomerID', ''))
        
        return {
            "success": True,
            "message": f"Successfully regenerated recommendations for {success_count} customers",
            "total_customers": len(all_recommendations),
            "successful_customers": success_count,
            "failed_customers": failed_count,
            "recommendations": all_recommendations
        }
        
    except Exception as e:
        logger.error(f"❌ Error in regeneration: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Regeneration error: {str(e)}"
        )

@app.post("/regenerate_recommendations/{customer_id}", response_model=Dict[str, Any])
async def regenerate_customer_recommendations(customer_id: CustomerID, request: RegenerateRequest, background_tasks: BackgroundTasks):
    """
    Regenerate recommendations for a specific customer
    
    - **customer_id**: Customer ID (C001, C002, or C003)
    - **force_regenerate**: Whether to force regeneration (default: True)
    
    Generates new recommendations first, then deletes old JSON and PDF files only after successful generation.
    Returns the complete recommendation data for the customer.
    """
    if not engine:
        raise HTTPException(
            status_code=500, 
            detail="Optimized recommendation engine not available"
        )
    
    try:
        logger.info(f"🔄 Starting regeneration of recommendations for customer {customer_id}...")
        
        # Extract just the value part from the enum (e.g., "C001" from CustomerID.C001)
        customer_id_str = str(customer_id).split('.')[-1] if '.' in str(customer_id) else str(customer_id)
        
        # Generate recommendations first (before deleting old files)
        start_time = time.time()
        result = await engine.generate_recommendations(customer_id)
        
        if result is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate recommendations"
            )
        
        recommendations, rejected_recommendations, already_purchased_recommendations, customer_classification = result
        
        processing_time = time.time() - start_time
        
        # Create timestamp for file naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare data
        recommendations_data = {
            "customer_id": customer_id,
            "timestamp": timestamp,
            "processing_time": processing_time,
            "CustomerInfo": {
                "CustomerID": customer_id,
                "CustomerName": get_customer_name(customer_id)
            },
            "CustomerClassification": customer_classification,
            "AcceptedRecommendations": recommendations,
            "RejectedRecommendations": rejected_recommendations,
            "AlreadyPurchasedRecommendations": already_purchased_recommendations,
            "Summary": {
                "TotalCrossSell": len(recommendations),
                "TotalRejected": len(rejected_recommendations),
                "TotalAlreadyPurchased": len(already_purchased_recommendations),
                "TotalRecommendations": len(recommendations)
            },
            "PerformanceStats": engine.performance_stats
        }
        
        # Save JSON file
        await save_recommendations_to_file(customer_id, recommendations_data)
        
        # Generate PDF report
        await generate_pdf_report(customer_id, recommendations_data)
        
        # Now that new recommendations are successfully generated, delete old files
        logger.info(f"🗑️  Deleting old recommendation files for customer {customer_id}...")
        await clear_old_customer_recommendation_files(customer_id, timestamp)
        
        # Remove old data from memory storage (if it existed)
        if customer_id_str in recommendations_storage:
            del recommendations_storage[customer_id_str]
        
        # Store the new data in memory
        recommendations_storage[customer_id_str] = recommendations_data
        
        logger.info(f"✅ Successfully regenerated recommendations for {customer_id} in {processing_time:.2f}s")
        
        # Return the actual recommendation data instead of just success message
        return convert_numpy_types(recommendations_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error regenerating recommendations for {customer_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error regenerating recommendations: {str(e)}"
        )

@app.get("/performance-stats", response_model=Dict[str, Any])
async def get_performance_stats():
    """
    Get current performance statistics from the engine
    
    Returns performance metrics and cache statistics.
    """
    if not engine:
        raise HTTPException(
            status_code=500,
            detail="Recommendation engine not available"
        )
    
    try:
        # Get cache statistics
        cache_dir = "cache"
        cache_files = glob.glob(os.path.join(cache_dir, "*.pkl"))
        cache_size = sum(os.path.getsize(f) for f in cache_files)
        
        return {
            "success": True,
            "engine_performance_stats": engine.performance_stats,
            "cache_statistics": {
                "cache_files_count": len(cache_files),
                "cache_size_bytes": cache_size,
                "cache_size_mb": round(cache_size / (1024 * 1024), 2),
                "product_embeddings_count": len(engine.product_embeddings),
                "ingredient_cache_count": len(engine.ingredient_cache)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error retrieving performance stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving performance stats: {str(e)}"
        )

@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """
    Health check endpoint with performance metrics
    
    Returns the health status of the API and recommendation engine.
    """
    engine_status = "healthy" if engine else "unhealthy"
    
    # Get basic performance info
    performance_info = {}
    if engine:
        performance_info = {
            "embedding_cache_hits": engine.performance_stats.get('embedding_cache_hits', 0),
            "embedding_cache_misses": engine.performance_stats.get('embedding_cache_misses', 0),
            "api_calls_made": engine.performance_stats.get('api_calls_made', 0)
        }
    
    return {
        "status": "healthy",
        "engine_status": engine_status,
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "performance_info": performance_info
    }

# Helper functions
def check_existing_recommendations(customer_id: str) -> Optional[Dict[str, str]]:
    """Check if recommendations already exist for a customer"""
    try:
        # Ensure customer_id is a string
        # Extract just the value part from the enum (e.g., "C001" from CustomerID.C001)
        customer_id_str = str(customer_id).split('.')[-1] if '.' in str(customer_id) else str(customer_id)
        
        # Look for existing JSON files
        json_pattern = f"recommendations/recommendations_{customer_id_str}_*.json"
        json_files = glob.glob(json_pattern)
        
        if not json_files:
            return None
        
        # Get the most recent JSON file
        latest_json = max(json_files, key=os.path.getctime)
        
        # Extract timestamp from filename
        filename = os.path.basename(latest_json)
        timestamp = filename.replace(f"recommendations_{customer_id_str}_", "").replace(".json", "")
        
        # Check if corresponding PDF exists
        pdf_file = f"reports/analysis_report_{customer_id_str}_{timestamp}.pdf"
        
        return {
            "json_file": latest_json,
            "pdf_file": pdf_file if os.path.exists(pdf_file) else None,
            "timestamp": timestamp
        }
        
    except Exception as e:
        logger.error(f"❌ Error checking existing recommendations: {str(e)}")
        return None

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, 'item'):  # Handle other numpy scalar types
        return obj.item()
    else:
        return obj

async def save_recommendations_to_file(customer_id: str, recommendations_data: Dict[str, Any]):
    """Save recommendations to file"""
    try:
        # Ensure customer_id is a string
        # Extract just the value part from the enum (e.g., "C001" from CustomerID.C001)
        customer_id_str = str(customer_id).split('.')[-1] if '.' in str(customer_id) else str(customer_id)
        timestamp = recommendations_data["timestamp"]
        filename = f"recommendations/recommendations_{customer_id_str}_{timestamp}.json"
        
        # Ensure directory exists
        os.makedirs("recommendations", exist_ok=True)
        
        # Convert numpy types and save to file
        converted_data = convert_numpy_types(recommendations_data)
        with open(filename, 'w') as f:
            json.dump(converted_data, f, indent=2, default=str)
        
        logger.info(f"💾 Recommendations saved to: {filename}")
        
    except Exception as e:
        logger.error(f"❌ Error saving recommendations to file: {str(e)}")

async def generate_pdf_report(customer_id: str, recommendations_data: Dict[str, Any]):
    """Generate PDF report"""
    try:
        # Ensure customer_id is a string
        # Extract just the value part from the enum (e.g., "C001" from CustomerID.C001)
        customer_id_str = str(customer_id).split('.')[-1] if '.' in str(customer_id) else str(customer_id)
        timestamp = recommendations_data["timestamp"]
        json_filename = f"recommendations/recommendations_{customer_id_str}_{timestamp}.json"
        pdf_filename = f"reports/analysis_report_{customer_id_str}_{timestamp}.pdf"
        
        # Ensure reports directory exists
        os.makedirs("reports", exist_ok=True)
        
        # Check if pdf_report_generator.py exists
        pdf_generator_path = "pdf_report_generator.py"
        if not os.path.exists(pdf_generator_path):
            logger.warning(f"⚠️  PDF report generation skipped (pdf_report_generator.py not found)")
            return False
        
        # Import and use the PDF generator
        try:
            from pdf_report_generator import PDFReportGenerator
            generator = PDFReportGenerator()
            success = generator.generate_pdf_report(json_filename, pdf_filename)
            
            if success:
                logger.info(f"📄 PDF report generated: {pdf_filename}")
                return True
            else:
                logger.error(f"❌ Failed to generate PDF report: {pdf_filename}")
                return False
                
        except ImportError as import_err:
            logger.error(f"❌ Import error for PDF generator: {str(import_err)}")
            logger.info("💡 Make sure reportlab is installed: pip install reportlab")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error generating PDF report: {str(e)}")
        return False

async def clear_all_recommendation_files():
    """Clear all recommendation files"""
    try:
        # Clear recommendations directory
        if os.path.exists("recommendations"):
            for file in glob.glob("recommendations/*.json"):
                os.remove(file)
            logger.info("🗑️  Cleared all JSON recommendation files")
        
        # Clear reports directory
        if os.path.exists("reports"):
            for file in glob.glob("reports/*.pdf"):
                os.remove(file)
            logger.info("🗑️  Cleared all PDF report files")
        
    except Exception as e:
        logger.error(f"❌ Error clearing recommendation files: {str(e)}")

async def clear_customer_recommendation_files(customer_id: str):
    """Clear recommendation files for a specific customer"""
    try:
        # Ensure customer_id is a string
        # Extract just the value part from the enum (e.g., "C001" from CustomerID.C001)
        customer_id_str = str(customer_id).split('.')[-1] if '.' in str(customer_id) else str(customer_id)
        
        # Clear JSON files for customer
        json_pattern = f"recommendations/recommendations_{customer_id_str}_*.json"
        json_files = glob.glob(json_pattern)
        for file in json_files:
            os.remove(file)
        
        # Clear PDF files for customer
        pdf_pattern = f"reports/analysis_report_{customer_id_str}_*.pdf"
        pdf_files = glob.glob(pdf_pattern)
        for file in pdf_files:
            os.remove(file)
        
        logger.info(f"🗑️  Cleared recommendation files for customer {customer_id_str}")
        
    except Exception as e:
        logger.error(f"❌ Error clearing files for customer {customer_id}: {str(e)}")

async def clear_old_customer_recommendation_files(customer_id: str, exclude_timestamp: str):
    """Clear old recommendation files for a specific customer, excluding the newly generated ones"""
    try:
        # Ensure customer_id is a string
        # Extract just the value part from the enum (e.g., "C001" from CustomerID.C001)
        customer_id_str = str(customer_id).split('.')[-1] if '.' in str(customer_id) else str(customer_id)
        
        deleted_count = 0
        
        # Clear old JSON files for customer (excluding the new one)
        json_pattern = f"recommendations/recommendations_{customer_id_str}_*.json"
        json_files = glob.glob(json_pattern)
        for file in json_files:
            # Check if this is NOT the newly generated file
            if exclude_timestamp not in file:
                os.remove(file)
                deleted_count += 1
                logger.info(f"🗑️  Deleted old JSON file: {os.path.basename(file)}")
        
        # Clear old PDF files for customer (excluding the new one)
        pdf_pattern = f"reports/analysis_report_{customer_id_str}_*.pdf"
        pdf_files = glob.glob(pdf_pattern)
        for file in pdf_files:
            # Check if this is NOT the newly generated file
            if exclude_timestamp not in file:
                os.remove(file)
                deleted_count += 1
                logger.info(f"🗑️  Deleted old PDF file: {os.path.basename(file)}")
        
        logger.info(f"🗑️  Cleared {deleted_count} old recommendation files for customer {customer_id_str}")
        
    except Exception as e:
        logger.error(f"❌ Error clearing old files for customer {customer_id}: {str(e)}")

async def clear_old_all_recommendation_files(exclude_timestamps: dict):
    """Clear old recommendation files for all customers, excluding the newly generated ones"""
    try:
        total_deleted = 0
        
        # Clear old JSON files (excluding the new ones)
        json_pattern = "recommendations/recommendations_*.json"
        json_files = glob.glob(json_pattern)
        for file in json_files:
            filename = os.path.basename(file)
            # Extract customer ID and timestamp from filename
            parts = filename.replace("recommendations_", "").replace(".json", "").split("_")
            if len(parts) >= 2:
                customer_id = parts[0]
                file_timestamp = "_".join(parts[1:])
                
                # Check if this file should be excluded (it's a newly generated one)
                if customer_id in exclude_timestamps and exclude_timestamps[customer_id] == file_timestamp:
                    continue  # Skip this file, it's newly generated
                
                os.remove(file)
                total_deleted += 1
                logger.info(f"🗑️  Deleted old JSON file: {filename}")
        
        # Clear old PDF files (excluding the new ones)
        pdf_pattern = "reports/analysis_report_*.pdf"
        pdf_files = glob.glob(pdf_pattern)
        for file in pdf_files:
            filename = os.path.basename(file)
            # Extract customer ID and timestamp from filename
            parts = filename.replace("analysis_report_", "").replace(".pdf", "").split("_")
            if len(parts) >= 2:
                customer_id = parts[0]
                file_timestamp = "_".join(parts[1:])
                
                # Check if this file should be excluded (it's a newly generated one)
                if customer_id in exclude_timestamps and exclude_timestamps[customer_id] == file_timestamp:
                    continue  # Skip this file, it's newly generated
                
                os.remove(file)
                total_deleted += 1
                logger.info(f"🗑️  Deleted old PDF file: {filename}")
        
        logger.info(f"🗑️  Cleared {total_deleted} old recommendation files for all customers")
        
    except Exception as e:
        logger.error(f"❌ Error clearing old files for all customers: {str(e)}")

def get_customer_name(customer_id: str) -> str:
    """Get customer name from the engine"""
    if not engine:
        return "Unknown"
    
    try:
        if customer_id in engine.customer_lookup:
            return engine.customer_lookup[customer_id]['CustomerName']
        return "Unknown"
    except:
        return "Unknown"

async def generate_combined_portfolio_report():
    """Generate combined portfolio report using the combined report generator"""
    try:
        # Import the combined report generator
        from combined_report_generator import CombinedReportGenerator
        
        # Find all JSON files in recommendations directory
        json_files = [f for f in os.listdir('recommendations') if f.startswith('recommendations_') and f.endswith('.json')]
        
        if not json_files:
            logger.warning("⚠️ No recommendation JSON files found for combined report")
            return False
        
        # Sort by modification time to get the most recent
        json_files.sort(key=lambda x: os.path.getmtime(os.path.join('recommendations', x)), reverse=True)
        
        logger.info(f"📄 Found {len(json_files)} JSON files for combined report")
        
        # Generate combined PDF report
        generator = CombinedReportGenerator()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"reports/combined_portfolio_report_{timestamp}.pdf"
        
        success = generator.generate_combined_report(json_files, output_path)
        
        if success:
            logger.info(f"✅ Combined portfolio report generated: {output_path}")
            return True
        else:
            logger.error("❌ Failed to generate combined portfolio report")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error generating combined portfolio report: {str(e)}")
        return False

async def create_combined_pdf(pdf_files: List[str]) -> str:
    """Create a combined PDF from multiple PDF files"""
    try:
        from PyPDF2 import PdfMerger
        import tempfile
        
        # Create a temporary file for the combined PDF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_filename = f"reports/combined_analysis_report_{timestamp}.pdf"
        
        # Ensure reports directory exists
        os.makedirs("reports", exist_ok=True)
        
        # Create PDF merger
        merger = PdfMerger()
        
        # Add each PDF to the merger
        for pdf_file in pdf_files:
            if os.path.exists(pdf_file):
                try:
                    merger.append(pdf_file)
                    logger.info(f"📄 Added {os.path.basename(pdf_file)} to combined PDF")
                except Exception as e:
                    logger.warning(f"⚠️  Error adding {pdf_file} to combined PDF: {str(e)}")
                    continue
        
        # Write the combined PDF
        with open(combined_filename, 'wb') as output_file:
            merger.write(output_file)
        
        merger.close()
        
        logger.info(f"✅ Created combined PDF: {combined_filename}")
        return combined_filename
        
    except ImportError:
        logger.error("❌ PyPDF2 not installed. Please install it: pip install PyPDF2")
        return None
    except Exception as e:
        logger.error(f"❌ Error creating combined PDF: {str(e)}")
        return None

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            message="Request failed",
            error=exc.detail,
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"❌ Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            message="Internal server error",
            error="An unexpected error occurred",
            timestamp=datetime.now().isoformat()
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Strategy Agent 3.0 Optimized API Server")
    print("=" * 60)
    print("📚 API Documentation: http://localhost:8002/docs")
    print("📖 ReDoc Documentation: http://localhost:8002/redoc")
    print("🔍 Health Check: http://localhost:8002/health")
    print("📊 Performance Stats: http://localhost:8002/performance-stats")
    print("=" * 60)
    print("🚀 Features:")
    print("   ✅ Automatic recommendation generation on startup")
    print("   ✅ Async API calls with connection pooling")
    print("   ✅ Vector persistence and caching")
    print("   ✅ Batch similarity calculations")
    print("   ✅ Parallel processing")
    print("   ✅ Performance monitoring")
    print("   ✅ Background task processing")
    print("=" * 60)
    
    uvicorn.run(
        "api_endpoint_optimized:app",
        host="127.0.0.1",
        port=8002,
        reload=True,
        log_level="info"
    )