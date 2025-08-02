# from fastapi import FastAPI
# from api.routes import router as hackrx

# app = FastAPI()

# app.include_router(hackrx)

# @app.get("/")
# def health_check():
#     return {"status": "ok"}

# import logging
# logging.basicConfig(level=logging.INFO)

# from fastapi import FastAPI
# from api.routes import router as hackrx
# from config.settings import settings

# app = FastAPI()

# @app.on_event("startup")
# async def startup_event():
#     try:
#         settings.validate()  # Add validation
#         print("✅ All environment variables loaded successfully")
#     except Exception as e:
#         print(f"❌ Startup error: {e}")
#         raise

# app.include_router(hackrx)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
import asyncio
from api.routes import router as hackrx
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG API Service",
    description="Document Q&A service using RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event to validate configuration
@app.on_event("startup")
async def startup_event():
    try:
        logger.info("🚀 Starting RAG API Service...")
        
        # Validate environment variables
        required_vars = [
            "HUGGINGFACEHUB_ACCESS_TOKEN",
            "PINECONE_API_KEY", 
            "PINECONE_INDEX_NAME"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not getattr(settings, var, None):
                missing_vars.append(var)
        
        if missing_vars:
            error_msg = f"❌ Missing required environment variables: {', '.join(missing_vars)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        # Test Pinecone connection
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            index = pc.Index(settings.PINECONE_INDEX_NAME)
            stats = index.describe_index_stats()
            logger.info(f"✅ Pinecone connection successful. Index stats: {stats}")
        except Exception as e:
            logger.error(f"❌ Pinecone connection failed: {e}")
            raise
            
        logger.info("✅ All systems initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

# Include the main router
app.include_router(hackrx, prefix="/api/v1")

@app.get("/")
async def root():
    """Root health check endpoint"""
    return {
        "status": "ok",
        "message": "RAG API Service is running",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    try:
        # Check Pinecone connectivity
        from pinecone import Pinecone
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index = pc.Index(settings.PINECONE_INDEX_NAME)
        
        # Quick ping to Pinecone
        start_time = time.time()
        stats = index.describe_index_stats()
        pinecone_latency = round((time.time() - start_time) * 1000, 2)
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "services": {
                "pinecone": {
                    "status": "connected",
                    "latency_ms": pinecone_latency,
                    "total_vectors": stats.total_vector_count if stats else 0,
                    "namespaces": len(stats.namespaces) if stats and stats.namespaces else 0
                },
                "huggingface": {
                    "status": "configured",
                    "token_available": bool(settings.HUGGINGFACEHUB_ACCESS_TOKEN)
                }
            },
            "environment": {
                "pinecone_index": settings.PINECONE_INDEX_NAME,
                "pinecone_environment": settings.PINECONE_ENVIRONMENT
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": time.time(),
                "error": str(e)
            }
        )

@app.get("/api/v1/hackrx/status")
async def hackrx_status():
    """Status check specifically for the hackrx endpoint"""
    try:
        # Test the core components
        from services.vector_store import embedding_model
        from pinecone import Pinecone
        
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index = pc.Index(settings.PINECONE_INDEX_NAME)
        
        # Test embedding service
        start_time = time.time()
        test_embedding = embedding_model.embed_query("test query")
        embedding_latency = round((time.time() - start_time) * 1000, 2)
        
        # Test Pinecone query
        start_time = time.time()
        test_results = index.query(
            vector=test_embedding,
            namespace="test",
            top_k=1,
            include_metadata=False
        )
        query_latency = round((time.time() - start_time) * 1000, 2)
        
        return {
            "status": "ready",
            "timestamp": time.time(),
            "endpoint": "/api/v1/hackrx/run",
            "components": {
                "embedding_service": {
                    "status": "working",
                    "latency_ms": embedding_latency,
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                },
                "vector_search": {
                    "status": "working",
                    "latency_ms": query_latency
                },
                "pdf_parser": {
                    "status": "loaded",
                    "library": "pdfplumber"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"HackRX status check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "timestamp": time.time(),
                "endpoint": "/api/v1/hackrx/run",
                "error": str(e)
            }
        )

@app.middleware("http")
async def log_requests(request, call_next):
    """Log all requests for debugging"""
    start_time = time.time()
    
    # Log request
    logger.info(f"📨 {request.method} {request.url}")
    
    response = await call_next(request)
    
    # Log response
    process_time = round((time.time() - start_time) * 1000, 2)
    logger.info(f"📤 {request.method} {request.url} - {response.status_code} ({process_time}ms)")
    
    return response

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": time.time()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")