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
import os
from contextlib import asynccontextmanager
from api.routes import router as hackrx
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global startup status
startup_success = False
startup_error = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events"""
    global startup_success, startup_error
    
    # Startup
    try:
        logger.info("🚀 Starting RAG API Service...")
        
        # Validate environment variables
        required_vars = [
            ("HUGGINGFACEHUB_ACCESS_TOKEN", settings.HUGGINGFACEHUB_ACCESS_TOKEN),
            ("PINECONE_API_KEY", settings.PINECONE_API_KEY),
            ("PINECONE_INDEX_NAME", settings.PINECONE_INDEX_NAME)
        ]
        
        missing_vars = []
        for var_name, var_value in required_vars:
            if not var_value or str(var_value).strip() == "" or str(var_value).strip() == "None":
                missing_vars.append(var_name)
        
        if missing_vars:
            error_msg = f"❌ Missing required environment variables: {', '.join(missing_vars)}"
            logger.error(error_msg)
            startup_error = error_msg
            raise Exception(error_msg)
        
        # Test Pinecone connection with timeout
        try:
            from pinecone import Pinecone
            
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            index = pc.Index(settings.PINECONE_INDEX_NAME)
            
            # Test connection with timeout
            stats = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: index.describe_index_stats()
                ),
                timeout=15.0
            )
            
            logger.info(f"✅ Pinecone connection successful. Total vectors: {stats.total_vector_count if stats else 0}")
            
        except asyncio.TimeoutError:
            error_msg = "❌ Pinecone connection timeout (15s)"
            logger.error(error_msg)
            startup_error = error_msg
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"❌ Pinecone connection failed: {str(e)}"
            logger.error(error_msg)
            startup_error = error_msg
            raise Exception(error_msg)
        
        # Test HuggingFace embedding service (optional)
        try:
            logger.info("🔧 Testing HuggingFace embedding service...")
            from services.vector_store import embedding_model
            
            # Quick test embedding with timeout
            test_result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: embedding_model.embed_query("test connection")
                ),
                timeout=30.0
            )
            
            if test_result and len(test_result) > 0:
                logger.info("✅ HuggingFace embedding service working")
            else:
                logger.warning("⚠️ HuggingFace embedding service returned empty result")
                
        except asyncio.TimeoutError:
            logger.warning("⚠️ HuggingFace embedding test timeout (30s) - continuing anyway")
        except Exception as e:
            logger.warning(f"⚠️ HuggingFace embedding test failed: {e} - continuing anyway")
            
        startup_success = True
        logger.info("✅ All systems initialized successfully")
        
    except Exception as e:
        startup_success = False
        startup_error = str(e)
        logger.error(f"❌ Startup failed: {e}")
        # Don't raise here - let the app start but mark as unhealthy
        
    yield  # Application runs here
    
    # Shutdown
    logger.info("🛑 Shutting down RAG API Service...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="RAG API Service",
    description="Document Q&A service using RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the main router
app.include_router(hackrx, prefix="/api/v1")

@app.get("/")
async def root():
    """Root health check endpoint"""
    global startup_success, startup_error
    
    status = "ok" if startup_success else "startup_failed"
    response = {
        "status": status,
        "message": "RAG API Service is running" if startup_success else f"Startup failed: {startup_error}",
        "timestamp": time.time(),
        "version": "1.0.0"
    }
    
    if not startup_success:
        return JSONResponse(status_code=503, content=response)
    
    return response

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    global startup_success, startup_error
    
    if not startup_success:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": time.time(),
                "error": startup_error or "Startup failed",
                "services": {"startup": "failed"}
            }
        )
    
    try:
        # Check Pinecone connectivity
        from pinecone import Pinecone
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index = pc.Index(settings.PINECONE_INDEX_NAME)
        
        # Quick ping to Pinecone with timeout
        start_time = time.time()
        try:
            stats = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: index.describe_index_stats()
                ),
                timeout=10.0
            )
            pinecone_latency = round((time.time() - start_time) * 1000, 2)
            pinecone_status = "connected"
        except asyncio.TimeoutError:
            pinecone_latency = round((time.time() - start_time) * 1000, 2)
            pinecone_status = "timeout"
            stats = None
        
        return {
            "status": "healthy" if pinecone_status == "connected" else "degraded",
            "timestamp": time.time(),
            "services": {
                "pinecone": {
                    "status": pinecone_status,
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
                "pinecone_environment": getattr(settings, 'PINECONE_ENVIRONMENT', 'not_set')
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
    global startup_success, startup_error
    
    if not startup_success:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "timestamp": time.time(),
                "endpoint": "/api/v1/hackrx/run",
                "error": startup_error or "Service not initialized"
            }
        )
    
    try:
        # Test the core components
        from services.vector_store import embedding_model
        from pinecone import Pinecone
        
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index = pc.Index(settings.PINECONE_INDEX_NAME)
        
        # Test embedding service with timeout
        start_time = time.time()
        try:
            test_embedding = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: embedding_model.embed_query("test query")
                ),
                timeout=15.0
            )
            embedding_latency = round((time.time() - start_time) * 1000, 2)
            embedding_status = "working"
        except asyncio.TimeoutError:
            embedding_latency = round((time.time() - start_time) * 1000, 2)
            embedding_status = "timeout"
            test_embedding = None
        except Exception as e:
            embedding_latency = round((time.time() - start_time) * 1000, 2)
            embedding_status = f"error: {str(e)}"
            test_embedding = None
        
        # Test Pinecone query if embedding worked
        query_status = "skipped"
        query_latency = 0
        
        if test_embedding:
            start_time = time.time()
            try:
                test_results = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: index.query(
                            vector=test_embedding,
                            namespace="test",
                            top_k=1,
                            include_metadata=False
                        )
                    ),
                    timeout=10.0
                )
                query_latency = round((time.time() - start_time) * 1000, 2)
                query_status = "working"
            except asyncio.TimeoutError:
                query_latency = round((time.time() - start_time) * 1000, 2)
                query_status = "timeout"
            except Exception as e:
                query_latency = round((time.time() - start_time) * 1000, 2)
                query_status = f"error: {str(e)}"
        
        overall_status = "ready" if embedding_status == "working" and query_status == "working" else "degraded"
        
        return {
            "status": overall_status,
            "timestamp": time.time(),
            "endpoint": "/api/v1/hackrx/run",
            "components": {
                "embedding_service": {
                    "status": embedding_status,
                    "latency_ms": embedding_latency,
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                },
                "vector_search": {
                    "status": query_status,
                    "latency_ms": query_latency
                },
                "pdf_parser": {
                    "status": "loaded",
                    "library": "PyMuPDF with pdfplumber fallback"
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
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    logger.warning(f"⚠️ HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle all other exceptions"""
    logger.error(f"❌ Unhandled exception: {exc}", exc_info=True)
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
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        app, 
        host=host, 
        port=port, 
        log_level="info",
        access_log=True,
        loop="asyncio"
    )
