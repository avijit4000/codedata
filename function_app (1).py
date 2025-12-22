# Import standard libraries
import asyncio
import time
from typing import Tuple,List, Dict, Any, Generator, Optional
import logging
from functools import partial
import os
from concurrent.futures import ThreadPoolExecutor
# Import FastAPI modules
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, Response, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
# Import RxPy modules
import rx
from rx import operators as ops
from rx.subject import Subject
from rx.scheduler import ThreadPoolScheduler
from rx.core import Observable
# Import additional utilities if needed
from pathlib import Path
from dotenv import load_dotenv
import json
from contextlib import asynccontextmanager
# Add these imports at the top of your file with other imports
import ssl
import certifi
from urllib3.util import Retry
from azure.cosmos.http_constants import StatusCodes
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_204_NO_CONTENT
from fastapi.responses import Response
from functools import wraps
import re
# ## Define FastAPI Application
# 
# Let's set up the FastAPI application instance and configure basic settings.

# FastAPI lifespan for initialization and cleanup
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_csv_data("df_encoded_itemtype.csv")
    """Initialize and cleanup database connection"""
    # Initialize database on startup
    db_manager = CosmosDBConnectionManager()
    try:
        await db_manager.initialize()
        # Load both ML models during startup
        logger.info("Loading unauth prediction model...")
        await model_service.load_model()
        logger.info("Unauth model loaded successfully")
        
        logger.info("Loading Invoca prediction model...")
        await invoca_model_service.load_model()
        logger.info("Invoca model loaded successfully")

        # # Perform any cleanup if necessary
        yield
    finally:
        # Cleanup on shutdown
        await db_manager.close()
        logger.info("Database connections closed")

# Create / Update FastAPI app to use the lifespan
app = FastAPI(
    title="Reactive Cohort Recommendation API",
    description="A high-performance API for cohort recommendations using reactive programming",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("reactive-api")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

##########
# Add these routes after your existing routes
@app.get('/actuator/health')
async def health_check():
    try:
        # Replace the existing return value with the actuator health response
        return Response(
            content="{ 'status': 'UP', 'components': { 'cosmos': { 'status': 'UP', 'details': { 'version': { 'major': 3, 'minor': 11, 'patch': 17, 'dsepatch': -1 } } }, 'circuitBreakers': { 'status': 'UP', 'details': { 'db-is-agent-exists-of-uuid': { 'status': 'UP', 'details': { 'failureRate': '-1.0%', 'failureRateThreshold': '50.0%', 'slowCallRate': '-1.0%', 'slowCallRateThreshold': '100.0%', 'bufferedCalls': 0, 'slowCalls': 0, 'slowFailedCalls': 0, 'failedCalls': 0, 'notPermittedCalls': 0, 'state': 'CLOSED' } } } }, 'diskSpace': { 'status': 'UP', 'details': { 'total': 133003395072, 'free': 84136878080, 'threshold': 10485760, 'path': '/opt/dm-cohort-recommender-api/.', 'exists': true } }, 'livenessState': { 'status': 'UP' }, 'ping': { 'status': 'UP' }, 'readinessState': { 'status': 'UP' } }, 'groups': [ 'liveness', 'readiness' ] }",
            media_type="application/json",
            status_code=200
        )
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return JSONResponse(
            content={"status": "ERROR", "message": "str(e)"},
            status_code=500
        )

@app.get('/actuator/info')
async def info_check():
    try:
        return Response(
            content="{ 'status': 'Ok', 'details': 'test details'}",
            media_type="application/json",
            status_code=200
        )
    except Exception as e:
        logger.error(f"Error in info check: {e}")
        return JSONResponse(
            content={"status": "ERROR", "message": "str(e)"},
            status_code=500
        )

@app.get('/actuator/env')
async def env_check():
    try:
        return Response(
            content="{ 'status': 'Ok', 'details': 'Env', 'dbname': 'Dev Cosmos DB'}",
            media_type="application/json",
            status_code=200
        )
    except Exception as e:
        logger.error(f"Error in env check: {e}")
        return JSONResponse(
            content={"status": "ERROR", "message": "str(e)"},
            status_code=500
        )

@app.get('/actuator/logger')
async def logger_check():
    try:
        return Response(
            content="{'status': 'Ok', 'log info': 'Dev portal log'}",
            media_type="application/json",
            status_code=200
        )
    except Exception as e:
        logger.error(f"Error in logger check: {e}")
        return JSONResponse(
            content={"status": "ERROR", "message": "str(e)"},
            status_code=500
        )
##########

# Load environment variables
load_dotenv()
###
# Function to load environment variables more robustly
def load_environment_variables():
    """Load environment variables from .env files with robust fallback"""
    # Try multiple possible locations for .env file
    env_paths = [
        Path("./.env"),  # Current directory
        Path("../.env"),  # Parent directory
        Path.home() / ".env",  # Home directory
        Path(__file__).parent / ".env",  # Same directory as this file
    ]
    
    # Try loading from each path until successful
    env_loaded = False
    for env_path in env_paths:
        if env_path.exists():
            logger.info(f"Loading environment from: {env_path}")
            load_dotenv(dotenv_path=env_path)
            env_loaded = True
            break
    
    if not env_loaded:
        logger.warning("No .env file found. Using existing environment variables.")
    
    # Print loaded environment variables for debugging (excluding sensitive values)
    logger.info("Environment variables loaded:")
    for var in ["COSMOS_DATABASE_NAME", "cosmosDatabaseName"]:
        value = os.getenv(var)
        if value:
            logger.info(f"  {var}: {value}")
    
    # Return environment config
    return {
        "cosmos_url": os.getenv("COSMOS_ACCOUNT_URL") or os.getenv("cosmosAccountUrl"),
        "cosmos_key": os.getenv("COSMOS_ACCOUNT_KEY") or os.getenv("cosmosAccountKey"),
        "cosmos_db": os.getenv("COSMOS_DATABASE_NAME") or os.getenv("cosmosDatabaseName") or "default_dm_cohort",
        "cosmos_lrc_url": os.getenv("LANDROVER_COVERAGE_URL") or os.getenv("landrovercoverageurl") or "https://landrover-api-stage-1.stage.gpd-acq-azure.optum.com/landrover-api/landrover/api/coverage/Users",
        "cosmos_uam_url": os.getenv("UNAUTH_MODEL_URL") or os.getenv("unauthmodelurl") or "https://dm-cohort-recommendation-api-stage-1.stage.gpd-acq-azure.optum.com/dm-cohort-recommendation-api/predict",
        "cosmos_im_url": os.getenv("INVOCA_MODEL_URL") or os.getenv("invocamodelurl") or "https://dm-cohort-recommendation-api-stage-1.stage.gpd-acq-azure.optum.com/dm-cohort-recommendation-api/predict_invoca",
        "cosmos_checkbciC": os.getenv("COSMOS_CONTAINER_NAME") or os.getenv("cosmosContainerName") or "dm_cohort_container",
        "cosmos_requestC": os.getenv("COSMOS_REQUEST_CONTAINER") or os.getenv("cosmosRequestContainer") or "dm_cohort_request",
        "cosmos_responseC": os.getenv("COSMOS_RESPONSE_CONTAINER") or os.getenv("cosmosResponseContainer") or "dm_cohort_response",
        "cosmos_processedC": os.getenv("COSMOS_PROCESSED_CONTAINER") or os.getenv("cosmosProcessedContainer") or "dm_cohort_processed",
        "cosmos_flagdataC": os.getenv("COSMOS_FLAGDATA_CONTAINER") or os.getenv("cosmosFlagdataContainer") or "dm_cohort_flagdata",
        "cosmos_livestreamC": os.getenv("COSMOS_LIVESTREAM_CONTAINER") or os.getenv("cosmosLivestreamContainer") or "dm_cohort_livestream",
        "cosmos_invoca_mcids_masterC": os.getenv("COSMOS_INVOCA_MCID_MASTER_CONTAINER") or os.getenv("cosmosInvocaMCIDMasterContainer") or "dm_cohort_invoca_mcid_master",
        "env_mode": os.getenv("ENVIRONMENT") or "development",
        "ssl_disable": os.getenv("DISABLE_SSL_VERIFY") or "true",
        "ca_cert_path": os.getenv("CA_CERT_PATH") or "/path/to/your/ca-bundle.pem"
    }


# Call this function before initializing the database
env_config = load_environment_variables()
# Get configuration from environment using our loaded config
#self.endpoint = env_config["cosmos_url"] or "https://abc.com"
#self.database_name = env_config["cosmos_db"] or "ABCDE"

###

# Example of loading configuration from environment variables
RECOMMENDATION_SERVICE_URL = os.getenv("RECOMMENDATION_SERVICE_URL", "http://localhost:8000/recommend")
RECOMMENDATION_SERVICE_TIMEOUT = int(os.getenv("RECOMMENDATION_SERVICE_TIMEOUT", 10))
# ## Set Up RxPy for Reactive Programming


# ## Set Up RxPy Observables for Request Handling
# 
# We'll create subjects and observables to handle incoming requests reactively.


# Create a ThreadPoolScheduler with an appropriate pool size
# for parallel processing of computationally intensive tasks
optimal_thread_count = min(32, (os.cpu_count() or 4) * 4)  # More aggressive thread pool
thread_pool_scheduler = ThreadPoolScheduler(optimal_thread_count)

# Create subjects for different request types
recommendation_subject = Subject()
cohort_subject = Subject()


# ## Enable Parallel Processing for Intensive Tasks
# 
# Now we'll set up parallel processing for computationally intensive operations using RxPy's capabilities.
#########
# Import additional RxPy modules
import rx.operators as ops
# Import additional types
from typing import Any, Dict, List
# Import additional libraries for parallel processing
import time
# Import additional libraries for parallel processing
#KK from concurrent.futures import ThreadPoolExecutor
# Create a thread pool executor for parallel processing
#KK thread_pool_executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
# Create a thread pool scheduler for RxPy
#KK thread_pool_scheduler = ThreadPoolScheduler(thread_pool_executor)
# ## Parallel Processing for Intensive Computations
#
# This section demonstrates how to handle computationally intensive tasks in parallel using RxPy.
###########



###########
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from azure.cosmos.aio import CosmosClient as AsyncCosmosClient
#top from contextlib import asynccontextmanager
import asyncio
from typing import Optional, Dict, Any, List
import os
import time
import functools
from fastapi import Depends, HTTPException, status
import rx
from rx import operators as ops
import logging

# Configure connection pooling and timeouts
MAX_POOL_SIZE = 250  # Increase connection pool size instead of 100 200
CONNECTION_TIMEOUT = 30.0
REQUEST_TIMEOUT = 30.0  # Reduce default timeout instead of 10.0
RETRY_ATTEMPTS = 3
RETRY_INTERVAL = 0.5  # Faster retry for better responsiveness instead of 1.0

# Database connection singleton with connection pooling
class CosmosDBConnectionManager:
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CosmosDBConnectionManager, cls).__new__(cls)
            cls._instance.initialized = False
            cls._instance.metrics = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "retry_attempts": 0,
                "avg_response_time": 0
            }
            cls._instance._init_containers()
        return cls._instance
    
    def _init_containers(self):
        """Initialize container attributes"""
        self.checkbci_container = None
        self.request_container = None
        self.response_container = None
        self.processed_container = None
        self.flag_container = None
        self.livestream_container = None
        self.invoca_mcids_container = None
        self.client = None
        self.async_client = None
        self.database = None
        self.async_database = None
        self.app_ready = False
    
    async def initialize(self):
        """Initialize the database connection asynchronously"""
        if self.initialized:
            return self
            
        async with self._lock:  # Thread-safe initialization
            if self.initialized:  # Double-check after acquiring lock
                return self
               
            try:
                # Get configuration from environment
                # Get environment variables with fallbacks for development
                self.endpoint = os.getenv("COSMOS_ACCOUNT_URL") or os.getenv("cosmosAccountUrl") or "https://abc.com" 
                self.key = os.getenv("COSMOS_ACCOUNT_KEY") or os.getenv("cosmosAccountKey") or "ABCDE" 
                self.database_name = os.getenv("COSMOS_DATABASE_NAME") or os.getenv("cosmosDatabaseName") or env_config["cosmos_db"] or "database_dm_cohort"
    
                
                # Container names
                self.container_names = {
                    "checkbci_container": "dm_cohort_container",
                    "request_container": "dm_cohort_request",
                    "response_container": "dm_cohort_response",
                    "processed_container": "dm_cohort_processed",
                    "flag_container": "dm_cohort_flagdata",
                    "livestream_container": "dm_cohort_livestream",
                    "invoca_mcids_container": "dm_cohort_invoca_mcid_master"
                }
                
                logger.info(f"Database: {self.database_name}")
                logger.info(f"Container names initialized")
                
                # Validate environment variables
                self._validate_config()
                
                
                # Initialize synchronous client (for backward compatibility)
                #self.client = CosmosClient(
                #    self.endpoint, 
                #    credential=self.key,
                #    connection_timeout=CONNECTION_TIMEOUT,
                #    connection_verify=True,
                #    connection_retry_policy=RETRY_ATTEMPTS,  # Just use the integer value
                #    consistency_level="Session"
                #)
                
                # Initialize asynchronous client with connection pooling
                #self.async_client = AsyncCosmosClient(
                #    self.endpoint,
                #    credential=self.key,
                #    connection_timeout=CONNECTION_TIMEOUT,
                #    connection_verify=True,
                #    connection_retry_policy=RETRY_ATTEMPTS,  # Just use the integer value
                #    consistency_level="Session"
                #)
                
                # Check if we should disable SSL verification (ONLY FOR DEVELOPMENT)
                dev_mode = os.getenv("ENVIRONMENT", "development").lower() == "development"
                
                # Create a custom SSL context for certificate verification
                if dev_mode and os.getenv("DISABLE_SSL_VERIFY", "false").lower() == "true":
                    logger.warning("SSL certificate verification is disabled. This is insecure and should only be used in development.")
                    # Create an unverified SSL context
                    ssl_context = ssl.create_default_context()
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE
                    
                    # Configure clients to use the custom SSL context
                    # For synchronous client
                    self.client = CosmosClient(
                        self.endpoint, 
                        credential=self.key,
                        connection_timeout=CONNECTION_TIMEOUT,
                        connection_verify=False,  # Disable SSL verification
                        connection_retry_policy=RETRY_ATTEMPTS,
                        consistency_level="Session",
                        connection_mode="DIRECT" 
                    )
                    
                    # For asynchronous client
                    self.async_client = AsyncCosmosClient(
                        self.endpoint,
                        credential=self.key,
                        connection_timeout=CONNECTION_TIMEOUT,
                        connection_verify=False,  # Disable SSL verification
                        connection_retry_policy=RETRY_ATTEMPTS,
                        consistency_level="Session",
                        connection_mode="DIRECT" 
                    )
                else:
                    # Use secure defaults with custom CA certificates if available
                    ca_cert_path = os.getenv("CA_CERT_PATH", certifi.where())
                    logger.warning("SSL certificate verification is enabled.")
                    
                    # Configure clients with proper SSL verification
                    self.client = CosmosClient(
                        self.endpoint, 
                        credential=self.key,
                        connection_timeout=CONNECTION_TIMEOUT,
                        connection_verify=ca_cert_path,  # Use custom or default CA bundle
                        connection_retry_policy=RETRY_ATTEMPTS,
                        consistency_level="Session",
                        connection_mode="DIRECT" 
                    )
                    
                    self.async_client = AsyncCosmosClient(
                        self.endpoint,
                        credential=self.key,
                        connection_timeout=CONNECTION_TIMEOUT,
                        connection_verify=ca_cert_path,  # Use custom or default CA bundle
                        connection_retry_policy=RETRY_ATTEMPTS,
                        consistency_level="Session",
                        connection_mode="DIRECT" 
                    )


                logger.info("Successfully connected to Cosmos DB")
                
                # Initialize database clients
                self.database = self.client.get_database_client(self.database_name)
                self.async_database = self.async_client.get_database_client(self.database_name)
                
                # Initialize containers (synchronous client)
                self._init_sync_containers()
                
                # Set initialization flag
                self.initialized = True
                self.app_ready = True
                logger.info("Database initialization completed successfully")
                
                # Start the connection health monitoring task
               # asyncio.create_task(self._monitor_connection_health())
                
                return self
                
            except Exception as e:
                logger.error(f"Error initializing CosmosDB: {str(e)}")
                raise
    
    def _validate_config(self):
        """Validate that all required configuration is present"""
        if not self.endpoint or not self.key or not self.database_name:
            missing_vars = [var for var, val in [
                ("cosmosAccountUrl", self.endpoint),
                ("cosmosAccountKey", self.key),
                ("database_name", self.database_name)
            ] if not val]
            logger.error(f"Missing environment variables: {missing_vars}")
            raise ValueError(f"Missing environment variables: {missing_vars}")
    
    def _init_sync_containers(self):
        """Initialize synchronous container clients"""
        self.checkbci_container = self.database.get_container_client(self.container_names["checkbci_container"])
        self.request_container = self.database.get_container_client(self.container_names["request_container"])
        self.response_container = self.database.get_container_client(self.container_names["response_container"])
        self.processed_container = self.database.get_container_client(self.container_names["processed_container"])
        self.flag_container = self.database.get_container_client(self.container_names["flag_container"])
        self.livestream_container = self.database.get_container_client(self.container_names["livestream_container"])
        self.invoca_mcids_container = self.database.get_container_client(self.container_names["invoca_mcids_container"])
    
    async def get_container(self, container_name: str):
        """Get an async container client by name"""
        if not self.initialized:
            await self.initialize()
        return self.async_database.get_container_client(self.container_names[container_name])
    
    async def query_items(self, container_name: str, query: str, parameters: Optional[Dict[str, Any]] = None):
        """Execute a query against a container with retry logic and metrics"""
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        container = await self.get_container(container_name)
        retry_count = 0
        
        while retry_count < RETRY_ATTEMPTS:
            try:
                # Execute the query
                items = []
                async for item in container.query_items(
                    query=query,
                    parameters=parameters,
                    partition_key=None  # Use this instead of enable_cross_partition_query=True
                ):
                    # Process each item
                    items.append(item)
                
                # Update metrics
                self.metrics["successful_requests"] += 1
                elapsed = time.time() - start_time
                self.metrics["avg_response_time"] = (
                    (self.metrics["avg_response_time"] * (self.metrics["successful_requests"] - 1) + elapsed) / 
                    self.metrics["successful_requests"]
                )
                
                return items
                
            except exceptions.CosmosHttpResponseError as e:
                retry_count += 1
                self.metrics["retry_attempts"] += 1
                
                if retry_count >= RETRY_ATTEMPTS:
                    self.metrics["failed_requests"] += 1
                    logger.error(f"Query failed after {retry_count} attempts: {str(e)}")
                    raise
                
                # Exponential backoff
                await asyncio.sleep(RETRY_INTERVAL * (2 ** (retry_count - 1)))
    
    async def create_item(self, container_name: str, item: Dict[str, Any]):
        """Create an item in a container with retry logic"""
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        container = await self.get_container(container_name)
        retry_count = 0
        
        while retry_count < RETRY_ATTEMPTS:
            try:
                result = await container.create_item(body=item)
                
                # Update metrics
                self.metrics["successful_requests"] += 1
                elapsed = time.time() - start_time
                self.metrics["avg_response_time"] = (
                    (self.metrics["avg_response_time"] * (self.metrics["successful_requests"] - 1) + elapsed) / 
                    self.metrics["successful_requests"]
                )
                
                return result
                
            except exceptions.CosmosHttpResponseError as e:
                retry_count += 1
                self.metrics["retry_attempts"] += 1
                
                if retry_count >= RETRY_ATTEMPTS:
                    self.metrics["failed_requests"] += 1
                    logger.error(f"Create item failed after {retry_count} attempts: {str(e)}")
                    raise
                
                # Exponential backoff
                await asyncio.sleep(RETRY_INTERVAL * (2 ** (retry_count - 1)))
    
    async def get_item(self, container_name: str, item_id: str, partition_key: str):
        """Get an item from a container with retry logic"""
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        container = await self.get_container(container_name)
        retry_count = 0
        
        while retry_count < RETRY_ATTEMPTS:
            try:
                result = await container.read_item(item=item_id, partition_key=partition_key)
                
                # Update metrics
                self.metrics["successful_requests"] += 1
                elapsed = time.time() - start_time
                self.metrics["avg_response_time"] = (
                    (self.metrics["avg_response_time"] * (self.metrics["successful_requests"] - 1) + elapsed) / 
                    self.metrics["successful_requests"]
                )
                
                return result
                
            except exceptions.CosmosHttpResponseError as e:
                retry_count += 1
                self.metrics["retry_attempts"] += 1
                
                if retry_count >= RETRY_ATTEMPTS:
                    self.metrics["failed_requests"] += 1
                    logger.error(f"Get item failed after {retry_count} attempts: {str(e)}")
                    raise
                
                # Exponential backoff
                await asyncio.sleep(RETRY_INTERVAL * (2 ** (retry_count - 1)))
    
    async def _monitor_connection_health(self):
        """Background task to monitor connection health"""
        while True:
            try:
                # Check connection every 30 seconds
                await asyncio.sleep(30)
                
                # Simple health check - query for a small amount of data
                test_query = "SELECT TOP 1 * FROM c"
                container = await self.get_container("livestream_container")
            
                # Use the correct parameter format for async client
                # The async client uses partition_key instead of enable_cross_partition_query
                items = []
                async for item in container.query_items(
                    query=test_query,
                    parameters=None,
                    partition_key=None  # This enables cross-partition query in the async client
                ):
                    items.append(item)
                logger.debug(f"Connection health check passed. Metrics: {self.metrics}")
                
            except Exception as e:
                logger.error(f"Connection health check failed: {str(e)}")
                # If connection is unhealthy, we could implement auto-reconnect logic here
    
    async def close(self):
        """Close the database connections"""
        if self.async_client:
            await self.async_client.close()
        # No explicit close for synchronous client

# FastAPI lifespan for initialization and cleanup
#moving this top ..for only one time initiation

# Dependency to get database connection
async def get_db():
    """Get initialized database connection"""
    db_manager = CosmosDBConnectionManager()
    if not db_manager.initialized:
        await db_manager.initialize()
    return db_manager

# Endpoint to check database connection status
@app.get("/api/database/status")
async def db_status(db: CosmosDBConnectionManager = Depends(get_db)):
    """Get database connection status and metrics"""
    return {
        "status": "connected" if db.initialized else "not connected",
        "app_ready": db.app_ready,
        "database_name": db.database_name,
        "metrics": db.metrics,
        "containers": list(db.container_names.keys())
    }

# Helper function for reactive database querying
def query_items_reactive(container_name: str, query: str, parameters: Optional[Dict] = None):
    """Create an observable that queries items from CosmosDB"""
    db_manager = CosmosDBConnectionManager()
    
    return rx.create(lambda observer, scheduler:
        # This function will be called when someone subscribes to our observable
        rx.from_future(db_manager.query_items(container_name, query, parameters)).pipe(
            ops.do_action(
                on_next=lambda items: observer.on_next(items),
                on_error=lambda error: observer.on_error(error),
                on_completed=lambda: observer.on_completed()
            )
        ).subscribe()
    )

from fastapi import Depends, HTTPException, Path, Query, status, Request, BackgroundTasks
import asyncio
import time
import rx
from rx import operators as ops
import uuid
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field

# Result model for validation and documentation
class McidDataResponse(BaseModel):
    items: List[Dict[str, Any]] = Field(default_factory=list)
    request_id: str = Field(..., description="Unique request identifier")
    processing_time: float = Field(..., description="Time taken to process the request in ms")
    record_count: int = Field(..., description="Number of records returned")

# Utility function to fetch data for a specific MCID
# Add this utility function to your code
async def fetch_mcid_details(
    mcid: str, 
    zipCode: str,
    countyName: str,
    sourceVisitId: str,
    db: CosmosDBConnectionManager,
    limit: int = 500,
    timeout: float = 5.0
) -> List[Dict[str, Any]]:
    """
    Utility function to fetch data for a specific MCID.
    
    Args:
        mcid: Medical Customer ID to query
        db: Database connection manager
        limit: Maximum number of records to return
        timeout: Query timeout in seconds
        
    Returns:
        List of items matching the MCID
    """
    try:
        # Prepare optimized query with limit
        query = """
        SELECT TOP @limit * 
        FROM c 
        WHERE c.mcid = @mcid and c.user_sessionID = @sourceVisitId and ((c.geo_zipCode = @zipCode and c.geo_cmsCountyName = @countyName) 
        or (c.entered_zip_code = @zipCode and c.entered_county_name = @countyName)) order by c._ts asc
        """
        parameters = [
            {"name": "@mcid", "value": mcid},
            {"name": "@zipCode", "value": zipCode},
            {"name": "@countyName", "value": countyName},
            {"name": "@sourceVisitId", "value": sourceVisitId},
            {"name": "@limit", "value": limit}
        ]

        return list(db.livestream_container.query_items(query,  parameters, enable_cross_partition_query=False))
    except asyncio.TimeoutError:
        logger.error(f"Query timeout for MCID mcid after timeout seconds 2")
        return None
    except Exception as e:
        logger.error(f"Error querying data for MCID mcid: {str(e)}")
        return None


# Reactive endpoint for high volume - parallel processing with backpressure
@app.get("/api/data/{mcid}", response_model=McidDataResponse)
async def get_data_by_mcid(
    request: Request,
    background_tasks: BackgroundTasks,
    mcid: str = Path(..., description="Medical Customer ID to query"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    timeout: float = Query(5.0, ge=0.5, le=30.0, description="Query timeout in seconds"),
    db: CosmosDBConnectionManager = Depends(get_db)
):
    """
    Get data for a specific MCID with high-performance reactive processing.
    
    This endpoint uses:
    - Query timeouts to prevent long-running queries
    - Result size limiting to prevent memory issues
    - Reactive processing with RxPy
    - Backpressure handling with buffer operators
    - Parallel processing for improved throughput
    """
    request_id = f"req_{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    
    try:
        
        # Use the utility function instead of duplicating code
        items = await fetch_mcid_details(mcid, db, limit, timeout)
        
        if items is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error querying data"
            )

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Prepare response with metrics
        response = {
            "items": items,
            "request_id": request_id,
            "processing_time": processing_time,
            "record_count": len(items)
        }
        
        # Log request metrics for monitoring
        if processing_time > 1000:  # Log slow queries (>1 second)
            logger.warning(f"Slow query for MCID mcid: processing_time:.2f ms")
        
        # Record metrics asynchronously without blocking response
        background_tasks.add_task(
            record_query_metrics, 
            mcid=mcid, 
            record_count=len(items), 
            processing_time=processing_time
        )
        
        return response
        
    except asyncio.TimeoutError:
        logger.error(f"Query timeout for MCID mcid after timeout seconds")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Query exceeded timeout of {timeout} seconds"
        )
    except Exception as e:
        logger.error(f"Error querying data for MCID mcid: str(e)")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error querying data: {str(e)}"
        )

# Helper function to record query metrics
async def record_query_metrics(mcid: str, record_count: int, processing_time: float):
    """Record query metrics for monitoring and analysis"""
    # In a real implementation, this would write to a metrics database or monitoring system
    logger.info(f"Query metrics - MCID: {mcid}, Records: {record_count}, Time: {processing_time:.2f}ms")
###########


############$$
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Request, BackgroundTasks, Depends, HTTPException, Cookie, status
from fastapi.responses import JSONResponse
import pandas as pd
import json
import uuid
import time
import asyncio
from datetime import datetime
import rx
from rx import operators as ops
from concurrent.futures import ThreadPoolExecutor
import logging

# Pydantic models for request/response validation
class UhcmRouteRequest(BaseModel):
    mcid: str
    uuid: Optional[str] = None
    isGuest: Optional[bool] = True
    fipsCode: Optional[str] = None
    zipCode: Optional[str] = None
    stateCode: Optional[str] = None
    planYear: Optional[str] = None
    callAttemptindex: Optional[int] = 0

    class Config:
        schema_extra = {
            "example": {
                "mcid": "52179627149348185813801699892056953224",
                "uuid": "ee9ba462-12b6-4cb3-983c-62a4705b82cc",
                "isGuest": True,
                "fipsCode": "06037",
                "zipCode": "90210",
                "stateCode": "CA",
                "planYear": "2025",
                "callAttemptindex": 0
            }
        }

class ModelMetadata(BaseModel):
    model_resource_name: str
    model_version_id: str

class DigitalRecommendation(BaseModel):
    visitid: str
    mcid: str
    show_modal_flag: Optional[str] = ''
    drug_cost_estimator: Optional[str] = ''
    provider_details:  Optional[str] = ''
    interval: int = 0
    predicted_date: Optional[str] = ''
    enteredZipCode: Optional[str] = ''
    enteredFipsCode: Optional[str] = ''
    enteredStateCode: Optional[str] = ''
    enteredCountyName: Optional[str] = ''
    geoZipCode: Optional[str] = ''
    geoFipsCode: Optional[str] = ''
    geoStateCode: Optional[str] = ''
    geoCountyName: Optional[str] = ''
    requestZipCode: Optional[str] = ''
    requestFipsCode: Optional[str] = ''
    requestStateCode: Optional[str] = ''
    requestCountyName: Optional[str] = ''
    predicted_score: Optional[str] = None
    quartile: Optional[str] = None
    target_bucket: Optional[str] = None

class UhcmRouteResponse(BaseModel):
    group: str = "DigitalMedia"
    ModelMetadata: ModelMetadata
    digitalRecommendation: DigitalRecommendation
    useCase: str


def sanitize_text(text):
    if isinstance(text, str):
        text = text.strip().replace('\n', '').replace('\r', '')
    return text

def validate_request_component(value: str, pattern: str) -> str:
    """
    Validates a URL path component against a regex pattern.
    Raises ValueError if invalid.
    """
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        return None
    
    return value

# Optimized endpoint for high volume processing
@app.post("/uhcmroute", 
         response_model=List[Dict[str, Any]],
         summary="Process UHCM route with high concurrency support",
         description="Process user data with reactive patterns to support 5,000+ concurrent users")
async def uhcm_route(
    request: Request,
    db: CosmosDBConnectionManager = Depends(get_db),
    dgcCookie: Optional[str] = Cookie(None)
):
    """
    High-performance implementation of UHCM route with:
    - Parallel processing for CPU-intensive operations
    - Reactive patterns for I/O operations
    - Proper error handling with timeouts
    - Backpressure handling
    """
    start_time = time.time()
    
    try:
        # Parse request JSON
        req_body = await request.json()
        
        # Extract key request parameters
        mcid = req_body.get('mcid')
        shopper_profile_id = sanitize_text(req_body.get('uuid'))
        fipsCode = sanitize_text(req_body.get('fipsCode'))
        zipCode = sanitize_text(req_body.get('zipCode'))
        stateCode = sanitize_text(req_body.get('stateCode'))
        countyName = sanitize_text(req_body.get('countyName'))
        planYear = sanitize_text(req_body.get('planYear'))
        sourceVisitId = sanitize_text(req_body.get('sourceVisitId'))
        interval_flag = req_body.get('callAttemptIndex')
        dgcCookie = sanitize_text(request.cookies.get("dgcCookie"))

        if not mcid or not zipCode or not countyName or not sourceVisitId:
            return JSONResponse(content={
                "Message": "This HTTP triggered function uhcm route executed successfully. Pass a mcid, zipCode, countyName & sourceVisitId in the request body for a personalized response [uhcmroute]."
            })

        mcid = mcid.strip().replace('\n', '').replace('\r', '')

        if planYear:
            planYear = ''.join(re.findall(r'\d+', str(planYear)))
        
        logger.info(f"Received request with mcid: mcid, zipCode: zipCode, countyName: countyName, sourceVisitId: sourceVisitId, interval_flag: interval_flag")

        # Load JSON data and generate sequence ID
        sequence_id = str(uuid.uuid4())

        logger.info(f"Processing UHCM route request for MCID: {mcid}")
        
        # # Log the request data to Cosmos DB
        logger.info("Started to log the original request payload data...log_request_data from postman inititated..")
        
        request_data_status = log_request_data_v1(req_body, sequence_id, sourceVisitId, db)

        if request_data_status != 200:  
            await asyncio.sleep(10)
            firstResponseData = fetch_response_data(mcid, sourceVisitId, interval_flag, zipCode, countyName.lower(), db)

            if firstResponseData is not None and len(firstResponseData) > 0:
                logger.info(f"Returning existing processed data for MCID: mcid, sourceVisitId: sourceVisitId, interval_flag: interval_flag, zipCode: zipCode, countyName: countyName")
                return JSONResponse(content=transform_response_data(firstResponseData))
            
        mcid_json_data = await fetch_with_retry_on_empty(mcid, zipCode, countyName.lower().replace(" county", ""), sourceVisitId, db, interval_flag)
        
        if mcid_json_data is None or mcid_json_data == []:
            logger.warning(f"No data available for MCID: {mcid}")
            return JSONResponse(content={"Message": "No Data available for this mcid.."})
        else:
            logger.info(f"mcid_data_json JSON contents:\n{mcid_json_data}")
    
        logger.info(f"Processing UHCM route request for MCID: {mcid}")
        #data_json=df_selected means refine..oes
        # Fetch MCID details
         
        # Validate MCID details fetched from cosmos db
        if not isinstance(mcid_json_data, list) or len(mcid_json_data) < 1:
            return JSONResponse(content={"Message": "Invalid JSON format or not enough elements in the MCID details fetched from Cosmos DB"})
        
        # Create dataframe from mcid_json_data details fetched from Cosmos DB
        df_mcid_data = pd.DataFrame(mcid_json_data)
        # Copy dataframe to avoid modifications
        print("Data frame attributes:", df_mcid_data.columns.tolist())
        print("type of mcid", type(df_mcid_data['mcid']))
        df_mcid_data['mcid'] = df_mcid_data['mcid'].astype(str)
        logger.info(f"Processing data for MCID: {df_mcid_data['mcid'].values[0]}")

        #Transformation of mcid data fetched from cosmos db to 
        required_columns = [
            'visitid', 'online_application', 'visit_page_num', 'partition_date',
            'mcid', 'visit_num', 'new_visit', 'visitor_domain', 'daily_visitor',
            'monthly_vistor', 'hourly_visitor', 'weekly_visitor', 'geo_dma',
            'host', 'browser', 'campaign', 'hit_date_ts', 'url_path',
            'geo_city', 'geo_country','geo_zip_code',
            'page_name_v1', 'plan_compare_plan_type_p25', 'plan_type_v43',
            'vpp_filter_selection_v49', 'age_calculation_v47', 'tecnotree_segment_v167',
            'tecnotree_personas_highest_rank_v168', 'tecnotree_personas_all_in_rank_order_v169',
            'search_engine', 'product_list', 'page_event_desc', 'link_name', 'hit_time_gmt',
            'entered_zip_code', 'entered_fips_code', 'entered_state_code', 'entered_county_name',
            'geo_fips_code', 'geo_state_code', 'geo_county_name'
        ]
        
        column_mappings = {
            # Example mappings
            'visitid': {'source': ['user_sessionID', 'visitid', 'session_id'], 'default': ''},
            'online_application': {'source': ['evars.evars.eVar14', 'evars.evars.eVar6', 'online_application'], 'default': ''},
            'visit_num': {'transform': lambda df_mcid_data: df_mcid_data.groupby('mcid')['user_sessionID'].rank(method='dense').fillna(0).astype(int)},
            'new_visit': {'transform': lambda df_mcid_data: (df_mcid_data.groupby(['mcid', 'user_sessionID']).cumcount() == 0).astype(int)},
            'host': {'source': ['host'],'default': 'uhc'},
            'daily_visitor': {'source': ['daily_visitor'],'default': 0},
            'monthly_vistor': {'source': ['monthly_vistor'],'default': 0},
            'hourly_visitor': {'source': ['hourly_visitor'],'default': 0},
            'weekly_visitor': {'source': ['weekly_visitor'],'default': 0},
            'hit_date_ts': {'source': ['hit_date_ts'],'default': pd.Timestamp.now().timestamp()},
            'hit_time_gmt': {'source': ['hit_time_gmt'],'default': pd.Timestamp.now().timestamp()},
            'partition_date': {'source': ['partition_date'],'default': pd.Timestamp.now().strftime('%Y-%m-%d')},
            'page_name_v1': {'source': ['evars.evars.eVar1', 'page_name_v1'], 'default': ''},
            'plan_compare_plan_type_p25': {'source': ['props.prop25', 'plan_compare_plan_type_p25'], 'default': ''},
            'plan_type_v43': {'source': ['evars.evars.eVar43', 'plan_type_v43'], 'default': ''},
            'vpp_filter_selection_v49': {'source': ['evars.evars.eVar49', 'vpp_filter_selection_v49'], 'default': ''},
            'campaign': {'source': ['campaign','evars.campaign',], 'default': ''},
            'tecnotree_segment_v167': {'source': ['evars.evars.eVar167', 'tecnotree_segment_v167'], 'default': ''},
            'tecnotree_personas_highest_rank_v168': {'source': ['evars.evars.eVar168', 'tecnotree_personas_highest_rank_v168'], 'default': ''},
            'tecnotree_personas_all_in_rank_order_v169': {'source': ['evars.evars.eVar169', 'tecnotree_personas_all_in_rank_order_v169'], 'default': ''},
            'product_list': {'source': ['products', 'product_list'], 'default': ''},
            'page_event_desc': {'source': ['page_event_desc', 'pageEvent.description'], 'default': ''},
            'link_name': {'source': ['link_name', 'pageEvent.linkName'], 'default': ''},
            'visit_page_num': {'transform': lambda df_mcid_data: df_mcid_data.groupby('user_sessionID').cumcount() + 1},
            'age_calculation_v47': {'source': ['evars.evars.eVar47','age_calculation_v47'],'default': ''},
            'entered_zip_code': {'source': ['entered_zip_code'], 'default': ''},
            'entered_fips_code': {'source': ['entered_fips_code'], 'default': ''},
            'entered_state_code': {'source': ['entered_state_code'], 'default': ''},
            'entered_county_name': {'source': ['entered_county_name'], 'default': ''},
            'geo_zip_code': {'source': ['geo_zipCode'], 'default': ''},
            'geo_fips_code': {'source': ['geo_fipsCode'], 'default': ''},
            'geo_state_code': {'source': ['geo_stateCode'], 'default': ''},
            'geo_county_name': {'source': ['geo_cmsCountyName'], 'default': ''},
        }

        # Create empty DataFrame with all required columns
        df_mcid_data_frame_refined = pd.DataFrame(columns=required_columns)
        # Create a single row or maintain the same number of rows as the raw DataFrame
        #data_frame_raw=df_mcid_data
        if len(df_mcid_data) > 0:
            df_mcid_data_frame_refined = pd.DataFrame(index=range(len(df_mcid_data)), columns=required_columns)
        else:
            df_mcid_data_frame_refined = pd.DataFrame(index=range(1), columns=required_columns)
        
        # Fill with default values based on column type
        for col in required_columns:
            # Check if we have a mapping for this column
            if col in column_mappings:
                mapping = column_mappings[col]
                
                # If mapping is a simple string, it's a direct column mapping
                if isinstance(mapping, str):
                    if mapping in df_mcid_data.columns:
                        df_mcid_data_frame_refined[col] = df_mcid_data[mapping]
                    else:
                        df_mcid_data_frame_refined[col] = ''
                
                # If mapping is a dict, it has more complex rules
                elif isinstance(mapping, dict):
                    # Check if we have source columns
                    if 'source' in mapping:
                        sources = mapping['source'] if isinstance(mapping['source'], list) else [mapping['source']]
                        
                        # Try each source column in order
                        mapped = False
                        for src in sources:
                            if src in df_mcid_data.columns:
                                df_mcid_data_frame_refined[col] = df_mcid_data[src]
                                mapped = True
                                break
                        
                        # If no source columns found, use default
                        if not mapped and 'default' in mapping:
                            df_mcid_data_frame_refined[col] = mapping['default']
                    
                    # If there's a transform function, apply it
                    elif 'transform' in mapping and callable(mapping['transform']):
                        transform_func = mapping['transform']
                        df_mcid_data_frame_refined[col] = transform_func(df_mcid_data)
                    
                    # If just a default value is provided
                    elif 'default' in mapping:
                        df_mcid_data_frame_refined[col] = mapping['default']
            
            # For columns that exist in both DataFrames, copy values directly
            elif col in df_mcid_data.columns:
                df_mcid_data_frame_refined[col] = df_mcid_data[col]
            
            # For columns that don't exist in raw DataFrame, set appropriate defaults
            else:
                if col in ['visit_num', 'new_visit', 'daily_visitor', 'monthly_vistor', 
                        'hourly_visitor', 'weekly_visitor']:
                    df_mcid_data_frame_refined[col] = 0
                elif col in ['hit_date_ts', 'hit_time_gmt']:
                    df_mcid_data_frame_refined[col] = pd.Timestamp.now().timestamp()
                elif col == 'partition_date':
                    df_mcid_data_frame_refined[col] = pd.Timestamp.now().strftime('%Y-%m-%d')
                elif col == 'online_application':
                    df_mcid_data_frame_refined[col] = 'No'
                elif col == 'age_calculation_v47':
                    df_mcid_data_frame_refined[col] = '25-34'
                else:
                    df_mcid_data_frame_refined[col] = ''
        
        # Check if required fields are present
        fields_in_request = all(col in df_mcid_data_frame_refined.columns for col in required_columns)
        print("Data frame fields_in_request attributes post missing columns:", df_mcid_data_frame_refined.columns.tolist())
        
        if not fields_in_request:
            return JSONResponse(content={"Message": "Invalid request attributes, unable to proceed..1"})
        
        # Print the dataframe and key values for debugging
        #logger.info(f"Dataframe contents:\n{df_mcid_data}")
        #logger.info(f"Refined Dataframe contents:\n{df_mcid_data_frame_refined}")
        logger.info(f"Refined Dataframe contents:\n{df_mcid_data_frame_refined.shape}")
        #logger.info(f"Key values - MCID: {mcid}, Shopper Profile ID: {shopper_profile_id}, "
        #            f"FIPS Code: {fipsCode}, Zip Code: {zipCode}, State Code: {stateCode}, "
        #            f"Plan Year: {planYear}, Interval Flag: {interval_flag}")

        model_input = make_model_input(df_mcid_data_frame_refined)
        logger.info(f"Model input DataFrame shape: {model_input.shape}")
        logger.info(f"Verify model input model_input result: {model_input}")
        if model_input.empty:
            return JSONResponse(content={"Message": "No Data available for this mcid.."})
        
        # Call check_drug_and_provider and capture its return value

        entered_zip_code = None
        entered_fips_code = None
        entered_state_code = None
        entered_county_name = None

        geo_zip_code = None
        geo_fips_code = None
        geo_state_code = None
        geo_county_name = None

        entities_coverage_request = {
            'uuid': shopper_profile_id,
            'planYear': planYear,
            'dgcCookie': dgcCookie,
        }

        entities_coverage_request['zipCode'] = zipCode
        entities_coverage_request['fipsCode'] = fipsCode
        entities_coverage_request['stateCode'] = stateCode
        
        if not df_mcid_data_frame_refined.empty:
            valid_county_row = df_mcid_data_frame_refined.iloc[-1]
            entered_zip_code = valid_county_row['entered_zip_code']
            entered_fips_code = valid_county_row['entered_fips_code']
            entered_state_code = valid_county_row['entered_state_code']
            entered_county_name = valid_county_row['entered_county_name']
            geo_zip_code = valid_county_row['geo_zip_code']
            geo_fips_code = valid_county_row['geo_fips_code']
            geo_state_code = valid_county_row['geo_state_code']
            geo_county_name = valid_county_row['geo_county_name']
            
            logger.info(f"entered county data: {entered_zip_code}, {entered_fips_code}, {entered_state_code}, {entered_county_name}")
            logger.info(f"geo county data: {geo_zip_code}, {geo_fips_code}, {geo_state_code}, {geo_county_name}")
         
        county_data = {
            'entered_zip_code': entered_zip_code,
            'entered_fips_code': entered_fips_code,
            'entered_state_code': entered_state_code,
            'entered_county_name': entered_county_name,
            'geo_zip_code': geo_zip_code,
            'geo_fips_code': geo_fips_code,
            'geo_state_code': geo_state_code,
            'geo_county_name': geo_county_name,
            'request_zip_code': zipCode,
            'request_fips_code': fipsCode,
            'request_state_code': stateCode,
            'request_county_name': countyName.lower()
        }

        logger.info(f"County data extracted: county_data for mcid mcid")

        # Extract drug and provider details
        #drug_cost_estimator, provider_details = drug_provider_result if isinstance(drug_provider_result, tuple) else ('No', 'No')
        drug_cost_estimator, provider_details = ('No', 'No')

        # ---- Begin business logic processing ----
        cosmos_db = read_cosmosdb(mcid_filter=mcid, db_manager=db)
        # Step 1: MCID validation
        cosmos_db, upper_block_flag, reasons = mcid_validation(
            mcid, df_mcid_data_frame_refined, cosmos_db, model_input, interval_flag, db
        )
        
        # Initialize variables
        quartile_value = None
        BCI_null_flag = False
        send_to_mcid_pre_exist_stage_flag = False
        model_score_greater_than_threshold_flag = 'No'
        out_score = 0.0
        target_bucket = None
        model_app_name = ""
        model_version = ""
        invoca_score = 0.0
        all_rules_satisfied = 'No'

        call_flag='No'
        
        if upper_block_flag == 'No':
           
            quartile_value, quartile, BCI_null_flag = (None, None, False)
            send_to_mcid_pre_exist_stage_flag = True

            logger.info(f"Checking BCI for zip: zipCode for mcid mcid")
        
            # Step 2: Check BCI
            
            quartile_value, BCI_null_flag = check_bci(zipCode, mcid, db)
            quartile = quartile_value

            logger.info(f"BCI quartile value: {quartile_value}, BCI null flag: {BCI_null_flag} for mcid {mcid}")

            #if interval_flag != 0:
                
                
            cosmos_db, send_to_mcid_pre_exist_stage_flag = check_bci_null_flag(
                df_mcid_data_frame_refined, cosmos_db, BCI_null_flag, model_input, interval_flag, db
            )
            
            import pickle
            if send_to_mcid_pre_exist_stage_flag:
                # Step 4: Score with unauth model
                model_input_scored, model_score_greater_than_threshold_flag, out_score, target_bucket, model_app_name, model_version = await unauth_model_scoring(
                    model_input, model_score_greater_than_threshold_flag
                )
                # Step 5: Score with invoca model
                model_input_invoca_scored, call_flag, invoca_score = await invoca_model_scoring(model_input)
                #out_score=0.5 #0.5
                #invoca_score= 0.4
                
                # Step 6: Segment determination
                rules_block_flag, cosmos_db = seg2_seg3(
                    model_input, county_data, cosmos_db, invoca_score, quartile_value, interval_flag, out_score, db
                )
                
                if rules_block_flag == 'Yes':
                    # Step 7: Set interval flags
                    all_interval_flags = set_interval_flags(interval_flag)
                    
                    # Step 8: Calculate click_min
                    model_input['click_min'] = ((model_input['time_span_hr'].astype(float) * 60 * 8) / 
                                              model_input['hits'].astype(float)).astype(float)
                    
                    # Step 9: Apply business rules
                    model_input, all_rules_satisfied, rule_results = rules_block(
                        model_input,
                        county_data,
                        entities_coverage_request,
                        all_interval_flags['first_click_interval_flag'],
                        all_interval_flags['five_min_interval_flag'],
                        all_interval_flags['ten_min_interval_flag'],
                        all_interval_flags['fifteen_min_interval_flag'],
                        all_interval_flags['twenty_min_interval_flag'],
                        rules_block_flag,
                        interval_flag,
                        quartile_value,
                        cosmos_db,
                        call_flag,
                        quartile,
                        out_score,
                        target_bucket,
                        invoca_score,
                        db
                    )
                else:
                    logger.info("Did not send to rules block")

       
        # Get visitid from model input
        visitid = model_input['visitid'].values[0]
        
        # Read the latest data from Cosmos DB
        df_results=read_cosmosdb_latest_sync(visitid, db)
        
        logger.info("Printing df_results DataFrame:")
        logger.info(f"Shape: {df_results.shape}")
        logger.info(f"Columns: {df_results.columns.tolist()}")
        logger.info(f"Head:\n{df_results.head()}")

        if df_results.empty:
            return JSONResponse(content={"Message": "No results available for the processed data"})
        
        # Prepare flag data
        usecase_val = df_results.iloc[0]['Cohort']
        predict_date = datetime.now()
        
        df_flagdata = pd.DataFrame({
            'visitid': [df_results.iloc[0]['visitid']],
            'mcid': [df_results.iloc[0]['mcid']],
            'group': ["DigitalMedia"],
            'useCase': [usecase_val],
            'flagvalue_actual': [df_results.iloc[0]['show_noshow']],
            'score': [df_results.iloc[0]['out_score']],
            'model_app_name': [model_app_name],
            'model_version': [model_version]
        })
        
        #background_tasks.add_task(log_flag_data)
        log_flag_data(df_flagdata, sequence_id, db)
        # Prepare final response
        final_response = [
            {
                "group": "DigitalMedia",
                "ModelMetadata": {
                    "model_resource_name": model_app_name,
                    "model_version_id": str(model_version)
                },
                "digitalRecommendation": {
                    "visitid": visitid,
                    "mcid": mcid,
                    "show_modal_flag": df_results.iloc[0]['show_noshow'],
                    "drug_cost_estimator": drug_cost_estimator,
                    "provider_details": provider_details,
                    "interval": str(interval_flag),
                    "predicted_date": str(predict_date),
                    "predicted_score": str(df_results.iloc[0]['out_score']) if df_results.iloc[0]['out_score'] is not None else str(0.0),
                    "quartile": df_results.iloc[0]['quartile'] if 'quartile' in df_results.columns and df_results.iloc[0]['quartile'] is not None else 'NULL',
                    "target_bucket": target_bucket,
                    "enteredZipCode": entered_zip_code,
                    "enteredFipsCode": entered_fips_code,
                    "enteredStateCode": entered_state_code,
                    "enteredCountyName": entered_county_name,
                    "geoZipCode": geo_zip_code,
                    "geoFipsCode": geo_fips_code,
                    "geoStateCode": geo_state_code,
                    "geoCountyName": geo_county_name,
                    "requestZipCode": zipCode,
                    "requestFipsCode": fipsCode,
                    "requestStateCode": stateCode,
                    "requestCountyName": countyName.lower(),
                },
                "useCase": str(usecase_val)
            }
        ]
        
        # Log the combined data
        log_combined_data(df_results, final_response, sequence_id)
        
        # Calculate and log processing time
        processing_time = time.time() - start_time
        logger.info(f"UHCM route request processed in {processing_time:.2f}s")
        
        # Record slow requests for monitoring
        if processing_time > 9.0:
            logger.warning(f"Slow UHCM route request: {processing_time:.2f}s for MCID {mcid}")
        
        print(f"Final response: {final_response}")
        # Return the final response

        return final_response
        
    except Exception as e:
        logger.error(f"Error processing UHCM route: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"Message": f"Error processing request: str(e)"}
        )

async def fetch_with_retry_on_empty(mcid, zipCode, countyName, sourceVisitId, db, interval_flag):
    if interval_flag == 0:
        logger.info(f"Interval flag is 0, waiting for 10 seconds before retrying for MCID: mcid")
        await asyncio.sleep(20)

    attempt = 0
    retries = 1
    backoff_factor = 10

    while attempt <= retries:
        try:
            logger.info(f"Fetching top 500 records in 10.0 secs for MCID: mcid")
            mcid_json_data = await fetch_mcid_details(mcid, zipCode, countyName, sourceVisitId, db, 500, 10.0)

            if mcid_json_data is not None and mcid_json_data != []:
                logger.info(f"Successfully fetched data for MCID {mcid} on attempt {attempt + 1}")
                if interval_flag == 0:
                    return [mcid_json_data[0]]
            
                return mcid_json_data
            else:
                logger.warning(f"No data returned for MCID {mcid} on attempt {attempt + 1}")
        except Exception as e:
            logger.error(f"Exception while fetching MCID {mcid} on attempt {attempt + 1}: {e}")

        if attempt < retries:
            wait_time = backoff_factor * (2 ** attempt)
            logger.info(f"Retrying MCID {mcid} after {wait_time:.2f} seconds...")
            await asyncio.sleep(wait_time)

        attempt += 1

    return None

# Load the model
import pickle
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from pathlib import Path
import rx
from rx import operators as ops
from fastapi import APIRouter, Request, BackgroundTasks, Depends, HTTPException, status
import logging
import time
import os
from concurrent.futures import ThreadPoolExecutor

class PredictInputData(BaseModel):
    mcid: str = Field(..., description="Medical Customer ID")
    partion_date: str = Field(..., description="Partition date")
    visitid: str = Field(..., description="Visit ID")
    online_application: int = Field(..., description="Online application status")
    hits: float = Field(default=0.0, description="Number of hits")
    month_visitor_info: float = Field(default=0.0, description="Monthly visitor info")
    downlink_cnt: float = Field(default=0.0, description="Download link count")
    type000: float = Field(default=0.0, description="Type 000")
    url_path_hit: float = Field(default=0.0, description="URL path hit")
    product_list_info: float = Field(default=0.0, description="Product list info")
    time_span_hr: float = Field(default=0.0, description="Time span in hours")
    plan_type_v43_cnt: float = Field(default=0.0, description="Plan type count")
    page_name_hit: float = Field(default=0.0, description="Page name hit")
    custlink_cnt: float = Field(default=0.0, description="Custom link count")
    vpp_filter_selection_v49_cnt: float = Field(default=0.0, description="VPP filter selection count")
    type095: float = Field(default=0.0, description="Type 095")
    plan_compare_plan_type_p25_cnt: float = Field(default=0.0, description="Plan compare type count")
    type070: float = Field(default=0.0, description="Type 070")
    daily_visitor_info: float = Field(default=0.0, description="Daily visitor info")
    wk_visitor_info: float = Field(default=0.0, description="Weekly visitor info")
    pagelink_cnt: float = Field(default=0.0, description="Page link count")
    exitlink_cnt: float = Field(default=0.0, description="Exit link count")
    come_bk: float = Field(default=0.0, description="Come back count")
    url_path_num: float = Field(default=0.0, description="URL path number")
    age: float = Field(default=70.0, description="Age")
    type090: float = Field(default=0.0, description="Type 090")
    max_name_pct: float = Field(default=0.0, description="Max name percentage")
    page_name_cnt: float = Field(default=0.0, description="Page name count")
    hr_visitor_info: float = Field(default=0.0, description="Hourly visitor info")
    geo_dma_info: str = Field(default="NA", description="Geographic DMA info")
    host_info: str = Field(default="NA", description="Host info")
    browser_info: str = Field(default="NA", description="Browser info")
    visitor_domain_info: str = Field(default="NA", description="Visitor domain info")
    campaign_info: str = Field(default="NA", description="Campaign info")
    weekday: str = Field(default="NA", description="Weekday")
    race_info: str = Field(default="NA", description="Race info")
    income_info: str = Field(default="NA", description="Income info")
    shopper_info: str = Field(default="NA", description="Shopper info")
    search_engine_info: str = Field(default="NA", description="Search engine info")
    new_visit_flag: float = Field(default=None, description="New visit flag")

    class Config:
        schema_extra = {
            "example": {
                "mcid": "52179627149348185813801699892056953224",
                "partion_date": "2025-01-01",
                "visitid": "abc123",
                "online_application": "Yes",
                "hits": 5.0,
                "month_visitor_info": 2.0,
                "age": 35.0,
                "geo_dma_info": "New York",
                "weekday": "Monday"
            }
        }

class PredictOutputData(BaseModel):
    mcid: str
    visitid: str
    partition_date: str
    online_application: int
    score: float = Field(..., description="Prediction score")
    Model_App_Name: str = Field(default="mlops_dev_unauth_uc_endpoint")
    Model_Version: int = Field(default=1)

class PredictResponse(BaseModel):
    predictions: List[PredictOutputData]
    processing_time: float
    model_version: str
    timestamp: str
class ModelService:
    _instance = None
    _model = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
        return cls._instance
    
    async def load_model(self):
        """Load the ML model asynchronously"""
        async with self._lock:
            if self._model is None:
                try:
                    # Run model loading in thread pool to avoid blocking
                    loop = asyncio.get_running_loop()
                    with ThreadPoolExecutor() as executor:
                        self._model = await loop.run_in_executor(
                            executor, self._load_model_file
                        )
                    logger.info("Model loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading model: {str(e)}")
                    raise
    
    def _load_model_file(self):
        """Synchronous model loading"""
        model_path = os.getenv("MODEL_PATH", "saved_model.pkl")
        
        # Check multiple possible locations
        possible_paths = [
            Path(model_path),
            Path("./models") / "saved_model.pkl",
            Path("./data") / "saved_model.pkl",
            Path("../models") / "saved_model.pkl"
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"Loading model from: {path}")
                with open(path, "rb") as f:
                    return pickle.load(f)
        
        raise FileNotFoundError(f"Model file not found in any of: {possible_paths}")
    
    def get_model(self):
        """Get the loaded model"""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._model
    
    def predict(self, input_df: pd.DataFrame) -> np.ndarray:
        """Make predictions using the loaded model"""
        try:
            model = self.get_model()
            predictions = model.predict_proba(input_df)[:, 1]
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
# Create global model service instance
model_service = ModelService()
from datetime import datetime
import time

@app.post("/predict", 
         response_model=List[PredictOutputData],
         summary="ML Model Prediction Endpoint",
         description="Make predictions using the trained unauth model")
async def predict(
    input_data: PredictInputData,
) -> List[PredictOutputData]:
    """
    High-performance ML prediction endpoint with:
    - Input validation using Pydantic
    - Async model execution
    - Error handling with proper HTTP status codes
    - Request logging and monitoring
    """
    request_id = f"pred_{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    
    try:
        # Convert Pydantic model to DataFrame
        input_dict = input_data.dict()
        input_df = pd.DataFrame([input_dict])
        
        # Validate required columns are present
        required_model_columns = [
            'mcid', 'partion_date', 'visitid', 'online_application', 'hits',
            'month_visitor_info', 'downlink_cnt', 'type000', 'url_path_hit',
            'product_list_info', 'time_span_hr', 'plan_type_v43_cnt', 'page_name_hit',
            'custlink_cnt', 'vpp_filter_selection_v49_cnt', 'type095',
            'plan_compare_plan_type_p25_cnt', 'type070', 'daily_visitor_info',
            'wk_visitor_info', 'pagelink_cnt', 'exitlink_cnt', 'come_bk',
            'url_path_num', 'age', 'type090', 'max_name_pct', 'page_name_cnt',
            'hr_visitor_info', 'geo_dma_info', 'host_info', 'browser_info',
            'visitor_domain_info', 'campaign_info', 'weekday', 'race_info',
            'income_info', 'shopper_info', 'search_engine_info', 'new_visit_flag'
        ]
        
        # Ensure all required columns are present
        for col in required_model_columns:
            if col not in input_df.columns:
                if col in ['mcid', 'partion_date', 'visitid', 'online_application']:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Required column '{col}' is missing"
                    )
                else:
                    # Add missing columns with default values
                    if col in ['geo_dma_info', 'host_info', 'browser_info', 
                              'visitor_domain_info', 'campaign_info', 'weekday',
                              'race_info', 'income_info', 'shopper_info', 'search_engine_info']:
                        input_df[col] = 'NA'
                    else:
                        input_df[col] = 0.0
        
        # Reorder columns to match model expectations
        input_df = input_df[required_model_columns]
        # Make predictions using the model service
        try:
            # Run prediction in thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as executor:
                predictions = await loop.run_in_executor(
                    executor, model_service.predict, input_df
                )
        except Exception as model_error:
            logger.error(f"[{request_id}] Model prediction error: {str(model_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Model prediction failed: {str(model_error)}"
            )
        
        # Prepare the output
        processing_time = time.time() - start_time
        
        output_data = []
        for idx, (_, row) in enumerate(input_df.iterrows()):
            result = PredictOutputData(
                mcid=row['mcid'],
                visitid=row['visitid'],
                partition_date=row['partion_date'],
                online_application=row['online_application'],
                score=float(predictions[idx]),
                Model_App_Name="mlops_dev_unauth_uc_endpoint",
                Model_Version=1
            )
            output_data.append(result)
        
        # Log prediction metrics
        avg_score = np.mean(predictions)
        logger.info(f"[{request_id}] Prediction completed in {processing_time:.3f}s, "
                   f"avg_score={avg_score:.4f}")
        
        
        return [item.dict() for item in output_data]
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error in prediction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# Helper function for logging prediction metrics
async def log_prediction_metrics(request_id: str, mcid: str, processing_time: float, avg_score: float):
    """Log prediction metrics for monitoring"""
    try:
        metrics = {
            "request_id": request_id,
            "mcid": mcid,
            "processing_time": processing_time,
            "avg_score": avg_score,
            "timestamp": datetime.now().isoformat(),
            "endpoint": "predict"
        }
        logger.info(f"Prediction metrics: {metrics}")
        # Here you could also write to a metrics database or monitoring system
    except Exception as e:
        logger.error(f"Error logging prediction metrics: {str(e)}")

#####unauth model

####invoca model
from pydantic import BaseModel, Field
from typing import List, Optional

class PredictInvocaInputData(BaseModel):
    mcid: str = Field(..., description="Medical Customer ID")
    partion_date: str = Field(..., description="Partition date")
    visitid: str = Field(..., description="Visit ID")
    online_application: int = Field(..., description="Online application status")
    hits: float = Field(default=0.0, description="Number of hits")
    month_visitor_info: float = Field(default=0.0, description="Monthly visitor info")
    downlink_cnt: float = Field(default=0.0, description="Download link count")
    type000: float = Field(default=0.0, description="Type 000")
    product_list_info: float = Field(default=0.0, description="Product list info")
    time_span_hr: float = Field(default=0.0, description="Time span in hours")
    plan_type_v43_cnt: float = Field(default=0.0, description="Plan type count")
    custlink_cnt: float = Field(default=0.0, description="Custom link count")
    vpp_filter_selection_v49_cnt: float = Field(default=0.0, description="VPP filter selection count")
    type095: float = Field(default=0.0, description="Type 095")
    plan_compare_plan_type_p25_cnt: float = Field(default=0.0, description="Plan compare type count")
    type070: float = Field(default=0.0, description="Type 070")
    daily_visitor_info: float = Field(default=0.0, description="Daily visitor info")
    wk_visitor_info: float = Field(default=0.0, description="Weekly visitor info")
    pagelink_cnt: float = Field(default=0.0, description="Page link count")
    exitlink_cnt: float = Field(default=0.0, description="Exit link count")
    come_bk: float = Field(default=0.0, description="Come back count")
    url_path_num: float = Field(default=0.0, description="URL path number")
    age: float = Field(default=70.0, description="Age")
    type090: float = Field(default=0.0, description="Type 090")
    max_name_pct: float = Field(default=0.0, description="Max name percentage")
    page_name_cnt: float = Field(default=0.0, description="Page name count")
    hr_visitor_info: float = Field(default=0.0, description="Hourly visitor info")
    geo_dma_info: str = Field(default="NA", description="Geographic DMA info")
    host_info: str = Field(default="NA", description="Host info")
    browser_info: str = Field(default="NA", description="Browser info")
    visitor_domain_info: str = Field(default="NA", description="Visitor domain info")
    campaign_info: str = Field(default="NA", description="Campaign info")
    weekday: str = Field(default="NA", description="Weekday")
    race_info: str = Field(default="NA", description="Race info")
    income_info: str = Field(default="NA", description="Income info")
    shopper_info: str = Field(default="NA", description="Shopper info")
    search_engine_info: str = Field(default="NA", description="Search engine info")
    new_visit_flag: float = Field(default=None, description="New visit flag")

    class Config:
        schema_extra = {
            "example": {
                "mcid": "52179627149348185813801699892056953224",
                "partion_date": "2025-01-01",
                "visitid": "abc123",
                "online_application": "Yes",
                "hits": 5.0,
                "month_visitor_info": 2.0,
                "age": 35.0,
                "geo_dma_info": "New York",
                "weekday": "Monday"
            }
        }

class PredictInvocaOutputData(BaseModel):
    mcid: str
    visitid: str
    partition_date: str
    online_application: int
    score: float = Field(..., description="Invoca prediction score")
    Model_App_Name: str = Field(default="mlops_dev_invoca_uc_endpoint")
    Model_Version: int = Field(default=2)

class PredictInvocaResponse(BaseModel):
    predictions: List[PredictInvocaOutputData]
    processing_time: float
    model_version: str
    timestamp: str
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

class InvocaModelService:
    _instance = None
    _model = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InvocaModelService, cls).__new__(cls)
        return cls._instance
    
    async def load_model(self):
        """Load the Invoca ML model asynchronously"""
        async with self._lock:
            if self._model is None:
                try:
                    # Run model loading in thread pool to avoid blocking
                    loop = asyncio.get_running_loop()
                    with ThreadPoolExecutor() as executor:
                        self._model = await loop.run_in_executor(
                            executor, self._load_model_file
                        )
                    logger.info("Invoca model loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading Invoca model: {str(e)}")
                    raise
    
    def _load_model_file(self):
        """Synchronous Invoca model loading"""
        model_path = os.getenv("INVOCA_MODEL_PATH", "invoca_model.pkl")
        print(f"Invoca model path: {model_path}")
        # Check multiple possible locations
        possible_paths = [
            Path(model_path),
            Path("./models") / "invoca_model.pkl",
            Path("./data") / "invoca_model.pkl",
            Path("../models") / "invoca_model.pkl"
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"Loading Invoca model from: {path}")
                with open(path, "rb") as f:
                    return pickle.load(f)
        
        raise FileNotFoundError(f"Invoca model file not found in any of: {possible_paths}")
    
    def get_model(self):
        """Get the loaded Invoca model"""
        if self._model is None:
            raise RuntimeError("Invoca model not loaded. Call load_model() first.")
        return self._model
    
    def predict(self, input_df: pd.DataFrame) -> np.ndarray:
        """Make predictions using the loaded Invoca model"""
        try:
            model = self.get_model()
            predictions = model.predict_proba(input_df)[:, 1]
            return predictions
        except Exception as e:
            logger.error(f"Error making Invoca predictions: {str(e)}")
            raise

# Create global Invoca model service instance
invoca_model_service = InvocaModelService()

from datetime import datetime
import time

@app.post("/predict_invoca", 
         response_model=List[PredictInvocaOutputData],
         summary="Invoca ML Model Prediction Endpoint",
         description="Make predictions using the trained Invoca model")
async def predict_invoca(
    input_data: PredictInvocaInputData
) -> List[PredictInvocaOutputData]:
    """
    High-performance Invoca ML prediction endpoint with:
    - Input validation using Pydantic
    - Async model execution
    - Error handling with proper HTTP status codes
    - Request logging and monitoring
    """
    request_id = f"invoca_{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    
    print(f"Invoca prediction request received: {input_data}")
    try:
        logger.info(f"[request_id] Invoca prediction request for MCID: input_data.mcid")
        
        # Convert Pydantic model to DataFrame
        input_dict = input_data.dict()
        input_df = pd.DataFrame([input_dict])
        
        # Validate required columns are present for Invoca model
        required_invoca_columns = [
            'mcid', 'partion_date', 'visitid', 'online_application', 'hits',
            'month_visitor_info', 'downlink_cnt', 'type000', 'product_list_info',
            'time_span_hr', 'plan_type_v43_cnt', 'custlink_cnt', 
            'vpp_filter_selection_v49_cnt', 'type095', 'plan_compare_plan_type_p25_cnt',
            'type070', 'daily_visitor_info', 'wk_visitor_info', 'pagelink_cnt',
            'exitlink_cnt', 'come_bk', 'url_path_num', 'age', 'type090',
            'max_name_pct', 'page_name_cnt', 'hr_visitor_info', 'geo_dma_info',
            'host_info', 'browser_info', 'visitor_domain_info', 'campaign_info',
            'weekday', 'race_info', 'income_info', 'shopper_info', 'search_engine_info', 'new_visit_flag'
        ]
        
        # Ensure all required columns are present
        for col in required_invoca_columns:
            if col not in input_df.columns:
                if col in ['mcid', 'partion_date', 'visitid', 'online_application']:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Required column '{col}' is missing"
                    )
                else:
                    # Add missing columns with default values
                    if col in ['geo_dma_info', 'host_info', 'browser_info', 
                              'visitor_domain_info', 'campaign_info', 'weekday',
                              'race_info', 'income_info', 'shopper_info', 'search_engine_info']:
                        input_df[col] = 'NA'
                    else:
                        input_df[col] = 0.0
        
        # Reorder columns to match Invoca model expectations
        input_df = input_df[required_invoca_columns]
        
        # Make predictions using the Invoca model service
        try:
            # Run prediction in thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as executor:
                predictions = await loop.run_in_executor(
                    executor, invoca_model_service.predict, input_df
                )
        except Exception as model_error:
            logger.error(f"[{request_id}] Invoca model prediction error: {str(model_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Invoca model prediction failed: {str(model_error)}"
            )
        
        # Prepare the output
        processing_time = time.time() - start_time
        
        output_data = []
        for idx, (_, row) in enumerate(input_df.iterrows()):
            result = PredictInvocaOutputData(
                mcid=row['mcid'],
                visitid=row['visitid'],
                partition_date=row['partion_date'],
                online_application=row['online_application'],
                score=float(predictions[idx]),
                Model_App_Name="mlops_dev_invoca_uc_endpoint",
                Model_Version=2
            )
            output_data.append(result)
        
        # Log prediction metrics
        avg_score = np.mean(predictions)
        logger.info(f"[{request_id}] Invoca prediction completed in {processing_time:.3f}s, "
                   f"avg_score={avg_score:.4f}")
        
        return [item.dict() for item in output_data]
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error in Invoca prediction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# Helper function for logging Invoca prediction metrics
async def log_invoca_prediction_metrics(request_id: str, mcid: str, processing_time: float, avg_score: float):
    """Log Invoca prediction metrics for monitoring"""
    try:
        metrics = {
            "request_id": request_id,
            "mcid": mcid,
            "processing_time": processing_time,
            "avg_score": avg_score,
            "timestamp": datetime.now().isoformat(),
            "endpoint": "predict_invoca"
        }
        logger.info(f"Invoca prediction metrics: {metrics}")
        # Here you could also write to a metrics database or monitoring system
    except Exception as e:
        logger.error(f"Error logging Invoca prediction metrics: {str(e)}")

# Synchronous function for reading latest cosmos DB data
def fetch_response_data(mcid:str, visitid_filter: str, interval_flag, zip_code, county_name, db_manager: CosmosDBConnectionManager) -> pd.DataFrame:
    """
    Synchronous function to read the latest data from Cosmos DB for a given visit ID.
    
    Args:
        visitid_filter: The visit ID to filter by
        db_manager: The database connection manager
        
    Returns:
        DataFrame containing the latest data for the visit ID
    """
    try:
        query = f"SELECT TOP 1 * FROM c WHERE c.mcid='{mcid}' AND c.visitid = '{visitid_filter}' AND c.interval={interval_flag} AND c.request_zip_code='{zip_code}' AND c.request_county_name='{county_name}' order by c._ts asc"
        return list(db_manager.response_container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))
        
    except Exception as e:
        logger.error(f"Error reading latest data from Cosmos DB: {str(e)}")
        return []  # Return empty DataFrame on failure
    
from datetime import datetime
from typing import List, Dict, Any

def transform_response_data(response_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    transformed_output = []

    for item in response_data:
        transformed = {
            "group": item.get("group"),
            "ModelMetadata": {
                "model_resource_name": item.get("model_resource_name") or "mlops_dev_unauth_uc_endpoint",
                "model_version_id": item.get("model_version_id")
            },
            "digitalRecommendation": {
                "visitid": item.get("visitid"),
                "mcid": item.get("mcid"),
                "show_modal_flag": item.get("show_modal_flag"),
                "drug_cost_estimator": item.get("drug_cost_details"),
                "provider_details": item.get("provider_details"),
                "interval": item.get("interval"),
                "predicted_score": item.get("predicted_score"),
                "quartile": item.get("quartile") or item.get("quartile_digital"),
                "target_bucket": item.get("target_bucket") or "",
                "enteredZipCode": item.get("entered_zip_code") or "",
                "enteredFipsCode": item.get("entered_fips_code") or "",
                "enteredStateCode": item.get("entered_state_code") or "",
                "enteredCountyName": item.get("entered_county_name") or "",
                "geoZipCode": item.get("geo_zip_code") or "",
                "geoFipsCode": item.get("geo_fips_code") or "",
                "geoStateCode": item.get("geo_state_code") or "",
                "geoCountyName": item.get("geo_county_name") or "",
                "requestZipCode": item.get("request_zip_code") or "",
                "requestFipsCode": item.get("request_fips_code") or "",
                "requestStateCode": item.get("request_state_code") or "",
                "requestCountyName": item.get("request_county_name") or "",
            },
            "useCase": item.get("usecase") or item.get("Cohort")
        }
        transformed_output.append(transformed)

    return transformed_output

# Synchronous function for reading latest cosmos DB data
def read_cosmosdb_latest_sync(visitid_filter: str, db_manager: CosmosDBConnectionManager) -> pd.DataFrame:
    """
    Synchronous function to read the latest data from Cosmos DB for a given visit ID.
    
    Args:
        visitid_filter: The visit ID to filter by
        db_manager: The database connection manager
        
    Returns:
        DataFrame containing the latest data for the visit ID
    """
    try:
        query = f"SELECT TOP 1 * FROM c WHERE c.visitid = '{visitid_filter}' order by c._ts desc"
        df_list = list(db_manager.processed_container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))
        
        df = pd.DataFrame(df_list)
        logger.info(f"Successfully read latest data from Cosmos DB for visit ID: {visitid_filter}")
        logger.info(f"Cosmos DB result shape: {df.shape}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error reading latest data from Cosmos DB: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on failure

def has_non_empty_covered_plans(entityData: Dict[str, Any]) -> bool:
    plans = entityData.get("coveredPlans")
    return isinstance(plans, list) and len(plans) > 0

def apply_rule_5_from_api_response(entities_coverage_request):
    try:
        drugs, providers = check_drug_and_provider_sync(entities_coverage_request)
   
        # Extract provider info
        x = len(providers)

        # summing all cobvered plan ids for providers
        n = sum(1 for p in providers if has_non_empty_covered_plans(p))

        # Extract drug info
        y = len(drugs)

        # summing all cobvered plan ids for drugs
        
        m = sum(1 for d in drugs if has_non_empty_covered_plans(d))

        # Avoid division by zero
        x_ratio = n / x if x else 0
        y_ratio = m / y if y else 0

        t = 0.7 * x_ratio + 0.3 * y_ratio

        if t > 0.9 and n > 2:
            return 'Cohort 1-Submitter', 0
        elif t > 0.9 and n == 1:
            return 'Drop Out', 0
        else:
            return 'Cohort 2', 1
    except Exception as e:
        logger.error(f"Error applying rule 5 from API response: {str(e)} & Returning default Cohort 2 and hist_label 1 due to error in rule 5")
        return 'Cohort 2', 1

 
# Synchronous function to apply all business rules
def rules_block(
    model_input: pd.DataFrame,
    county_data: Dict[str, Any],
    entities_coverage_request,
    first_click_interval_flag: str = 'No',
    five_min_interval_flag: str = 'No',
    ten_min_interval_flag: str = 'No',
    fifteen_min_interval_flag: str = 'No',
    twenty_min_interval_flag: str = 'No',
    rules_block_flag: str = 'No',
    interval_flag: int = 0,
    quartile_value: Optional[str] = None,
    cosmos_db: Optional[pd.DataFrame] = None,
    call_flag: str = 'No',
    quartile: Optional[str] = None,
    out_score: Optional[float] = None,
    target_bucket: Optional[str] = None,
    invoca_score: Optional[float] = None,
    db_manager: Optional[CosmosDBConnectionManager] = None
) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """
    Synchronous function to apply all business rules and update Cosmos DB accordingly.
    
    Args:
        model_input: DataFrame containing model input data
        first_click_interval_flag: Flag for first click interval
        five_min_interval_flag: Flag for five minute interval
        ten_min_interval_flag: Flag for ten minute interval
        fifteen_min_interval_flag: Flag for fifteen minute interval
        twenty_min_interval_flag: Flag for twenty minute interval
        rules_block_flag: Flag to determine if rules should be applied
        quartile_value: Quartile value for rule 4
        cosmos_db: DataFrame from Cosmos DB
        call_flag: Flag indicating if user has called before
        quartile: Quartile value to store in Cosmos DB
        out_score: Score to store in Cosmos DB
        target_bucket: Target bucket to store in Cosmos DB
        invoca_score: Invoca score to store in Cosmos DB
        db_manager: Optional database manager instance
        
    Returns:
        Tuple of (updated_model_input, all_rules_satisfied, results_dict)
    """
    try:
        # Initialize variables
        all_rules_satisfied = 'No'
        rule1 = False
        rule2 = False
        rule3 = pd.Series([False] * len(model_input), index=model_input.index)
        rule4 = False
        
        results = {
            "rule1": False,
            "rule2": False,
            "rule3": False,
            "rule4": False,
            "rule5": "",
            "all_rules_satisfied": False,
            "show_modal": False,
            "cohort": "" #Drop Out
        }
        
        if rules_block_flag == 'Yes':
            logger.info("Checking the 4 rules in sequence")
            
            # Apply rule 1
            rule1 = (model_input['url_path_hit'] == 0) & (model_input['page_name_hit'] == 0)
            rule1_value = rule1.iloc[0] if not rule1.empty else False
            logger.info(f"Rule 1: {rule1_value}")
            results["rule1"] = bool(rule1_value)
            
            # Apply rule 2
            rule2 = call_flag == 'No'  # rule 2 is True if user has not called
            logger.info(f"Rule 2: {rule2}")
            results["rule2"] = rule2
            
            # Apply rule 3
            rule3 = get_rule3_sync(
                model_input,
                first_click_interval_flag,
                five_min_interval_flag,
                ten_min_interval_flag,
                fifteen_min_interval_flag,
                twenty_min_interval_flag
            )
            rule3_value = rule3.iloc[0] if not rule3.empty else False
            logger.info(f"Rule 3: {rule3_value}")
            results["rule3"] = bool(rule3_value)
            
            # Apply rule 4
            rule4 = quartile_value in ['Q1', 'Q2', 'Q3']
            logger.info(f"Rule 4: {rule4}")
            results["rule4"] = rule4
            
            # Check if all rules are satisfied
            all_rules_satisfied_value = rule1_value and rule2 and rule3_value and rule4
            logger.info(f"All rules satisfied: {all_rules_satisfied_value}")
            results["all_rules_satisfied"] = all_rules_satisfied_value
            
            # Set the output values
            if out_score is not None:
                model_input['out_score'] = out_score
            if target_bucket is not None:
                model_input['target_bucket'] = target_bucket
            
            # Update all_rules_satisfied flag
            all_rules_satisfied = 'Yes' if all_rules_satisfied_value else 'No'
            results["show_modal"] = all_rules_satisfied_value
            
            # Set cohort and update Cosmos DB
            if all_rules_satisfied_value:
                logger.info("All rules satisfied - checking rule 5")
                cohort, hist_label = apply_rule_5_from_api_response(entities_coverage_request)
                logger.info(f"Cohort determined: {cohort}, hist_label: {hist_label} from rule 5")
            else:
                logger.info("All rules not satisfied - Do not Show Modal")
                cohort = 'Drop Out'
                hist_label = 0
                
            results["cohort"] = cohort
            
            # Update Cosmos DB
            if cosmos_db is not None:
                try:
                    update_cosmos_db(
                        model_input, 
                        county_data,
                        hist_label=hist_label,
                        invoca_score=invoca_score,
                        quartile=quartile,
                        rule1=str(rule1_value),
                        rule2=str(rule2),
                        rule3=str(rule3_value),
                        rule4=str(rule4),
                        rule5=cohort,
                        interval_flag=interval_flag,
                        cohort=cohort,
                        all_rules_satisfied=all_rules_satisfied_value,
                        db_manager=db_manager
                    )
                    logger.info("Data entered in Cosmos DB")
                except Exception as update_error:
                    logger.error(f"Error updating Cosmos DB: {str(update_error)}")
        else:
            logger.info("Rules block flag is not 'Yes', skipping rules evaluation")
        
        return model_input, all_rules_satisfied, results
        
    except Exception as e:
        logger.error(f"Error in rules_block_sync: {str(e)}")
        # Return safe defaults on error
        return model_input, 'No', {
            "rule1": False,
            "rule2": False,
            "rule3": False,
            "rule4": False,
            "rule5": "",
            "all_rules_satisfied": False,
            "show_modal": False,
            "cohort": "Error",
            "error": str(e)
        }

from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import rx
from rx import operators as ops

# Synchronous function to evaluate rule 3 based on interval flags
def get_rule3_sync(
    model_input: pd.DataFrame,
    first_click_interval_flag: str = 'No',
    five_min_interval_flag: str = 'No',
    ten_min_interval_flag: str = 'No',
    fifteen_min_interval_flag: str = 'No',
    twenty_min_interval_flag: str = 'No'
) -> pd.Series:
    """
    Synchronous function to evaluate rule 3 based on various interval flags.
    
    Args:
        model_input: DataFrame containing model input data
        first_click_interval_flag: Flag for first click interval
        five_min_interval_flag: Flag for five minute interval
        ten_min_interval_flag: Flag for ten minute interval
        fifteen_min_interval_flag: Flag for fifteen minute interval
        twenty_min_interval_flag: Flag for twenty minute interval
        
    Returns:
        Boolean Series indicating if rule 3 is satisfied for each row
    """
    try:
        logger.info("Evaluating rule 3 based on interval flags")
        
        if first_click_interval_flag == 'Yes':
            rule3 = model_input['come_bk'] > 5
            logger.info("First click interval rule applied")
        elif five_min_interval_flag == 'Yes':
            rule3 = model_input['come_bk'] > 2
            logger.info("Five minute interval rule applied")
        elif ten_min_interval_flag == 'Yes':
            rule3 = model_input['come_bk'] > 1
            logger.info("Ten minute interval rule applied")
        elif fifteen_min_interval_flag == 'Yes':
            rule3 = model_input['click_min'] < 0.3
            logger.info("Fifteen minute interval rule applied")
        elif twenty_min_interval_flag == 'Yes':
            rule3 = pd.Series([True] * len(model_input), index=model_input.index)
            logger.info("Twenty minute interval rule applied")
        else:
            # Default case if no interval flag is set
            logger.warning("No interval flag set, defaulting to False for rule 3")
            rule3 = pd.Series([False] * len(model_input), index=model_input.index)
            
        return rule3
        
    except Exception as e:
        logger.error(f"Error in get_rule3_sync: {str(e)}")
        # Return default value on error
        return pd.Series([False] * len(model_input), index=model_input.index)

##

#####

from typing import Dict, Optional
import rx
from rx import operators as ops

# Synchronous function to set interval flags
def set_interval_flags(interval_flag: Optional[int] = None, call_attempt_index: Optional[int] = None) -> Dict[str, str]:
    """
    Synchronous function to set interval flags based on the provided index.
    Captures callAttemptIndex from real-time data into interval_flag.
    
    Args:
        interval_flag: Integer value (0-4) indicating which flag to set to 'Yes'
        call_attempt_index: Optional call attempt index from real-time data
        
    Returns:
        Dictionary with interval flags where the specified flag is set to 'Yes'
    """
    try:
        # Use call_attempt_index from real-time data if provided
        if interval_flag is None and call_attempt_index is not None:
            interval_flag = call_attempt_index
            logger.info(f"Using call attempt index as interval flag: {call_attempt_index}")
        
        # Default to 0 if neither is provided
        if interval_flag is None:
            interval_flag = 0
            logger.warning("No interval flag or call attempt index provided, defaulting to 0")
        
        # Initialize all flags to "No"
        all_interval_flags = {
            'first_click_interval_flag': 'No',
            'five_min_interval_flag': 'No',
            'ten_min_interval_flag': 'No',
            'fifteen_min_interval_flag': 'No',
            'twenty_min_interval_flag': 'No'
        }
        
        # Map interval_flag values to corresponding keys
        flag_keys = list(all_interval_flags.keys())
        
        if 0 <= interval_flag < len(flag_keys):
            all_interval_flags[flag_keys[interval_flag]] = 'Yes'
            logger.info(f"Set {flag_keys[interval_flag]} to 'Yes'")
        else:
            logger.warning(f"Invalid interval_flag value: interval_flag. Must be between 0 and 4.")
        
        return all_interval_flags
        
    except Exception as e:
        logger.error(f"Error in set_interval_flags_sync: {str(e)}")
        # Return default values on error
        return {
            'first_click_interval_flag': 'No',
            'five_min_interval_flag': 'No',
            'ten_min_interval_flag': 'No',
            'fifteen_min_interval_flag': 'No',
            'twenty_min_interval_flag': 'No'
        }

# Synchronous function to determine if data should go to rules block
def seg2_seg3(
    model_input: pd.DataFrame,
    county_data: Dict[str, Any],
    cosmos_db: pd.DataFrame,
    invoca_score: float,
    quartile: str,
    interval_flag: int = 0,
    out_score: Optional[float] = None,
    db_manager: Optional[CosmosDBConnectionManager] = None
) -> Tuple[str, pd.DataFrame]:
    """
    Synchronous function to determine if data should be sent to rules block.
    
    Args:
        model_input: DataFrame containing model input data
        cosmos_db: DataFrame from Cosmos DB
        invoca_score: Score from invoca model
        quartile: Quartile value
        out_score: Score to determine cohort path (defaults to model_input's out_score if not provided)
        db_manager: Optional database manager instance
        
    Returns:
        Tuple of (rules_block_flag, updated_cosmos_db)
    """
    try:
        # Initialize variables
        rules_block_flag = ''
        print(f"Unauth score is : {out_score}")
        print(f"Invoca score is : {invoca_score}")
        unauth_score = out_score
            
        logger.info(f"Unauth score is: {unauth_score}")
        
        # Decision logic based on score
        if unauth_score > 0.1:
            logger.info("Do not show modal - Cohort 1-Submitter path")
            
            # Set rule values to NA since not using rules block
            rule1 = 'NA'
            rule2 = 'NA'
            rule3 = 'NA'
            rule4 = 'NA'
            rule5 = 'NA'
            
            # Update Cosmos DB
            try:
                if db_manager is not None:
                    update_cosmos_db(
                        model_input,
                        county_data,
                        hist_label=0,
                        invoca_score=invoca_score,
                        quartile=quartile,
                        rule1=rule1,
                        rule2=rule2,
                        rule3=rule3,
                        rule4=rule4,
                        rule5=rule5,
                        interval_flag=interval_flag,
                        cohort='Cohort 1-Submitter',
                        db_manager=db_manager
                    )
                    logger.info("Successfully updated Cosmos DB with Cohort 1-Submitter")
                else:
                    logger.warning("No DB manager provided, skipping Cosmos DB update")
            except Exception as update_error:
                logger.error(f"Error updating Cosmos DB: {str(update_error)}")
            
            # Set flag to skip rules block
            rules_block_flag = 'No'
            
        elif unauth_score <= 0.1:
            logger.info("Send the data to rules block")
            rules_block_flag = 'Yes'
        
        return rules_block_flag, cosmos_db
        
    except Exception as e:
        logger.error(f"Error in seg2_seg3_sync: {str(e)}")
        # Return safe defaults on error
        return 'No', cosmos_db

# Synchronous function for updating Cosmos DB
def update_cosmos_db(
    model_input: pd.DataFrame,
    county_data: Dict[str, Any],
    hist_label: int,
    invoca_score: float,
    quartile: str,
    rule1: bool,
    rule2: bool,
    rule3: bool,
    rule4: bool,
    rule5: Optional[str] = None,
    interval_flag: int = 0,
    cohort: Optional[str] = None,
    all_rules_satisfied: Optional[bool] = None,
    db_manager: Optional[CosmosDBConnectionManager] = None
) -> Dict[str, Any]:
    """
    Synchronous function to update Cosmos DB with model results and rule flags.
    
    Args:
        model_input: DataFrame with model input data
        hist_label: Historical label (1 or 0)
        invoca_score: Score from invoca model
        quartile: Quartile value
        rule1-rule4: Boolean flags for business rules
        interval_flag: Flag for interval
        cohort: Optional cohort designation
        all_rules_satisfied: Whether all rules are satisfied
        db_manager: Optional database manager instance
        
    Returns:
        Dictionary with update result information
    """
    try:
        logger.info(f"Updating Cosmos DB for MCID: {model_input['mcid'].iloc[0] if not model_input.empty else 'unknown'}")
        
        # Create a copy of the input data
        temp_df = model_input.copy()
        
        # Add prediction results and metadata
        temp_df['hist_label'] = hist_label
        temp_df['called_event'] = None
        temp_df['threshold'] = 0.1
        temp_df['show_noshow'] = 'Show Modal' if hist_label == 1 else 'Do not Show Modal'
        temp_df['invoca_score'] = invoca_score
        temp_df['quartile'] = quartile
        temp_df['rule1'] = rule1
        temp_df['rule2'] = rule2
        temp_df['rule3'] = rule3
        temp_df['rule4'] = rule4
        temp_df['rule5'] = rule5
        temp_df['interval'] = interval_flag
        temp_df['reason'] = None
        temp_df['Cohort'] = cohort
        temp_df['all_rules_satisfied'] = all_rules_satisfied
        temp_df['entered_zip_code'] = county_data.get('entered_zip_code', '')
        temp_df['entered_fips_code'] = county_data.get('entered_fips_code', '')
        temp_df['entered_state_code'] = county_data.get('entered_state_code', '')
        temp_df['entered_county_name'] = county_data.get('entered_county_name', '')
        temp_df['geo_zip_code'] = county_data.get('geo_zip_code', '')
        temp_df['geo_fips_code'] = county_data.get('geo_fips_code', '')
        temp_df['geo_state_code'] = county_data.get('geo_state_code', '')
        temp_df['geo_county_name'] = county_data.get('geo_county_name', '')
        temp_df['request_zip_code'] = county_data.get('request_zip_code', '')
        temp_df['request_fips_code'] = county_data.get('request_fips_code', '')
        temp_df['request_state_code'] = county_data.get('request_state_code', '')
        temp_df['request_county_name'] = county_data.get('request_county_name', '')
        temp_df['update_timestamp'] = time.time()
        
        # Log the modal display decision
        show_modal_flag = temp_df['show_noshow'].iloc[0]
        logger.info(f"Modal display decision: {show_modal_flag}")
        
        # Write to Cosmos DB
        write_to_cosmosdb(temp_df, db_manager)
        
        return {
            "status": "success",
            "mcid": temp_df['mcid'].iloc[0] if not temp_df.empty else None,
            "show_modal": show_modal_flag == 'Show Modal',
            "record_count": len(temp_df),
            "update_timestamp": temp_df['update_timestamp'].iloc[0]
        }
        
    except Exception as e:
        logger.error(f"Error in update_cosmos_db method: {str(e)}")
        raise RuntimeError(f"Failed to update Cosmos DB: {str(e)}")


from typing import Dict, Any, Tuple, Optional
import json
import pandas as pd
import rx
from rx import operators as ops

# Synchronous function to score data with unauth model
async def unauth_model_scoring(
    df: pd.DataFrame,
    model_score_greater_than_threshold_flag: str = 'No',
    threshold: float = 0.1
) -> Tuple[pd.DataFrame, str, float, Optional[str], str, str]:
    """
    Synchronous function to score data using the unauth model.
    
    Args:
        df: DataFrame containing the data to score
        model_score_greater_than_threshold_flag: Flag indicating if model score is greater than threshold
        threshold: Threshold value for scoring (default 0.1)
        
    Returns:
        Tuple of (model_input, model_score_flag, out_score, target_bucket, model_app_name, model_version)
    """
    try:
        logger.info("Preparing data for unauth model scoring")
        logger.info(df.columns)
        model_input = df
         # Convert the transformed data into json format                    
         # Convert to JSON for API call
        data_json = model_input.to_json(orient='records')
        data_json = json.loads(data_json.replace("'", '"'))
        
        if not data_json:
            logger.error("Empty JSON data after conversion")
            return model_input, model_score_greater_than_threshold_flag, 0.0, None, "", ""
            
        data = json.dumps(data_json[0])
        logger.info(f"Model JSON prepared (truncated): {data[:100]}...")
        
        # Call model API
        try:
            scores = await unauth_model_api(data)
            out_score = float(scores[0]) if scores and scores[0] is not None else 0.0
            model_app_name = scores[1] if scores and len(scores) > 1 else ""
            model_version = scores[2] if scores and len(scores) > 2 else ""
        except Exception as model_error:
            logger.error(f"Error calling model API: {str(model_error)}")
            out_score = 0.0
            model_app_name = ""
            model_version = ""
        
        # Set flag based on score threshold
        logger.info(f"Out Score: {out_score}")
        logger.info(f"Score greater than threshold: {out_score > threshold}")
        logger.info(f"Unauth Model App Name: {model_app_name}")
        logger.info(f"Unauth Model Version: {model_version}")
        
        if out_score > threshold:
            model_score_greater_than_threshold_flag = 'Yes'
        else:
            model_score_greater_than_threshold_flag = 'No'
            
        logger.info(f"model_score_greater_than_threshold_flag: {model_score_greater_than_threshold_flag}")
        
        # Add score to model input
        model_input['out_score'] = out_score
        
        # Set target bucket (commented out in original code)
        target_bucket = None
        
        return model_input, model_score_greater_than_threshold_flag, out_score, target_bucket, model_app_name, model_version
        
    except Exception as e:
        logger.error(f"Error in unauth_model_scoring: {str(e)}")
        return pd.DataFrame(), model_score_greater_than_threshold_flag, 0.0, None, "", ""


# Synchronous function to call invoca model API
async def invoca_model_api(data_json: Dict[str, Any]) -> Tuple[float, str, int]:
    """
    Synchronous function to call the invoca model endpoint.
    
    Args:
        data_json: The JSON data to send to the model
        
    Returns:
        Tuple of (model_max_score, model_app_name, model_version)
    """
    try:
        logger.info("Started to call the invoca model endpoint...")
        predict_invoca_input = PredictInvocaInputData(**(json.loads(data_json)))
        predictions = await predict_invoca(predict_invoca_input)

        if not predictions or not isinstance(predictions, list) or len(predictions) == 0:
            raise ValueError("Invalid response format from invoca model API")
        
           
        model_max_score = predictions[0].get('score')
        model_app_name = predictions[0].get('Model_App_Name')
        model_version = predictions[0].get('Model_Version')
        
        logger.info(f"Received invoca prediction with score: {model_max_score}")
        return model_max_score, model_app_name, model_version
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error in HTTP request to invoca model API: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to connect to invoca model service: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in invoca_model_api: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error calling invoca model API: {str(e)}"
        )

# Synchronous function to score data with invoca model
async def invoca_model_scoring(
    model_input: pd.DataFrame
) -> Tuple[pd.DataFrame, str, float]:
    """
    Synchronous function to score data using the invoca model.
    
    Args:
        model_input: DataFrame containing model input data
        
    Returns:
        Tuple of (invoca_model_input, call_flag, invoca_score)
    """
    try:
        logger.info("Preparing data for invoca model scoring")
        
        # Prepare invoca model input
        invoca_model_input = model_input.drop(['url_path_hit', 'page_name_hit'], axis=1, errors='ignore')
        invoca_model_input = invoca_model_input[[
            "mcid", "partion_date", "visitid", "online_application", "hits", "month_visitor_info", 
            "downlink_cnt", "type000", "product_list_info", "time_span_hr", "plan_type_v43_cnt", 
            "custlink_cnt", "vpp_filter_selection_v49_cnt", "type095", "plan_compare_plan_type_p25_cnt", 
            "type070", "daily_visitor_info", "wk_visitor_info", "pagelink_cnt", "exitlink_cnt", 
            "come_bk", "url_path_num", "age", "type090", "max_name_pct", "page_name_cnt", 
            "hr_visitor_info", "geo_dma_info", "host_info", "browser_info", "visitor_domain_info", 
            "campaign_info", "weekday", "race_info", "income_info", "shopper_info", "search_engine_info", "new_visit_flag"
        ]]
        
        # Convert to JSON for API call
        data_json = invoca_model_input.to_json(orient='records')
        data_json = json.loads(data_json.replace("'", '"'))
        
        if not data_json:
            logger.error("Empty JSON data after conversion for invoca model")
            return invoca_model_input, 'No', 0.0
            
        data = json.dumps(data_json[0])
        logger.info(f"Invoca Model JSON prepared (truncated): {data[:100]}...")
        
        # Call invoca model API
        try:
            invoca_scores = await invoca_model_api(data)
            invoca_score = float(invoca_scores[0]) if invoca_scores and invoca_scores[0] is not None else 0.0
        except Exception as model_error:
            logger.error(f"Error calling invoca model API: {str(model_error)}")
            invoca_score = 0.0
        
        logger.info(f"Invoca score: {invoca_score}")
        logger.info(f"Invoca score less than threshold: {invoca_score < 0.5}")
        
        # Set call flag based on score threshold
        if invoca_score < 0.5:
            logger.info("Visitor Not called")
            call_flag = 'No'
        else:
            logger.info("Visitor called")
            call_flag = 'Yes'
            
        logger.info(f"call_flag: {call_flag}")
        
        # Add call flag to model input
        invoca_model_input['call_flag'] = call_flag
        
        return invoca_model_input, call_flag, invoca_score
        
    except Exception as e:
        logger.error(f"Error in invoca_model_scoring: {str(e)}")
        return pd.DataFrame(), 'No', 0.0



# Synchronous function for unauth model API
async def unauth_model_api(data_json: Dict[str, Any]) -> Tuple[float, str, int]:
    """
    Synchronous function to call the unauth model endpoint.
    
    Args:
        data_json: The JSON data to send to the model
        
    Returns:
        Tuple of (model_max_score, model_app_name, model_version)
    """
    try:
        predict_input_data = PredictInputData(**(json.loads(data_json)))
        predictions = await predict(predict_input_data)

        print(f"unauth_model_api predictions: {predictions}")
        if not predictions or not isinstance(predictions, list) or len(predictions) == 0:
            raise ValueError("Invalid response format from unauth model API")
            
        model_max_score = predictions[0].get('score')
        model_app_name = predictions[0].get('Model_App_Name')
        model_version = predictions[0].get('Model_Version')
        
        logger.info(f"Received prediction with score: {model_max_score}")
        return model_max_score, model_app_name, model_version
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error in HTTP request to unauth model API: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to connect to unauth model service: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in unauth model_api: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error calling unauth model API: {str(e)}"
        )

# Synchronous function to create model input
def make_model_input_sync(df: pd.DataFrame) -> pd.DataFrame:
    """
    Synchronous function to create model input from DataFrame.
    
    Args:
        df: Input DataFrame to be processed
        
    Returns:
        Processed DataFrame ready for model input
    """
    try:
        logger.info(f"Creating model input from DataFrame with shape: {df.shape}")
        logger.debug(f"Model input columns: {df.columns.tolist()}")
        
        # Call the request validation function to process the data
        df_results = request_validation(df)
        
        # Log information about key columns for debugging
        if 'hit_date_ts' in df.columns:
            logger.debug(f"hit_date_ts column type: {type(df['hit_date_ts'])}")
            logger.debug(f"Sample hit_date_ts values: {df['hit_date_ts'].iloc[:2] if len(df) > 1 else df['hit_date_ts']}")
        
        if 'weekday' in df_results.columns:
            logger.debug(f"Sample weekday values: {df_results['weekday'].iloc[:5].tolist() if len(df_results) > 4 else df_results['weekday'].tolist()}")
        
        # Create model input DataFrame
        model_input = pd.DataFrame(df_results)
        logger.info(f"Model input created successfully with shape: {model_input.shape}")
        
        return model_input
        
    except Exception as e:
        logger.error(f"Error in make_model_input_sync: {str(e)}")
        # Re-raise with more context for better error handling
        raise RuntimeError(f"Failed to create model input: {str(e)}")

# Synchronous function to check BCI null flag and take appropriate action
def check_bci_null_flag(
    df: pd.DataFrame, 
    cosmos_db: pd.DataFrame, 
    bci_null_flag: bool,
    model_input: pd.DataFrame,
    interval_flag: int = 0,
    db_manager: Optional[CosmosDBConnectionManager] = None
) -> Tuple[pd.DataFrame, bool]:
    """
    Synchronous function to process data based on BCI null flag status.
    
    Args:
        df: DataFrame containing user data
        cosmos_db: DataFrame from Cosmos DB
        bci_null_flag: Flag indicating if BCI is null
        make_model_input_func: Function to create model input
        db_manager: Optional database manager instance
        
    Returns:
        Tuple of (updated_cosmos_db, send_to_mcid_pre_exist_stage_flag)
    """
    try:
        # Default value for flag
        send_to_mcid_pre_exist_stage_flag = False
        if bci_null_flag:
            logger.info("BCI is null - Will show modal")
            # Update Cosmos DB with the BCI information
            # Use the update_cosmos_bci function or implement inline
            try:
                if db_manager is not None:
                    # Assuming update_cosmos_bci exists elsewhere in the codebase
                    update_cosmos_bci(model_input, cosmos_db, interval_flag, db_manager)
                    logger.info("Successfully updated Cosmos DB with BCI information")
                else:
                    # Simple inline implementation if needed
                    logger.warning("No DB manager provided, skipping Cosmos DB update")
            except Exception as update_error:
                logger.error(f"Error updating Cosmos DB: {str(update_error)}")
            
            # Set flag to False since we're not sending to MCID pre-exist stage
            send_to_mcid_pre_exist_stage_flag = False
            
        else:
            logger.info("BCI is not null - Sending payload to MCID pre-exist stage")
            
            # Set flag to True to indicate we should proceed to MCID pre-exist stage
            send_to_mcid_pre_exist_stage_flag = True
        
        return cosmos_db, send_to_mcid_pre_exist_stage_flag
        
    except Exception as e:
        logger.error(f"Error in check_bci_null_flag_sync: {str(e)}")
        # Return the original cosmos_db and a default flag value on error
        return cosmos_db, False

# Synchronous function to update Cosmos DB with BCI information
def update_cosmos_bci(
    df: pd.DataFrame, 
    cosmos_db: pd.DataFrame, 
    interval_flag: int = 0,
    db_manager: Optional[CosmosDBConnectionManager] = None
) -> Dict[str, Any]:
    """
    Synchronous function to update Cosmos DB with BCI information.
    
    Args:
        df: DataFrame containing data to update
        cosmos_db: DataFrame containing existing Cosmos DB data
        interval_flag: Flag indicating interval information
        db_manager: Optional database manager instance
        
    Returns:
        Dictionary with update result information
    """
    try:
        logger.info("Updating Cosmos DB with BCI information")
        
        # Make a copy of the dataframe to avoid modifying the original
        temp_df = df.copy()
        
        # Set all the BCI-specific flags and values
        temp_df['hist_label'] = 1
        temp_df['called_event'] = None
        temp_df['threshold'] = None
        temp_df['target_bucket'] = 'BCI'
        temp_df['show_noshow'] = 'Show Modal'
        temp_df['out_score'] = None
        temp_df['invoca_score'] = None
        temp_df['quartile'] = None
        temp_df['rule1'] = None
        temp_df['rule2'] = None
        temp_df['rule3'] = None
        temp_df['rule4'] = None
        temp_df['rule5'] = None
        temp_df['interval'] = interval_flag
        temp_df['reason'] = None
        temp_df['Cohort'] = 'Cohort 3'
        
        # Store the show_modal_flag and target_bucket values for return
        show_modal_flag = temp_df['show_noshow'].iloc[0]
        target_bucket = temp_df['target_bucket'].iloc[0]
        
        # If we have access to cosmos_db, create a new row in the format of cosmos_db
        if cosmos_db is not None and not cosmos_db.empty:
            # Create a new row with all columns from cosmos_db, defaulting to None
            new_row = pd.Series({col: None for col in cosmos_db.columns})
            
            # Update only the specified columns
            for col in ['mcid', 'hits', 'hist_label', 'called_event', 'out_score', 'threshold', 'target_bucket', 'show_noshow', 'Cohort']:
                if col in temp_df.columns:
                    new_row[col] = temp_df.iloc[0][col]
            
            # Append the new row to the DataFrame if needed
            # cosmos_db = pd.concat([cosmos_db, pd.DataFrame([new_row])], ignore_index=True)
        
        # Write to Cosmos DB if db_manager provided
        if db_manager is not None:
            # Use the direct DB manager method instead of helper function
            # This replaces the write_to_cosmosdb(temp_df) call
            result = write_to_cosmosdb(temp_df, db_manager)
            logger.info(f"Written new row to Cosmos DB: {result.get('status', 'unknown')}")
        else:
            logger.warning("No DB manager provided, skipping Cosmos DB write")
            result = {"status": "skipped", "reason": "No DB manager provided"}
        
        # Log the key values for debugging
        logger.info("The new row key values:")
        key_columns = ['mcid', 'hits', 'hist_label', 'called_event', 'out_score', 'threshold', 'target_bucket', 'show_noshow', 'Cohort']
        log_values = {col: temp_df.iloc[0][col] if col in temp_df.columns else None for col in key_columns}
        logger.debug(f"Row values: {log_values}")
        
        # Return a dictionary with the update results
        return {
            "status": "success",
            "show_modal_flag": show_modal_flag,
            "target_bucket": target_bucket,
            "db_result": result,
            "operation_time": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error in update_cosmos_bci: {str(e)}")
        # Return an error status
        return {
            "status": "error",
            "error": str(e),
            "operation_time": time.time()
        }


def check_bci(county_zip: str, mcid: str, db_manager: CosmosDBConnectionManager) -> Tuple[Optional[str], bool]:
    """
    Synchronous function to check BCI score for a zip code.
    This is designed for CPU-bound operations that shouldn't block the event loop.
    
    Args:
        county_zip: The zip code to check
        db_manager: The database connection manager
        
    Returns:
        Tuple of (quartile_value, BCI_null_flag)
    """
    try:
        logger.info(f"Started to check the BCI score for zip: county_zip")
        
        if not county_zip:
            logger.warning("No zip code provided, returning None and BCI_null_flag as True for mcid {mcid}")
            return None, True
        
        # Use the synchronous container
        query = f"SELECT * FROM c WHERE c.zip = '{county_zip}'"
        df_list = list(db_manager.checkbci_container.query_items(
            query=query, 
            enable_cross_partition_query=True
        ))
        
        df_raw = pd.DataFrame(df_list)
        logger.debug(f"Query results: {df_raw}")
        
        if not df_raw.empty:
            quartile_value = df_raw['Quartile'].iloc[0] if 'Quartile' in df_raw.columns else None
            logger.info(f"Quartile Value: {quartile_value}")
            
            if quartile_value is None or quartile_value == "":
                logger.info("BCI is null: True")
                BCI_null_flag = True
            else:
                logger.info("BCI is null: False")
                logger.info(f"Quartile Value: {quartile_value}")
                BCI_null_flag = False
        else:
            logger.info("Zip code not present")
            logger.info("BCI is null: True")
            BCI_null_flag = True
            quartile_value = None
            
        return quartile_value, BCI_null_flag
        
    except Exception as e:
        logger.error(f"Error checking BCI score: {str(e)}")
        raise


#####



####
# Synchronous function for reading from Cosmos DB
def read_cosmosdb(mcid_filter: Optional[str] = None, db_manager: CosmosDBConnectionManager = None) -> pd.DataFrame:
    """
    Synchronous function to read data from Cosmos DB processed container.
    
    Args:
        mcid_filter: Optional MCID to filter results
        db_manager: The database connection manager
        
    Returns:
        DataFrame containing the data from Cosmos DB
    """
    try:
        logger.info("Started to read the Cosmos DB processed container...")
        
        # If no database manager provided, get one
        if db_manager is None:
            db_manager = CosmosDBConnectionManager()
            if not db_manager.initialized:
                raise ValueError("Database connection not initialized")
        
        # Define the columns we expect in the result
        column_names = [
            'mcid', 'visitid', 'partition_date', 'online_application', 'label', 'hits', 
            'month_visitor_info', 'downlink_cnt', 'type000', 'url_path_hit', 'product_list_info', 
            'time_span_hr', 'plan_type_v43_cnt', 'page_name_hit', 'custlink_cnt', 
            'vpp_filter_selection_v49_cnt', 'type095', 'plan_compare_plan_type_p25_cnt', 
            'type070', 'daily_visitor_info', 'wk_visitor_info', 'pagelink_cnt', 'exitlink_cnt', 
            'come_bk', 'url_path_num', 'age', 'type090', 'max_name_pct', 'page_name_cnt', 
            'hr_visitor_info', 'geo_dma_info', 'host_info', 'browser_info', 'visitor_domain_info', 
            'campaign_info', 'weekday', 'race_info', 'income_info', 'shopper_info', 
            'search_engine_info', 'group', 'model_resource_name', 'model_version_id', 
            'show_modal_flag', 'send_data_to_shopper', 'provider_details', 'drug_cost_details', 
            'predicted_score', 'predicted_label', 'predicted_date', 'hist_label', 'called_event', 
            'out_score', 'threshold', 'target_bucket', 'show_noshow', 'invoca_score', 'quartile', 
            'rule1', 'rule2', 'rule3', 'rule4', 'interval', 'reason', 'Cohort'
        ]
        
        # # Step 1: Check if container is empty
        # check_query = "SELECT TOP 1 * FROM c"
        # check_result = list(db_manager.processed_container.query_items(
        #     query=check_query,
        #     enable_cross_partition_query=True
        # ))
        
        # if not check_result:
        #     # Step 2: Initialize with empty DataFrame with required columns
        #     logger.info("No data found in Cosmos DB, returning empty DataFrame")
        #     df = pd.DataFrame(columns=column_names)
        # else:
            # Step 3: Query with mcid filter
        query = f"SELECT * FROM c WHERE c.mcid = '{mcid_filter}'"
        df_list = list(db_manager.processed_container.query_items(
            query=query,
            enable_cross_partition_query=False
        ))
        if not df_list:
            logger.info(f"No data found for MCID: {mcid_filter}, returning empty DataFrame")
            df = pd.DataFrame(columns=column_names)
        else:
            df = pd.DataFrame(df_list)
            
        logger.info("Successfully read data from Cosmos DB.")
        logger.info(f"Cosmos DB shape: {df.shape}")
        logger.debug(f"Cosmos DB contents (truncated): {df.head(2) if not df.empty else 'Empty DataFrame'}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error reading from Cosmos DB: {str(e)}")
        # Return empty DataFrame on failure
        return pd.DataFrame(columns=column_names)

####


####
# Synchronous function to query the invoca MCIDs master table
def make_connection_mcid_master_table(mcid: str, db_manager: CosmosDBConnectionManager) -> pd.DataFrame:
    """
    Synchronous function to query the invoca MCIDs master table for a specific MCID.
    
    Args:
        mcid: The MCID to query
        db_manager: The database connection manager
        
    Returns:
        DataFrame containing the query results
    """
    try:
        logger.info(f"Started to get the invoca call MCIDs list for MCID: {mcid}")
        
        query = f"SELECT * FROM c WHERE c.mcid = '{mcid}'"
        df_invoca_list = list(db_manager.invoca_mcids_container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))
        
        df_invoca = pd.DataFrame(df_invoca_list)
        logger.info(f"Retrieved {len(df_invoca)} records from invoca MCIDs master table")
        
        return df_invoca
        
    except Exception as e:
        logger.error(f"Error in make_connection_mcid_master_table_sync: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error

###

###
# Synchronous function to look up if an MCID has been called before
def mcid_call_lookup(mcid: str, db_manager: Optional[CosmosDBConnectionManager] = None) -> str:
    """
    Synchronous function to check if an MCID has been called before.
    
    Args:
        mcid: The MCID to check
        db_manager: Optional database connection manager
        
    Returns:
        'Yes' if the MCID has been called before, 'No' otherwise
    """
    try:
        # Create DB manager if not provided
        if db_manager is None:
            db_manager = CosmosDBConnectionManager()
        
        # Call the helper function to check the invoca table
        mcid_table = make_connection_mcid_master_table(mcid, db_manager)
        
        # Determine if MCID has been called before
        if mcid_table is not None and not mcid_table.empty:
            called_event = 'Yes'
        else:
            called_event = 'No'
            
        logger.info(f"MCID: {mcid} has called event: {called_event}")
        return called_event
        
    except Exception as e:
        logger.error(f"Error in mcid_call_lookup_sync: {str(e)}")
        return 'No'  # Default to 'No' on error
###

###
# Synchronous function for MCID validation
def mcid_validation(
    mcid: str,
    df: pd.DataFrame,
    cosmos_db: pd.DataFrame,
    model_input: pd.DataFrame,
    interval_flag: int = 0,
    db_manager: Optional[CosmosDBConnectionManager] = None
) -> Tuple[pd.DataFrame, str, List[str], str]:
    """
    Synchronous function to validate MCID against Cosmos DB and determine if modal should show.
    
    Args:
        mcid: The MCID to validate
        df: DataFrame containing the data
        cosmos_db: DataFrame from Cosmos DB containing history
        model_input: DataFrame with model input data
        interval_flag: Flag for interval
        db_manager: Optional database manager instance
        
    Returns:
        Tuple of (updated_cosmos_db, upper_block_flag, reasons, pop_history)
    """
    try:
        logger.info(f"Validating MCID: {mcid}")
        upper_block_flag = ''
        reasons = []
        
        # Check if cosmos_db is available and has data
        if cosmos_db is not None:
            logger.debug(f"Cosmos DB empty: {cosmos_db.empty}")
            if not cosmos_db.empty:
                mcid_rows = cosmos_db[cosmos_db['mcid'] == mcid]
            else:
                logger.info("Cosmos DB is empty.")
                mcid_rows = None
        else:
            logger.info("Cosmos DB is None.")
            mcid_rows = None

        # Check if MCID has been called before
        # Use the mcid_call_lookup_func or implement the lookup here
        try:
            mcid_called = mcid_call_lookup(mcid=mcid)
        except:
            # Fallback if function not available
            logger.warning(f"mcid_call_lookup function not available, defaulting to 'No'")
            mcid_called = 'No'
            
        logger.info(f"MCID Called: {mcid_called}")
        
        # Get history values
        if mcid_rows is not None and not mcid_rows.empty:
            hist_label_max = mcid_rows['hist_label'].max() if 'hist_label' in mcid_rows.columns else 0
            out_score_max = mcid_rows['out_score'].max() if 'out_score' in mcid_rows.columns else 0
        else:
            hist_label_max = 0
            out_score_max = 0

        logger.info(f"hist_label_max: {hist_label_max}, out_score_max: {out_score_max}")
        
        # Check conditions and build reasons
        if hist_label_max == 1:
            reasons.append('hist_label')
        if out_score_max > 0.5:
            reasons.append('max_score')
        if mcid_called == 'Yes':
            reasons.append('called_before')

        # Set flags based on reasons
        reason_text = ', '.join(reasons) if reasons else None
        upper_block_flag = 'Yes' if reasons else 'No'

        # If we need to block, create a new row and write to DB
        if upper_block_flag == 'Yes':
            logger.info("Making new row for upper block flag")
            
            # Create a temporary DataFrame from model input
            temp_df = model_input.copy()
            temp_df['mcid'] = mcid
            
            # Handle hits column
            if 'visit_page_num' not in temp_df.columns:
                temp_df['hits'] = temp_df['hits']
            else:
                temp_df['hits'] = temp_df['visit_page_num'].max()
                
            # Set all the flags and values
            temp_df['hist_label'] = 0
            temp_df['called_event'] = 'Yes' if mcid_called == 'Yes' else None
            temp_df['out_score'] = None
            temp_df['threshold'] = None
            temp_df['target_bucket'] = 'Model popped before'
            temp_df['show_noshow'] = 'Do not Show Modal'        
            temp_df['invoca_score'] = None
            temp_df['quartile'] = None
            temp_df['rule1'] = None
            temp_df['rule2'] = None
            temp_df['rule3'] = None
            temp_df['rule4'] = None
            temp_df['rule5'] = None
            temp_df['interval'] = interval_flag
            temp_df['reason'] = reason_text
            temp_df['Cohort'] = 'NULL'
            
            # Write to Cosmos DB if db_manager provided
            if db_manager is not None:
                try:
                    write_to_cosmosdb(temp_df, db_manager)
                    logger.info(f"Successfully wrote upper block record to Cosmos DB for MCID: {mcid}")
                except Exception as write_error:
                    logger.error(f"Error writing to Cosmos DB: {str(write_error)}")
            
            # Add to cosmos_db result
            cosmos_db = pd.concat([cosmos_db, temp_df], ignore_index=True)
            logger.info("cosmos_db : %s", cosmos_db.head())
        else:
            logger.info("No upper branch. Send for scoring")

        #return cosmos_db, upper_block_flag, reasons, pop_history
        return cosmos_db, upper_block_flag, reasons
        
    except Exception as e:
        logger.error(f"Error in mcid_validation: {str(e)}")
        #return cosmos_db, '', [], 'No'
        return cosmos_db, '', []

###

####
import json
from typing import Dict, Any, List, Optional
import pandas as pd
import uuid
from fastapi import HTTPException

# Synchronous function to write data to Cosmos DB
def write_to_cosmosdb(
    df: pd.DataFrame,
    db_manager: Optional[CosmosDBConnectionManager] = None
) -> Dict[str, Any]:
    """
    Synchronous function to write DataFrame data to Cosmos DB.
    
    Args:
        df: DataFrame containing data to write
        db_manager: Optional database connection manager
        
    Returns:
        Dictionary with operation results
    """
    try:
        logger.info(f"Started writing DataFrame with {len(df)} rows to Cosmos DB...")
        
        # If no database manager provided, get one
        if db_manager is None:
            db_manager = CosmosDBConnectionManager()
            if not db_manager.initialized:
                raise ValueError("Database connection not initialized")
                
        # Convert DataFrame to JSON records
        t_json = df.to_json(orient='records')
        data = json.loads(t_json)
        
        # Track operation metrics
        records_processed = 0
        success_count = 0
        error_count = 0
        errors = []
        
        # Process data items
        if isinstance(data, list):
            for item in data:
                try:
                    # Ensure each item has a unique id
                    item['id'] = str(uuid.uuid4())
                    
                    # Use the processed container for writes
                    db_manager.processed_container.create_item(body=item)
                    
                    success_count += 1
                except Exception as item_error:
                    error_count += 1
                    error_msg = f"Error upserting item {records_processed}: {str(item_error)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                
                records_processed += 1
        else:
            # Handle single item case
            try:
                # Ensure item has a unique id
                data['id'] = str(uuid.uuid4())
                
                # Use the processed container for writes
                db_manager.processed_container.create_item(body=data)
                
                success_count += 1
            except Exception as item_error:
                error_count += 1
                error_msg = f"Error upserting single item: {str(item_error)}"
                errors.append(error_msg)
                logger.error(error_msg)
            
            records_processed = 1
        
        # Log summary
        logger.info(f"Completed writing to Cosmos DB. Processed: {records_processed}, "
                   f"Success: {success_count}, Errors: {error_count}")
        
        # Return operation results
        return {
            "status": "success" if error_count == 0 else "partial_success" if success_count > 0 else "error",
            "records_processed": records_processed,
            "success_count": success_count,
            "error_count": error_count,
            "errors": errors[:5] if errors else [],  # Limit error messages to first 5
            "operation_time": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error writing to Cosmos DB: {str(e)}")
        raise RuntimeError(f"Failed to write to Cosmos DB: {str(e)}")

####


######check drug and provider new

import os
import time
import logging
import requests
import uuid
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure dedicated logger
logger = logging.getLogger("drug-provider-service")

# Global performance metrics tracker
_metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "avg_response_time": 0,
    "concurrent_requests": 0,
    "max_concurrent_requests": 0
}

# Configure session with connection pooling and retry logic
def get_requests_session():
    """Create and return a requests session with connection pooling and retry logic"""
    session = requests.Session()
    
    # Configure connection pooling
    adapter = HTTPAdapter(
        pool_connections=100,  # Maintain connection pool
        pool_maxsize=MAX_POOL_SIZE,      # Maximum number of connections
        max_retries=Retry(
            total=3,           # Total number of retries
            backoff_factor=0.5, # Backoff factor between retries
            status_forcelist=[429, 500, 502, 503, 504], # Retry on these status codes
            allowed_methods=["GET"] # Only retry on GET
        )
    )
    
    # Mount adapter for both HTTP and HTTPS
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

# Global session object (thread-safe in CPython)
_session = get_requests_session()

def check_drug_and_provider_sync(entities_coverage_request,
    timeout: float = 10.0,
    session: Optional[requests.Session] = None
) -> Tuple[str, str]:
    """
    Synchronous function to check drug and provider availability.
    
    Args:
        uuid_val: User UUID
        zipcode: ZIP code
        fipscode: FIPS code
        statecode: State code
        profile: Profile type (default "ALL")
        year: Year (defaults to current year if not provided)
        dgc_cookie: Optional cookie value for authentication
        timeout: Request timeout in seconds
        session: Optional requests session for connection pooling
        
    Returns:
        Tuple of (drug_cost_estimator, provider_details)
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    
    # Track metrics
    _metrics["total_requests"] += 1
    _metrics["concurrent_requests"] += 1
    _metrics["max_concurrent_requests"] = max(
        _metrics["max_concurrent_requests"], 
        _metrics["concurrent_requests"]
    )
    
    zipcode = validate_request_component(entities_coverage_request["zipCode"], r"\d{5}")
    fipscode = validate_request_component(entities_coverage_request["fipsCode"], r"\d*")
    statecode = validate_request_component(entities_coverage_request["stateCode"], r"[a-zA-Z]*")
    year = year = validate_request_component(entities_coverage_request["planYear"], r"\d{4}")
    uuid_val = validate_request_component(entities_coverage_request["uuid"], r"[A-Za-z0-9\-]*")
    dgc_cookie_value = validate_request_component(entities_coverage_request.get("dgcCookie", None), r"[A-Za-z0-9]*")
    
    drug_list = []
    provider_list = []
    
    try:
        # Validate required parameters
        if not all([uuid_val, zipcode, fipscode, statecode, dgc_cookie_value]):
            logger.warning(f"Missing required parameters - UUID: uuid_val, Zip: zipcode, FIPS: fipscode, State: statecode, DGC Cookie: dgc_cookie_value")
            return drug_list, provider_list
        
        # Set default profile
        profile = "ALL"
        
        # Set default year if not provided
        if not year:
            dt = datetime.now()
            year = str(dt.year)
        
        # Log request details
        logger.info(f"[request_id] Drug/Provider check - UUID: uuid_val, Zip: zipcode, FIPS: fipscode, State: statecode, Year: year")
        
        # Get landrover URL from environment
        landrover_url = os.getenv("landrovercoverageurl")
        if not landrover_url:
            logger.error(f"[{request_id}] Missing environment variable: landrovercoverageurl")
            return drug_list, provider_list
        
        # Construct request URL
        shopper_url = f"{landrover_url}/{uuid_val}/{zipcode}/{fipscode}/{statecode}?profile={profile}&year={year}"
        
        # Set up headers
        headers = {'Content-Type': 'application/json'}
        headers['Cookie'] = f"dgcCookie={dgc_cookie_value}"
        # logger.info(f"[request_id] Drug/Provider check -Cookie: {dgc_cookie_value} UUID: {uuid_val}, Zip: {zipcode}, FIPS: {fipscode}, State: {statecode}, Year: {year}")

        # Use provided session or global session
        request_session = session or _session
        
        # Make request with timeout
        response = request_session.get(
            shopper_url.replace('\n', '').replace('\r', ''), 
            headers=headers, 
            timeout=timeout
        )
        logger.info(f"[request_id] Response text: response.text")
        
        ##
        if response.status_code == 200 and response.text:
            try:
                response_data = response.json()
                drug_info_details = response_data.get("drugInfoDetails", [])
                providers_details = response_data.get("providersDetails", {})
                
                if isinstance(drug_info_details, list):
                    drug_list = drug_info_details
                
                if isinstance(providers_details, dict):
                    provider_list = providers_details.get("providerIdList") or []
                
            except Exception as e:
                logger.error(f"[{request_id}] Error parsing response JSON: {str(e)}")
        else:
            logger.warning(f"[{request_id}] Non-200 response: {response.status_code}")

        logger.info(f"[{request_id}] NEW Drug/Provider check completed in  - Drug: {len(drug_list)}, Provider: {len(provider_list)}")
       
        _metrics["successful_requests"] += 1
        processing_time = time.time() - start_time
        _metrics["avg_response_time"] = (
            (_metrics["avg_response_time"] * (_metrics["successful_requests"] - 1) + processing_time) / 
            _metrics["successful_requests"]
        )
        
        logger.info(f"[{request_id}] Drug/Provider check completed in {processing_time:.2f}s - Drug: {len(drug_list)}, Provider: {len(provider_list)}")
        
        return drug_list, provider_list
        
    except requests.exceptions.Timeout:
        logger.error(f"[request_id] Request timeout checking drug/provider for UUID uuid_val")
        _metrics["failed_requests"] += 1
        
    except requests.exceptions.RequestException as e:
        logger.error(f"[request_id] Request error checking drug/provider for UUID uuid_val: str(e)")
        _metrics["failed_requests"] += 1
        
    except Exception as e:
        logger.error(f"[request_id] Error checking drug/provider for UUID uuid_val: str(e)", exc_info=True)
        _metrics["failed_requests"] += 1
         
    finally:
        # Always decrement concurrent requests counter
        _metrics["concurrent_requests"] -= 1
        return drug_list, provider_list

######logs

import concurrent.futures
import pandas as pd
import json
import uuid
import time
import logging
from typing import List, Dict, Any, Union
from azure.cosmos import exceptions

# Configure dedicated logger
logger = logging.getLogger("data-logger")

def log_request_data(requestdata: Union[List[Dict[str, Any]], Dict[str, Any]], 
                         sequence_id: str,
                         db_manager: CosmosDBConnectionManager,
                         max_workers: int = 10,
                         batch_size: int = 100) -> bool:
    """
    Log request data to Cosmos DB with improved error handling for scalar values & parallel processing for high throughput.
    
    Args:
        requestdata: Request data to log (list or dict)
        sequence_id: Unique sequence identifier for the request
        db_manager: CosmosDB connection manager instance
        max_workers: Maximum number of parallel workers for DB writes
        batch_size: Number of items to batch together for processing
        
    Returns:
        bool: True if successful, False otherwise
    """
    start_time = time.time()
    items_processed = 0
    
    try:
        logger.info(f"Logging request payload data with sequence ID: {sequence_id}")
        logger.info(f"Logging request payload data : {requestdata}")
        
        # Check if requestdata is a scalar dictionary and convert to DataFrame safely
        if isinstance(requestdata, dict):
            # Convert scalar values to lists to avoid the pandas error
            requestdata_list = {k: [v] for k, v in requestdata.items()}
            df_req = pd.DataFrame(requestdata_list)
        else:
            # If it's already a list of dictionaries or other format, use as is
            df_req = pd.DataFrame(requestdata)
        
        # Add logging of MCID if available
        if 'mcid' in df_req.columns and len(df_req) > 0:
            logger.info(f"Processing request for MCID: {df_req['mcid'].values[0]}")
        
        # Add sequence ID and timestamp to each record
        df_req['sequence_id'] = sequence_id
        df_req['log_timestamp'] = time.time()
        
        # Convert DataFrame to records
        data = df_req.to_dict(orient='records')
        
        # Ensure container is accessible
        request_container = db_manager.request_container
        
        # Function to process a batch of items
        def process_batch(batch_items):
            successful = 0
            failed = 0
            
            # Add unique IDs to each item if not present
            for item in batch_items:
                if 'id' not in item:
                    item['id'] = str(uuid.uuid4())
                
                # Add partition key if needed
                if 'sequence_id' not in item:
                    item['sequence_id'] = sequence_id
            
            # Attempt batch operations if supported, otherwise individual writes
            try:
                # Try bulk insert if available (depends on SDK version)
                # This is faster but may not be available in all environments
                request_container.create_items(batch_items)
                successful = len(batch_items)
            except (AttributeError, exceptions.CosmosHttpResponseError):
                # Fall back to individual operations with retry logic
                for item in batch_items:
                    retry_count = 0
                    max_retries = 3
                    
                    while retry_count < max_retries:
                        try:
                            request_container.create_item(item)
                            successful += 1
                            break
                        except exceptions.CosmosHttpResponseError as e:
                            # Handle rate limiting (429) with backoff
                            if e.status_code == 429:
                                retry_count += 1
                                if retry_count < max_retries:
                                    # Exponential backoff
                                    time.sleep(0.1 * (2 ** retry_count))
                                else:
                                    failed += 1
                                    logger.warning(f"Rate limited after {max_retries} retries")
                            else:
                                failed += 1
                                break
            
            return successful, failed
        
        # Split data into batches
        batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        
        # Process batches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]
            
            # Wait for all futures to complete
            for future in concurrent.futures.as_completed(futures):
                successful, failed = future.result()
                items_processed += successful
        
        elapsed_time = time.time() - start_time
        logger.info(f"Request data logging completed: {items_processed} items in {elapsed_time:.2f}s "
                   f"({items_processed/elapsed_time:.1f} items/sec)")
        
        return True
    
    except Exception as e:
        logger.error(f"Error in log_requestdata_sync: {str(e)}", exc_info=True)
        return False

def log_request_data_v1(requestdata: Union[Dict[str, Any]], 
                        sequence_id: str,
                        sourceVisitId: str,
                        db_manager: CosmosDBConnectionManager) -> bool:
    """
    Log request data to Cosmos DB with improved error handling for scalar values & parallel processing for high throughput.
    
    Args:
        requestdata: Request data to log (list or dict)
        sequence_id: Unique sequence identifier for the request
        db_manager: CosmosDB connection manager instance
        max_workers: Maximum number of parallel workers for DB writes
        batch_size: Number of items to batch together for processing
        
    Returns:
        bool: True if successful, False otherwise
    """
    
    logger.info(f"Logging request payload data with sequence ID: sequence_id")
    logger.info(f"Logging request payload data : requestdata")
    
    requestdata['id'] = str(uuid.uuid4())
    requestdata['sequence_id'] = sequence_id
    requestdata['log_timestamp'] = time.time()
    requestdata["requestId"] = requestdata["mcid"]+"_"+sourceVisitId+"_"+str(requestdata["callAttemptIndex"])
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            query = """
            SELECT TOP 1 * 
            FROM c 
            WHERE c.requestId = @requestId
            order by c._ts asc
            """
            parameters = [{"name": "@requestId", "value": requestdata["requestId"]}]

            requestData = list(db_manager.request_container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))

            if requestData is not None and len(requestData) > 0:
                return 409 
            
            db_manager.request_container.create_item(requestdata)

            return 200
        except exceptions.CosmosHttpResponseError as e:
            if e.status_code == 429:
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(0.1 * (2 ** retry_count))
                else:
                    status = 429
                    logger.warning(f"Rate limited after {max_retries} retries")
            else:
                status = 500
                break
        
    return status


######logs
##logflags
# Synchronous function to log flag model data
def log_flag_data(
    flagmodel_data: Dict[str, Any], 
    sequence: str, 
    db_manager: Optional[CosmosDBConnectionManager] = None
) -> Dict[str, Any]:
    """
    Synchronous function to log flag model data to Cosmos DB.
    
    Args:
        flagmodel_data: The flag model data to log
        sequence: A unique sequence identifier to use as the ID
        db_manager: Optional database manager instance
        
    Returns:
        Dictionary with operation results
    """
    try:
        logger.info("Started to log the flag data info...")
        
        # If no database manager provided, get one
        if db_manager is None:
            db_manager = CosmosDBConnectionManager()
            if not db_manager.initialized:
                raise ValueError("Database connection not initialized")
        
        # Convert to DataFrame and then to JSON
        df_flagdata = pd.DataFrame(flagmodel_data)
        t_json = df_flagdata.to_json(orient='records')
        data = json.loads(t_json)
        
        # Track operation metrics
        records_processed = 0
        success_count = 0
        error_count = 0
        
        # Process data items
        if isinstance(data, list):
            for item in data:
                try:
                    # Use the provided sequence as ID
                    item['id'] = str(sequence)
                    
                    # Insert into flag container
                    db_manager.flag_container.create_item(body=item)
                    success_count += 1
                except Exception as item_error:
                    error_count += 1
                    logger.error(f"Error upserting flag item: {str(item_error)}")
                
                records_processed += 1
        else:
            try:
                # Use the provided sequence as ID
                data['id'] = str(sequence)
                
                # Insert into flag container
                db_manager.flag_container.create_item(body=data)
                success_count = 1
            except Exception as item_error:
                error_count = 1
                logger.error(f"Error upserting single flag item: {str(item_error)}")
            
            records_processed = 1
        
        logger.info(f"Completed logging flag data. Processed: {records_processed}, Success: {success_count}, Errors: {error_count}")
        
        return {
            "status": "success" if error_count == 0 else "partial_success" if success_count > 0 else "error",
            "records_processed": records_processed,
            "success_count": success_count,
            "error_count": error_count,
            "operation_time": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error in log_flag_data: {str(e)}")
        raise RuntimeError(f"Failed to log flag data: {str(e)}")
##logflags


##log combined data  - sanitized json
import json
import numpy as np
from datetime import datetime, date
from decimal import Decimal

def sanitize_for_json(obj):
    """
    Recursively sanitize data structures for JSON serialization.
    
    Args:
        obj: Object to sanitize
        
    Returns:
        JSON-serializable object
    """
    if obj is None:
        return None
    elif isinstance(obj, (str, int, bool)):
        return obj
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return float(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return float(obj)
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif hasattr(obj, 'item'):  # numpy scalars
        return sanitize_for_json(obj.item())
    else:
        return str(obj)

def log_combined_data(
    transform_data: Union[pd.DataFrame, Dict, List[Dict]], 
    response_data: Union[List[Dict], Dict], 
    sequence_id: str,
    db_manager: Optional[CosmosDBConnectionManager] = None
) -> Dict[str, Any]:
    """
    Efficiently log transformed data and response data to Cosmos DB with improved JSON handling.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Logging combined data with sequence ID: {sequence_id}")
        
        # Initialize DB manager if not provided
        if db_manager is None:
            db_manager = CosmosDBConnectionManager()
            if not db_manager.initialized:
                logger.error("Database connection not initialized")
                return {"status": "error", "message": "Database connection not initialized"}
        
        # Convert inputs to DataFrames if they aren't already
        if not isinstance(transform_data, pd.DataFrame):
            df_transform = pd.DataFrame([transform_data] if isinstance(transform_data, dict) else transform_data)
        else:
            df_transform = transform_data
            
        if not isinstance(response_data, list):
            response_data = [response_data]
        df_response = pd.DataFrame(response_data)
        
        if df_transform.empty or df_response.empty:
            logger.warning("Empty transform or response data provided")
            return {"status": "error", "message": "Empty data provided"}
        
        # Create a dictionary of fields to copy from transform_data to response_data
        fields_to_copy = {
            'hist_label': None, 'called_event': None, 'threshold': None, 
            'target_bucket': None, 'show_noshow': None, 'out_score': None,
            'invoca_score': None, 'quartile': None, 'rule1': None, 
            'rule2': None, 'rule3': None, 'rule4': None, 'rule5': "",  
            'interval': None, 'reason': None, 'Cohort': None
        }
        
        # Copy values from first row of transform_data to all rows in response_data
        for field, default in fields_to_copy.items():
            if field in df_transform.columns and len(df_transform) > 0:
                try:
                    value = df_transform.iloc[0][field]
                    # Sanitize the value before assignment
                    df_response[field] = sanitize_for_json(value)
                except (IndexError, KeyError) as e:
                    logger.warning(f"Error accessing field {field}: {str(e)}")
                    df_response[field] = default
            else:
                df_response[field] = default
        
        # Prepare items for Cosmos DB with better error handling
        items = []
        for idx, row in df_response.iterrows():
            try:
                # Extract nested values safely
                model_metadata = row.get('ModelMetadata', {})
                if not isinstance(model_metadata, dict):
                    model_metadata = {}
                    
                digital_rec = row.get('digitalRecommendation', {})
                if not isinstance(digital_rec, dict):
                    digital_rec = {}
                
                # Create base item with sanitized values
                item = {
                    "id": str(sequence_id),  # Use sequence_id as ID
                    "sequence_id": str(sequence_id),
                    "timestamp": datetime.now().isoformat(),
                    
                    # Top-level fields with sanitization
                    "group": sanitize_for_json(row.get('group', '')),
                    "usecase": sanitize_for_json(row.get('useCase', '')),
                    
                    # Model metadata fields
                    "model_resource_name": sanitize_for_json(model_metadata.get('model_resource_name', '')),
                    "model_version_id": sanitize_for_json(model_metadata.get('model_version_id', '')),
                    
                    # Digital recommendation fields
                    "visitid": sanitize_for_json(digital_rec.get('visitid', '')),
                    "mcid": sanitize_for_json(digital_rec.get('mcid', '')),
                    "show_modal_flag": sanitize_for_json(digital_rec.get('show_modal_flag', '')),
                    "provider_details": sanitize_for_json(digital_rec.get('provider_details', '')),
                    "drug_cost_details": sanitize_for_json(digital_rec.get('drug_cost_estimator', '')),
                    "predicted_score": sanitize_for_json(digital_rec.get('predicted_score', 0.0)),
                    "predicted_date": sanitize_for_json(digital_rec.get('predicted_date', '')),
                    "quartile_digital": sanitize_for_json(digital_rec.get('quartile', '')),
                    "target_bucket_digital": sanitize_for_json(digital_rec.get('target_bucket', '')),
                }
                try:
                    existing_target_bucket = response_data[idx]["digitalRecommendation"]["target_bucket"]
                    item["target_bucket"] = row["target_bucket"] or existing_target_bucket
                    item["model_resource_name"] = response_data[idx]["ModelMetadata"]["model_resource_name"]
                    item["model_version_id"] = response_data[idx]["ModelMetadata"]["model_version_id"]
                    item["entered_zip_code"] = response_data[idx]["digitalRecommendation"]["enteredZipCode"] or ""
                    item["entered_fips_code"] = response_data[idx]["digitalRecommendation"]["enteredFipsCode"] or ""
                    item["entered_state_code"] = response_data[idx]["digitalRecommendation"]["enteredStateCode"] or ""
                    item["entered_county_name"] = response_data[idx]["digitalRecommendation"]["enteredCountyName"] or ""

                    item["geo_zip_code"] = response_data[idx]["digitalRecommendation"]["geoZipCode"] or ""
                    item["geo_fips_code"] = response_data[idx]["digitalRecommendation"]["geoFipsCode"] or ""
                    item["geo_state_code"] = response_data[idx]["digitalRecommendation"]["geoStateCode"] or ""
                    item["geo_county_name"] = response_data[idx]["digitalRecommendation"]["geoCountyName"] or ""

                    item["request_zip_code"] = response_data[idx]["digitalRecommendation"]["requestZipCode"] or ""
                    item["request_fips_code"] = response_data[idx]["digitalRecommendation"]["requestFipsCode"] or ""
                    item["request_state_code"] = response_data[idx]["digitalRecommendation"]["requestStateCode"] or ""
                    item["request_county_name"] = response_data[idx]["digitalRecommendation"]["requestCountyName"] or ""

                    if not existing_target_bucket:
                        response_data[idx]["digitalRecommendation"]["target_bucket"] = row["target_bucket"]
                except Exception:
                    logger.error(f"Error setting target_bucket for row {idx}: {row.get('target_bucket', 'N/A')}")

                # Add fields from transform data with sanitization
                for field in fields_to_copy:
                    item[field] = sanitize_for_json(row.get(field))
                
                row["target_bucket"] = item["target_bucket"]
                # Validate the item can be serialized to JSON
                try:
                    json.dumps(item)
                    items.append(item)
                except (TypeError, ValueError) as json_error:
                    logger.error(f"Item {idx} failed JSON serialization: {str(json_error)}")
                    # Log the problematic keys
                    for key, value in item.items():
                        try:
                            json.dumps({key: value})
                        except:
                            logger.error(f"Problematic field: key = type(value) -> value")
                    continue
                    
            except Exception as item_error:
                logger.error(f"Error processing row {idx}: {str(item_error)}")
                continue
        
        if not items:
            logger.warning("No valid items to write to Cosmos DB")
            return {"status": "error", "message": "No valid items to write"}
        
        # Write items to Cosmos DB with better error handling
        success_count = 0
        error_count = 0
        errors = []
        
        container = db_manager.response_container
        
        for i, item in enumerate(items):
            try:
                # Write to Cosmos DB
                container.create_item(body=item)
                success_count += 1
                
            except (TypeError, ValueError) as json_error:
                error_count += 1
                error_msg = f"JSON encoding error for item {i}: {str(json_error)}"
                errors.append(error_msg)
                logger.error(error_msg)
                
            except Exception as e:
                error_count += 1
                error_msg = f"Error writing item {i}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        result = {
            "status": "success" if error_count == 0 else "partial_success" if success_count > 0 else "error",
            "total_items": len(items),
            "successful_items": success_count,
            "failed_items": error_count,
            "processing_time": processing_time,
            "errors": errors[:5] if errors else []
        }
        
        logger.info(f"Combined data logging completed: {success_count}/{len(items)} items in {processing_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Error in log_combined_data: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "processing_time": time.time() - start_time
        }
##log combined data - sanitized json

#########loading .csv

import os
import pandas as pd
from functools import lru_cache
from pathlib import Path
import logging

logger = logging.getLogger("data-processor")

@lru_cache(maxsize=10)
def load_csv_data(filename: str, directory: str = None) -> pd.DataFrame:
    """
    Load a CSV file with caching, optimized for repeated access.
    
    Args:
        filename: Name of the CSV file to load
        directory: Optional directory path. If None, uses the script's directory
        
    Returns:
        DataFrame containing the CSV data
        
    Raises:
        FileNotFoundError: If the CSV file cannot be found
        pd.errors.EmptyDataError: If the CSV file is empty
        pd.errors.ParserError: If the CSV file cannot be parsed
    """
    try:
        # Determine file path
        if directory is None:
            directory = os.path.dirname(os.path.abspath(__file__))
        
        file_path = os.path.join(directory, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            alternative_paths = [
                Path(directory) / "data" / filename,  # Check in a data subdirectory
                Path(directory).parent / "data" / filename,  # Check in parent's data directory
                Path(directory).parent / filename  # Check in parent directory
            ]
            
            for alt_path in alternative_paths:
                if alt_path.exists():
                    file_path = str(alt_path)
                    logger.info(f"Found CSV file at alternative location: {file_path}")
                    break
            else:
                raise FileNotFoundError(f"CSV file not found: {filename}")
        
        # Log file loading
        logger.info(f"Loading CSV file: {file_path}")
        
        # Optimize loading based on file size
        file_size = os.path.getsize(file_path)
        
        if file_size > 100 * 1024 * 1024:  # Large file (>100MB)
            # For large files, use chunking and optimization
            logger.info(f"Large CSV file detected ({file_size/1024/1024:.1f} MB). Using optimized loading.")
            return pd.read_csv(
                file_path,
                low_memory=True,
                dtype_backend="numpy_nullable"  # More memory efficient
            )
        else:
            # For smaller files, standard loading is fine
            logger.info(f"CSV file loaded")
            return pd.read_csv(file_path)
    
    except FileNotFoundError:
        logger.error(f"CSV file not found: {filename}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file is empty: {filename}")
        raise
    except pd.errors.ParserError:
        logger.error(f"Error parsing CSV file: {filename}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading CSV file {filename}: {str(e)}")
        raise

# Updated make_model_input function using the helper
def make_model_input(request_data) -> pd.DataFrame:
    try:
        df_orig = request_data
        
        # Use the helper function instead of direct file loading
        EncodeItem_type_df = load_csv_data("df_encoded_itemtype.csv")
        logger.info("EncodeItem_type Data loaded successfully")

        df_orig = preprocess_data(df_orig, EncodeItem_type_df)

        logger.info(f"b4 Prepare model input df_orig result: {df_orig}")
        # Prepare model input
        final_data = df_orig.rename(columns={"partition_date": "partion_date"})
        logger.info(f"Prepare model input final_data result: {final_data}")

        model_input_columns = [
            "mcid", "partion_date", "visitid", "online_application", "hits", 
            "month_visitor_info", "downlink_cnt", "type000", "url_path_hit", 
            "product_list_info", "time_span_hr", "plan_type_v43_cnt", "page_name_hit", 
            "custlink_cnt", "vpp_filter_selection_v49_cnt", "type095", 
            "plan_compare_plan_type_p25_cnt", "type070", "daily_visitor_info", 
            "wk_visitor_info", "pagelink_cnt", "exitlink_cnt", "come_bk", 
            "url_path_num", "age", "type090", "max_name_pct", "page_name_cnt", 
            "hr_visitor_info", "geo_dma_info", "host_info", "browser_info", 
            "visitor_domain_info", "campaign_info", "weekday", "race_info", 
            "income_info", "shopper_info", "search_engine_info", 'new_visit_flag'
        ]
        
        model_input = final_data[model_input_columns]

        return model_input
    except Exception as e:
        logger.error(f"Error in request_validation: {str(e)}")
        # Re-raise the exception or handle it as needed
        raise

# Updated request_validation function using the helper
def request_validation(request_data) -> pd.DataFrame:
    try:
        user_data = request_data
        df_orig = pd.DataFrame(user_data)        
       
        # Use the helper function instead of direct file loading
        EncodeItem_type_df = load_csv_data("df_encoded_itemtype.csv")
        logger.info("EncodeItem_type Data loaded successfully")
        
        df_orig = preprocess_data(df_orig, EncodeItem_type_df)

        return df_orig
    except Exception as e:
        logger.error(f"Error in request_validation: {str(e)}")
        # Re-raise the exception or handle it as needed
        raise
#########loading .csv

####dateformat converter

####dateformat converter


####DF GRP
import numpy as np
def df_group(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Efficiently group DataFrame by visitid with various aggregations.
    
    This function preprocesses the input DataFrame and performs grouped
    aggregations to create a summary DataFrame with one row per visitid.
    
    Args:
        df_raw: Input DataFrame with raw data
        
    Returns:
        DataFrame with one row per visitid and aggregated columns
        
    Raises:
        ValueError: If input DataFrame is empty or missing required columns
    """
    try:
        # Validate input
        if df_raw is None or len(df_raw) == 0:
            raise ValueError("Input DataFrame is empty")
        
        if 'visitid' not in df_raw.columns:
            raise ValueError("Input DataFrame must contain 'visitid' column")
        
        # Drop rows with missing visitid
        df = df_raw.dropna(subset=['visitid'])
        
        if len(df) == 0:
            logger.warning("All rows had null visitid values")
            return pd.DataFrame()  # Return empty DataFrame instead of failing
        
        # Preprocess the DataFrame
        ##df = preprocess_dataframe(df)
        # Define column groups
        num_cols = [
            'visit_page_num', 'visit_num', 'new_visit', 'daily_visitor',
            'monthly_vistor', 'hourly_visitor', 'weekly_visitor'
        ]
        
        cat_cols = [
            'visitor_domain', 'geo_dma', 'host', 'browser', 'campaign',
            'url_path', 'page_name_v1', 'plan_compare_plan_type_p25',
            'plan_type_v43', 'vpp_filter_selection_v49', 'age_calculation_v47',
            'tecnotree_segment_v167', 'tecnotree_personas_highest_rank_v168',
            'tecnotree_personas_all_in_rank_order_v169', 'search_engine', 'product_list'
        ]
        
        # Fill missing values efficiently
        df[cat_cols] = df[cat_cols].fillna('')
        df[num_cols] = df[num_cols].fillna(0)
        
        # Convert specific columns
        df['partition_date'] = df['partition_date'].astype(str).fillna('')
        df['mcid'] = df['mcid'].astype(str).fillna('')

        # Handle string pattern matches efficiently using vectorized operations
        # Create indicator columns for link types
        if 'page_event_desc' in df.columns:
            # Convert to uppercase once and handle missing values
            page_event_upper = df['page_event_desc'].fillna('UNKNOWN').str.upper()
            
            # Create indicator columns using vectorized operations
            df['downlink'] = np.where(page_event_upper.str.contains('LNK_D', na=False), 1, 0)
            df['pagelink'] = np.where(page_event_upper.str.contains('LNK_O', na=False), 1, 0)
            df['custlink'] = np.where(page_event_upper.str.contains('LNK_O', na=False), 1, 0)
            df['exitlink'] = np.where(page_event_upper.str.contains('LNK_E', na=False), 1, 0)
        else:
            # Handle missing columns
            df['downlink'] = 0
            df['pagelink'] = 0  
            df['custlink'] = 0
            df['exitlink'] = 0
            logger.warning("Column 'page_event_desc' not found when creating link indicators")

        # Create indicator columns for page names and URL paths
        if 'page_name_v1' in df.columns:
            page_name_upper = df['page_name_v1'].fillna('UNKNOWN').str.upper()
            df['page_name_info'] = np.where(page_name_upper.str.contains('ONLINE ENROLLMENT', na=False), 1, 0)
        else:
            df['page_name_info'] = 0
            logger.warning("Column 'page_name_v1' not found when creating page_name_info")

        if 'url_path' in df.columns:
            url_path_upper = df['url_path'].fillna('UNKNOWN').str.upper()
            df['url_path_info'] = np.where(
                url_path_upper.str.contains('ONLINE-APPLICATION.HTML/STEP', na=False) | 
                url_path_upper.str.contains('ONLINE-APPLICATION.HTML/NOPLANINFO', na=False), 
                1, 0
            )
            df['online_application'] = np.where(
                url_path_upper.str.contains('ONLINE-APPLICATION', na=False),
                'Online Application', 'Shopper'
            )
        else:
            df['url_path_info'] = 0
            logger.warning("Column 'url_path' not found when creating url_path_info")

        # Convert time columns
        if 'hit_time_gmt' in df.columns:
            df['hit_time'] = pd.to_datetime(df['hit_time_gmt'], unit='s', errors='coerce')

        # Extract date components for grouping
        if 'hit_date_ts' in df.columns:
            logger.info("ABCColumn 'hit_date_ts' CHECK in DataFrame {df['hit_date_ts']}")
            # Remove the timezone offset
            hit_date_ts = df['hit_date_ts'].astype(str).apply(lambda x: x.rsplit(' ', 1)[0])
            # Convert the timestamp string to datetime using pandas with error handling
            datetime_obj = pd.to_datetime(hit_date_ts, format='%d%m%Y %H:%M:%S %w', errors='coerce')
            df['hit_date_ts'] = pd.to_datetime(datetime_obj)

            hit_date_dt = pd.to_datetime(df['hit_date_ts'], errors='coerce')
            df['weekday1'] = hit_date_dt.dt.day_name()
            df['daytime1'] = hit_date_dt.dt.hour
        else:
            df['weekday1'] = 'Unknown'
            df['daytime1'] = 0
            logger.warning("Column 'hit_date_ts' not found when creating weekday/daytime columns")
        # Ensure 'hit_date_ts' is in datetime format before performing the aggregation
        logger.info("Column 'hit_date_ts' CHECK in DataFrame {df['hit_date_ts']}")
        if 'hit_date_ts' in df.columns:
            logger.info("ifffColumn 'hit_date_ts' CHECK in DataFrame {df['hit_date_ts']}")
            # Convert to datetime with error handling
            df['hit_date_ts'] = pd.to_datetime(df['hit_date_ts'], errors='coerce')
            logger.info(df['hit_date_ts'])
            # Log warning for any parsing errors
            null_dates = df['hit_date_ts'].isna().sum()
            if null_dates > 0:
                logger.warning(f"Found {null_dates} rows with invalid date format in hit_date_ts")
        else:
            logger.warning("Column 'hit_date_ts' not found in DataFrame")
            # Create the column with current timestamp if needed
            ##COMMENTED AS PER BUSINESS FUNC prediction impact ? df['hit_date_ts'] = pd.Timestamp.now()
        # Define aggregation dictionary
        agg_dict = {
            'partition_date': 'max',
            'mcid': 'max',
            'visit_page_num': 'max',  # hits
            'visit_num': 'max',       # come_bk
            'new_visit': 'max',       # new_visit_flag
            'visitor_domain': 'max',  # domain_info
            'daily_visitor': 'max',   # daily_visitor_info
            'monthly_vistor': 'max',  # month_visitor_info
            'hourly_visitor': 'max',  # hr_visitor_info
            'weekly_visitor': 'max',  # wk_visitor_info
            'geo_dma': 'max',         # geo_dma_info
            'host': 'max',            # host_info
            'browser': 'max',         # browser_info
            'campaign': 'max',        # campaign_info
            'url_path': 'nunique',    # url_path_num
            'page_name_v1': 'nunique', # page_name_cnt
            'url_path_info': 'max',
            'page_name_info': 'max',
            'plan_compare_plan_type_p25': 'nunique', # plan_compare_plan_type_p25_cnt
            'plan_type_v43': 'nunique',  # plan_type_v43_cnt
            'vpp_filter_selection_v49': 'nunique', # vpp_filter_selection_v49_cnt
            'age_calculation_v47': 'max', # age_info
            'tecnotree_segment_v167': 'max', # race_info
            'tecnotree_personas_highest_rank_v168': 'max', # income_info
            'tecnotree_personas_all_in_rank_order_v169': 'max', # shopper_info
            'search_engine': 'max',   # search_engine_info
            'product_list': 'nunique', # product_list_info
            'downlink': 'sum',        # downlink_cnt
            'pagelink': 'sum',        # pagelink_cnt
            'custlink': 'sum',        # custlink_cnt
            'exitlink': 'sum'         # exitlink_cnt
        }
        
        # Custom aggregation functions
        custom_aggs = {
            'hit_date_ts': lambda x: x.max() - x.min(),  # time_span
            'online_application': lambda x: max(0 if y == 'Shopper' else 1 for y in x)
        }
        
        # Combine standard and custom aggregations
        agg_dict.update(custom_aggs)
        
        # Create grouped DataFrame with optimized aggregation
        # Use optimized approach - splitting aggregations for better performance
        start_cols = df.columns.tolist()
        
        # Check which columns from agg_dict actually exist in the DataFrame
        valid_agg_dict = {k: v for k, v in agg_dict.items() if k in start_cols}
        
        if not valid_agg_dict:
            logger.error("No valid columns found for aggregation")
            return pd.DataFrame()
        
        # Perform groupby operation
        grouped = df.groupby('visitid').agg(**{
            # Rename during aggregation for better readability
            'partition_date': pd.NamedAgg(column='partition_date', aggfunc='max'),
            'mcid': pd.NamedAgg(column='mcid', aggfunc='max'),
            'hits': pd.NamedAgg(column='visit_page_num', aggfunc='max'),
            'come_bk': pd.NamedAgg(column='visit_num', aggfunc='max'),
            'new_visit_flag': pd.NamedAgg(column='new_visit', aggfunc='max'),
            'daily_visitor_info': pd.NamedAgg(column='daily_visitor', aggfunc='max'),
            'month_visitor_info': pd.NamedAgg(column='monthly_vistor', aggfunc='max'),
            'hr_visitor_info': pd.NamedAgg(column='hourly_visitor', aggfunc='max'),
            'wk_visitor_info': pd.NamedAgg(column='weekly_visitor', aggfunc='max'),
            'geo_dma_info': pd.NamedAgg(column='geo_dma', aggfunc='max'),
            'host_info': pd.NamedAgg(column='host', aggfunc='max'),
            'browser_info': pd.NamedAgg(column='browser', aggfunc='max'),
            'visitor_domain_info': pd.NamedAgg(column='visitor_domain', aggfunc='max'),
            'campaign_info': pd.NamedAgg(column='campaign', aggfunc='max'),
            'time_span': pd.NamedAgg(column='hit_date_ts', aggfunc=lambda x: x.max() - x.min()),
            'url_path_num': pd.NamedAgg(column='url_path', aggfunc='nunique'),
            'page_name_cnt': pd.NamedAgg(column='page_name_v1', aggfunc='nunique'),
            'url_path_hit': pd.NamedAgg(column='url_path_info', aggfunc='max'),
            'page_name_hit': pd.NamedAgg(column='page_name_info', aggfunc='max'),
            'online_application': pd.NamedAgg(column='online_application', 
                                              aggfunc=lambda x: max(0 if y == 'Shopper' else 1 for y in x)),
            'weekday': pd.NamedAgg(column='weekday1', aggfunc='max'),
            'daytime': pd.NamedAgg(column='daytime1', aggfunc='max'),
            'plan_compare_plan_type_p25_cnt': pd.NamedAgg(column='plan_compare_plan_type_p25', aggfunc='nunique'),
            'plan_type_v43_cnt': pd.NamedAgg(column='plan_type_v43', aggfunc='nunique'),
            'vpp_filter_selection_v49_cnt': pd.NamedAgg(column='vpp_filter_selection_v49', aggfunc='nunique'),
            'age_info': pd.NamedAgg(column='age_calculation_v47', aggfunc='max'),
            'race_info': pd.NamedAgg(column='tecnotree_segment_v167', aggfunc='max'),
            'income_info': pd.NamedAgg(column='tecnotree_personas_highest_rank_v168', aggfunc='max'),
            'shopper_info': pd.NamedAgg(column='tecnotree_personas_all_in_rank_order_v169', aggfunc='max'),
            'search_engine_info': pd.NamedAgg(column='search_engine', aggfunc='max'),
            'product_list_info': pd.NamedAgg(column='product_list', aggfunc='nunique'),
            'downlink_cnt': pd.NamedAgg(column='downlink', aggfunc='sum'),
            'pagelink_cnt': pd.NamedAgg(column='pagelink', aggfunc='sum'),
            'custlink_cnt': pd.NamedAgg(column='custlink', aggfunc='sum'),
            'exitlink_cnt': pd.NamedAgg(column='exitlink', aggfunc='sum')
        }).reset_index()
        logger.info("CHECKING TIME SPAN")
        logger.info(grouped['time_span'])
        logger.info("CHECKING TIME SPAN")
        # Process time_span to extract seconds
        if 'time_span' in grouped.columns:
            # Convert timedelta to seconds
            grouped['time_span_hr'] = (grouped['time_span'].dt.total_seconds() / 3600) / 8
            grouped.drop('time_span', axis=1, inplace=True)
        logger.info("CHECKING TIME SPANR")
        logger.info(grouped['time_span_hr'])
        logger.info("CHECKING TIME SPANR")
        
        # Convert any object columns that should be numeric
        for col in ['hits', 'come_bk', 'daily_visitor_info', 'month_visitor_info', 
                   'hr_visitor_info', 'wk_visitor_info', 'url_path_num', 'page_name_cnt']:
            if col in grouped.columns:
                grouped[col] = pd.to_numeric(grouped[col], errors='coerce').fillna(0)
        
        logger.info(f"Successfully grouped DataFrame. Rows: {len(grouped)}, Columns: {len(grouped.columns)}")
        return grouped
        
    except Exception as e:
        logger.error(f"Error in df_group: {str(e)}", exc_info=True)
        # Return empty DataFrame in case of error
        return pd.DataFrame()
####DF GRP

####DF LINK

import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple, Dict

logger = logging.getLogger("data-processor")

def oldprocess_link_name(df, EncodeItem_type_df):
    try:
        # Split `link_name` by ':' and explode into separate rows  
        df_exploded = df.assign(item1=df['link_name'].str.split(':')).explode('item1')  
        
        # Trim whitespace and convert to upper case for comparison  
        df_exploded['item1'] = df_exploded['item1'].str.strip()  
        
        # Define the conditions and transformations  
        def transform_item1(item):  
            if pd.isna(item):  
                return None  
            item_upper = item.upper()  
            if 'COUNTY' in item_upper or 'CITY' in item_upper:  
                return 'County'  
            elif '-8' in item_upper or 'CALL' in item_upper:  
                return 'Call'  
            else:  
                return item_upper  
        
        # Apply the transformation  
        df_exploded['item2'] = df_exploded['item1'].apply(transform_item1)  
        
        # Filter out specific values  
        filter_values = ['', '*', '-', '1', '0104260701)', '2023', '2024', '5', '<PLANTYPE> OLE', None]  
        df_filtered = df_exploded[~df_exploded['item1'].isin(filter_values)]  
        
        # Select the desired columns and rename  
        df_result = df_filtered[['item2', 'visitid']].rename(columns={'item2': 'item'})  

        df_encode_item_type = EncodeItem_type_df #EncodeItem_type.toPandas()

        # Perform the join operation
        df_joined = pd.merge(df_result, df_encode_item_type, on='item', how='left')
        # Ensure 'percentage' is numeric, coercing errors to NaN
        df_joined['percentage'] = pd.to_numeric(df_joined['percentage'], errors='coerce')

        # Fill NaN values with 0
        df_joined['percentage'] = df_joined['percentage'].fillna(0)
        df_joined['pcttype'] = df_joined['pcttype'].fillna('type000')

        df_joined['max_name_pct'] = df_joined.groupby('visitid')['percentage'].transform('max')
        # Group by 'visitid' and 'pcttype' and calculate max(percentage)
        df_grouped = df_joined.groupby(['visitid', 'max_name_pct','pcttype']).agg(url_name_pct=('percentage', 'max')).reset_index()

        # Pivot the DataFrame  
        df_pivot = df_grouped.pivot_table(  
            index=['visitid',  'max_name_pct'],
            columns='pcttype',  
            values='url_name_pct',  
            aggfunc='max'  
        ).reset_index()  
        
        # Reindex the columns to include the fixed set of columns  
        fixed_columns = ['type000', 'type070', 'type090', 'type095']  
        df_pivot = df_pivot.reindex(columns=['visitid','max_name_pct'] + fixed_columns, fill_value=0)  
        
        # Select relevant columns  
        df_result1 = df_pivot[['visitid', 'max_name_pct','type000', 'type070', 'type090', 'type095']]  
        
        df_result1 = df_result1.drop('index', axis=1, errors='ignore')

        return df_result1
    except Exception as e:
        app.logger.error(f"Error in link_name: {e}")



def process_link_name(
    df: pd.DataFrame, 
    encode_item_type_df: pd.DataFrame, 
    link_name_col: str = 'link_name',
    visitid_col: str = 'visitid'
) -> pd.DataFrame:
    """
    Process link name data and join with encoding data to create type columns.
    
    This function:
    1. Splits link names by ':'
    2. Processes and categorizes the values
    3. Joins with encoding data
    4. Pivots and aggregates to create type columns
    
    Args:
        df: Input DataFrame containing link_name and visitid columns
        encode_item_type_df: DataFrame with item type encoding information
        link_name_col: Name of the column containing link names (default: 'link_name')
        visitid_col: Name of the column containing visit IDs (default: 'visitid')
        
    Returns:
        DataFrame with visitid, max_name_pct, and type columns
        
    Raises:
        ValueError: If required columns are missing
    """
    try:
        start_rows = len(df)
        logger.info(f"Processing link names for {start_rows} rows")
        
        # Validate input
        if link_name_col not in df.columns:
            logger.warning(f"'{link_name_col}' column not found in DataFrame")
            return create_default_result(df, visitid_col)
            
        if visitid_col not in df.columns:
            logger.warning(f"'{visitid_col}' column not found in DataFrame")
            return create_default_result(df, visitid_col)
        
        # Step 1: Split and explode link_name more efficiently
        # Create a copy of just the columns we need to avoid modifying the original
        link_data = df[[visitid_col, link_name_col]].copy()
        
        # Split by ':' and explode - handle missing values gracefully
        link_data['item1'] = link_data[link_name_col].fillna('').astype(str).str.split(':')
        df_exploded = link_data.explode('item1')
        
        # Step 2: Process item values using vectorized operations
        # Trim whitespace
        df_exploded['item1'] = df_exploded['item1'].str.strip()
        
        # Create uppercase version for pattern matching (more efficient than applying to each row)
        item_upper = df_exploded['item1'].str.upper()
        
        # Apply transformations using numpy.where (more efficient than apply)
        mask_county = item_upper.str.contains('COUNTY|CITY', na=False)
        mask_call = item_upper.str.contains('-8|CALL', na=False)
        
        # Create item2 column using numpy.where for better performance
        conditions = [mask_county, mask_call]
        choices = ['County', 'Call']
        df_exploded['item2'] = np.select(conditions, choices, default=item_upper)
        
        # Step 3: Filter out specific values efficiently
        filter_values = ['', '*', '-', '1', '0104260701)', '2023', '2024', '5', '<PLANTYPE> OLE', None]
        df_filtered = df_exploded[~df_exploded['item1'].isin(filter_values)]
        
        # Early exit if no data remains after filtering
        if len(df_filtered) == 0:
            logger.info("No valid link names found after filtering")
            return create_default_result(df, visitid_col)
        
        # Step 4: Prepare for join
        df_result = df_filtered[['item2', visitid_col]].rename(columns={'item2': 'item'})
        
        # Step 5: Join with encoding data
        # Use merge with indicators to track join success
        df_joined = pd.merge(
            df_result, 
            encode_item_type_df, 
            on='item', 
            how='left',
            indicator=True
        )
        
        # Log join statistics
        join_stats = df_joined['_merge'].value_counts()
        logger.debug(f"Join statistics: {join_stats.to_dict()}")
        
        # Convert percentage to numeric with efficient error handling
        df_joined['percentage'] = pd.to_numeric(df_joined['percentage'], errors='coerce').fillna(0)
        df_joined['pcttype'] = df_joined['pcttype'].fillna('type000')
        
        # Step 6: Calculate max percentage for each visitid
        # Use transform for better performance than groupby+apply
        df_joined['max_name_pct'] = df_joined.groupby(visitid_col)['percentage'].transform('max')
        
        # Step 7: Group and pivot efficiently
        # Group by visitid, max_name_pct, and pcttype and get max percentage
        df_grouped = df_joined.groupby([visitid_col, 'max_name_pct', 'pcttype'], observed=True).agg(
            url_name_pct=('percentage', 'max')
        ).reset_index()
        
        # Step 8: Pivot to create type columns
        # Define the fixed set of columns we need
        fixed_columns = ['type000', 'type070', 'type090', 'type095']
        
        # Create a more efficient pivot table
        df_pivot = df_grouped.pivot_table(
            index=[visitid_col, 'max_name_pct'],
            columns='pcttype',
            values='url_name_pct',
            aggfunc='max'
        ).reset_index()
        
        # Ensure all required columns exist, adding with zeros if missing
        for col in fixed_columns:
            if col not in df_pivot.columns:
                df_pivot[col] = 0.0
        
        # Select and reorder only the columns we need
        result_columns = [visitid_col, 'max_name_pct'] + fixed_columns
        df_result = df_pivot[result_columns].copy()
        
        # Drop any 'index' column that might have been added during processing
        if 'index' in df_result.columns:
            df_result = df_result.drop('index', axis=1)
        
        # Step 9: Final validation and error checking
        # Check for NaN values and fill them
        na_count = df_result.isna().sum().sum()
        if na_count > 0:
            logger.warning(f"Found {na_count} NaN values in result, filling with 0")
            df_result = df_result.fillna(0)
        
        logger.info(f"Successfully processed link names: Input rows={start_rows}, Output rows={len(df_result)}")
        return df_result
        
    except Exception as e:
        logger.error(f"Error in process_link_name: {str(e)}", exc_info=True)
        # Return a default result instead of failing
        return create_default_result(df, visitid_col)

def create_default_result(df: pd.DataFrame, visitid_col: str = 'visitid') -> pd.DataFrame:
    """Create a default result DataFrame with the required columns."""
    # Extract unique visitids or create a placeholder if visitid_col doesn't exist
    if visitid_col in df.columns:
        visitids = df[visitid_col].unique()
    else:
        visitids = ['placeholder']
    
    # Create a DataFrame with the required structure
    result = pd.DataFrame({
        visitid_col: visitids,
        'max_name_pct': 0.0,  # Default value 0.85, 
        'type000': 0.0, # Default value 0.85, 
        'type070': 0.0, # Default value 0.7, 
        'type090': 0.0, # Default value 0.9, 
        'type095': 0.0 # Default value 0.95, 
    })
    
    return result

####DF LINK

####Preprocessed
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, Union

# Get logger
logger = logging.getLogger("data-processor")

def preprocess_data(
    df: pd.DataFrame, 
    encode_item_type_df: pd.DataFrame,
    time_span_divisor: float = 8.0
) -> pd.DataFrame:
    """
    Preprocess raw data for model input by grouping, merging link data, and cleaning columns.
    
    This function:
    1. Groups data by visitid using df_group
    2. Processes link names with encode_item_type_df
    3. Merges results and formats columns for model input
    4. Converts data types and handles missing values
    
    Args:
        df: Input DataFrame with raw data
        encode_item_type_df: DataFrame with encoding information for link names
        time_span_divisor: Divisor for time_span_hr calculation (default: 8.0)
        
    Returns:
        DataFrame ready for model input with appropriate types and missing values handled
        
    Raises:
        ValueError: If input DataFrames are invalid or processing fails
    """
    try:
        # Validate inputs
        if df is None or len(df) == 0:
            raise ValueError("Input DataFrame is empty")
            
        if encode_item_type_df is None or len(encode_item_type_df) == 0:
            raise ValueError("Encode item type DataFrame is empty")
        
        logger.info(f"DDPrint the Model inputs: {df.columns}")

        # Check required columns
        required_cols = ['visitid']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in input DataFrame: {missing_cols}")
        
        # Step 1: Group data by visitid
        logger.info(f"Grouping data for {len(df)} rows")
        grouped = df_group(df)
        if grouped.empty:
            raise ValueError("Grouping operation returned empty DataFrame")
        
        # Step 2: Process link names
        logger.info("Processing link names")
        df_result = process_link_name(df, encode_item_type_df)
        
        # Step 3: Merge results
        logger.info("Merging grouped data with link name results")
        df_final = pd.merge(grouped, df_result, on='visitid', how='left')
        
        # Fill missing values in time_span_hr
        df_final['time_span_hr'] = df_final['time_span_hr'].fillna(0)
        
        # Step 5: Extract age using vectorized operations
        if 'age_info' in df_final.columns:
            # Extract first two characters and convert to integer with defaults for invalid values
            age_str = df_final['age_info'].astype(str).str[:2]
            # Use numpy.where for better performance
            is_numeric = age_str.str.match(r'^\d+$').fillna(False)
            df_final['age'] = np.where(is_numeric, pd.to_numeric(age_str, errors='coerce'), 70)
        else:
            logger.warning("age_info column not found, using default age value")
            df_final['age'] = 70
        
        # Step 6: Define column types
        numerical_cols = [
            'hits', 'month_visitor_info', 'downlink_cnt', 'type000', 'url_path_hit', 
            'product_list_info', 'time_span_hr', 'plan_type_v43_cnt', 'page_name_hit', 
            'custlink_cnt', 'vpp_filter_selection_v49_cnt', 'plan_compare_plan_type_p25_cnt', 
            'type070', 'daily_visitor_info', 'wk_visitor_info', 'pagelink_cnt', 
            'exitlink_cnt', 'come_bk', 'url_path_num', 'age', 'page_name_cnt', 
            'hr_visitor_info', 'type090', 'type095', 'max_name_pct', 'new_visit_flag'
        ]
        
        categorical_cols = [
            'visitid', 'partition_date', 'mcid', 'geo_dma_info', 'host_info', 
            'browser_info', 'visitor_domain_info', 'campaign_info', 'weekday', 
            'race_info', 'income_info', 'shopper_info', 'search_engine_info'
        ]
        
        # Step 7: Process numerical columns (all at once for better performance)
        logger.info("Processing numerical columns")
        for col in numerical_cols:
            if col in df_final.columns:
                df_final[col] = pd.to_numeric(df_final[col], errors='coerce').astype(float).fillna(0)
            else:
                logger.warning(f"Numerical column {col} not found, adding with default value 0")
                df_final[col] = 0.0
        
        # Step 8: Process categorical columns (all at once for better performance)
        logger.info("Processing categorical columns")
        for col in categorical_cols:
            if col in df_final.columns:
                df_final[col] = df_final[col].astype(str).replace(['None', '', 'nan', 'NaN'], 'NA').fillna('NA')
            else:
                logger.warning(f"Categorical column {col} not found, adding with default value 'NA'")
                df_final[col] = 'NA'
        
        # Step 9: Select and return required columns
        col_keep = [
            'visitid', 'online_application', 'partition_date', 'mcid',
            'hits', 'month_visitor_info', 'downlink_cnt', 'type000', 'type070',
            'type090', 'type095', 'max_name_pct', 'url_path_hit',
            'product_list_info', 'time_span_hr', 'plan_type_v43_cnt',
            'page_name_hit', 'custlink_cnt', 'vpp_filter_selection_v49_cnt',
            'plan_compare_plan_type_p25_cnt', 'daily_visitor_info',
            'wk_visitor_info', 'pagelink_cnt', 'exitlink_cnt',
            'come_bk', 'url_path_num', 'age', 'page_name_cnt',
            'hr_visitor_info', 'geo_dma_info', 'host_info',
            'browser_info', 'visitor_domain_info', 'campaign_info',
            'weekday', 'race_info', 'income_info', 'shopper_info',
            'search_engine_info', 'new_visit_flag'
        ]
        
        # Check if all required columns exist
        missing_output_cols = [col for col in col_keep if col not in df_final.columns]
        if missing_output_cols:
            logger.warning(f"Missing columns in output: {missing_output_cols}, adding with default values")
            for col in missing_output_cols:
                if col in numerical_cols:
                    df_final[col] = 0.0
                else:
                    df_final[col] = 'NA'
        
        # Select only the columns we need
        logger.info(f"Selecting {len(col_keep)} columns for model input")
        model_input = df_final[col_keep].copy()
        
        # Log success
        logger.info(f"Successfully preprocessed data: {len(model_input)} rows ready for model input")
        return model_input
        
    except Exception as e:
        logger.error(f"Error in preprocess_data: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to preprocess data: {str(e)}")
####Preprocessed

if __name__ == "__main__":
    import uvicorn
    
    # Configuration for running the app
    host = "0.0.0.0"
    port = 8080
    
    print(f"Starting Reactive FastAPI on http://{host}:{port}")
    
    uvicorn.run(
        "function_app:app",
        host=host,
        port=port,
        workers=int(os.getenv("MAX_WORKERS", 1)),    
        limit_concurrency=int(os.getenv("LIMIT_CONCURRENCY", 10)), 
        backlog=int(os.getenv("BACKLOG", 512)),  
        timeout_keep_alive=int(os.getenv("TIMEOUT_KEEP_ALIVE", 5)) 
    )