# backend/main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from typing import List, Optional, Dict
import json
import os
import logging
from bson import ObjectId
from datetime import datetime
from pydantic import BaseModel
import google.generativeai as genai
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Infrastructure Monitoring API", version="1.0.0")

# CORS configuration for React frontend
origins = [
    "http://localhost:3001",  # React development server
    "http://localhost:5173",  # Vite development server
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection with error handling
MONGO_URI = 'mongodb+srv://mvishaalgokul8:IMTXb7QXknOIgFaw@infrahealth.vdxwhfq.mongodb.net/'
DB_NAME = 'logs'

# Gemini AI configuration
GEMINI_API_KEY = "AIzaSyBBh6qma7uR8pJdBOEGHOu1HOTEsyb0Xks"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Initialize MongoDB connections
try:
    mongo_client = MongoClient(MONGO_URI)
    # Test the connection
    mongo_client.admin.command('ping')
    db = mongo_client[DB_NAME]
    app_col = db['app']
    cpu_col = db['server']
    server_col = db['server']  # Alias for consistency
    logs_col = db['app']  # Alias for logs
    
    # LLM database for analysis
    llm_db = mongo_client["llm_response"]
    analysis_collection = llm_db["LogAnalysis"]
    
    # Chat collection
    chat_col = db['chat_history']
    
    logger.info("✅ MongoDB connection successful")
except Exception as e:
    logger.error(f"❌ MongoDB connection failed: {e}")
    # Continue without crashing - let the API handle individual request errors

# Custom JSON encoder for MongoDB ObjectId
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    context: Optional[Dict] = {}
    timestamp: Optional[str] = None

# Utility functions
def clean_gemini_response(response_text: str) -> str:
    """Clean Gemini response from markdown formatting"""
    # Remove markdown code blocks
    response_text = re.sub(r'``````', '', response_text)
    response_text = re.sub(r'```.*?```', '', response_text, flags=re.DOTALL)
    
    # Remove any extra whitespace
    response_text = response_text.strip()
    
    return response_text

# Initialize chat collection with index
try:
    chat_col.create_index([("timestamp", -1), ("session_id", 1)])
    logger.info("Chat collection initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize chat collection: {e}")

# Chat endpoints
@app.post("/api/chat/message")
async def send_chat_message(request: ChatMessage):
    try:
        message = request.message.lower()
        context = request.context or {}
        
        # Check if user is asking about specific data
        infrastructure_data = {}
        
        # Query recent server metrics if relevant
        if any(keyword in message for keyword in ['server', 'cpu', 'memory', 'performance', 'metrics']):
            try:
                recent_metrics = list(server_col.find().sort("timestamp", -1).limit(5))
                infrastructure_data['recent_metrics'] = [
                    {
                        "server": metric.get("server_name", metric.get("server", "Unknown")),
                        "cpu": metric.get("cpu_usage", 0),
                        "memory": metric.get("memory_usage", 0),
                        "status": metric.get("health_status", metric.get("server_health", "Unknown")),
                        "environment": metric.get("environment", "Unknown")
                    } for metric in recent_metrics
                ]
            except Exception as e:
                logger.warning(f"Could not fetch metrics: {e}")
        
        # Query recent logs if relevant
        if any(keyword in message for keyword in ['log', 'error', 'issue', 'problem']):
            try:
                recent_logs = list(logs_col.find(
                    {"level": {"$in": ["ERROR", "WARN", "CRITICAL"]}}
                ).sort("timestamp", -1).limit(3))
                infrastructure_data['recent_issues'] = [
                    {
                        "level": log.get("level"),
                        "message": log.get("message", "")[:100],
                        "application": log.get("application", log.get("app_name", "Unknown")),
                        "environment": log.get("environment", "Unknown")
                    } for log in recent_logs
                ]
            except Exception as e:
                logger.warning(f"Could not fetch logs: {e}")
        
        # Create enhanced prompt with real data
        data_context = ""
        if infrastructure_data:
            data_context = f"\nCurrent Infrastructure Data:\n{json.dumps(infrastructure_data, indent=2)}\n"
        
        prompt = f"""
        You are an intelligent infrastructure monitoring assistant with access to real-time data.
        
        Available environments: Dev, Staging, Production, QA
        Available applications: app1, app2, app3
        Available servers: server1, server2, server3
        
        {data_context}
        
        User question: {request.message}
        
        Provide specific, actionable responses based on the available data. If you see concerning metrics or errors, highlight them and suggest next steps.
        Keep responses concise but informative. Focus on infrastructure monitoring, server health, and system performance.
        """
        
        response = model.generate_content(prompt)
        response_text = clean_gemini_response(response.text)
        
        # Store chat history
        chat_record = {
            "user_message": request.message,
            "bot_response": response_text,
            "context": {**context, "infrastructure_data": infrastructure_data},
            "timestamp": datetime.utcnow(),
            "session_id": context.get("session_id", "default")
        }
        
        # Insert into chat collection
        chat_col.insert_one(chat_record)
        
        return {
            "response": response_text,
            "context": {
                "model": "gemini-1.5-flash",
                "infrastructure_data": infrastructure_data,
                "timestamp": datetime.utcnow().isoformat()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Enhanced chat error: {str(e)}")
        return {
            "response": "I'm having trouble accessing the infrastructure data right now. Please try again in a moment.",
            "context": {"error": True},
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/api/chat/history")
async def get_chat_history(limit: int = 50, session_id: str = "default"):
    try:
        # Get recent chat history
        history = list(chat_col.find(
            {"session_id": session_id},
            {"_id": 0}
        ).sort("timestamp", -1).limit(limit))
        
        # Convert to frontend format
        formatted_history = []
        for chat in reversed(history):  # Reverse to show oldest first
            # Add user message
            formatted_history.append({
                "id": f"user_{int(chat['timestamp'].timestamp() * 1000)}",
                "type": "user",
                "content": chat["user_message"],
                "timestamp": chat["timestamp"].isoformat()
            })
            
            # Add bot response
            formatted_history.append({
                "id": f"bot_{int(chat['timestamp'].timestamp() * 1000)}",
                "type": "bot",
                "content": chat["bot_response"],
                "context": chat.get("context", {}),
                "timestamp": chat["timestamp"].isoformat()
            })
        
        return formatted_history
        
    except Exception as e:
        logger.error(f"Chat history error: {str(e)}")
        return []

# Existing API endpoints (unchanged)
@app.get("/api/log_analysis/{log_id}")
def get_log_analysis(log_id: str):
    # Clean and validate the log_id
    log_id = log_id.strip()
    
    if not log_id:
        return JSONResponse(status_code=400, content={"error": "Invalid log ID"})
    
    logger.info(f"Looking for original_log_id: '{log_id}'")
    
    try:
        # Try multiple search strategies
        doc = None
        
        # Strategy 1: Search by original_log_id as string
        doc = analysis_collection.find_one({"original_log_id": log_id})
        
        # Strategy 2: If not found and looks like ObjectId, try as ObjectId
        if not doc and ObjectId.is_valid(log_id):
            doc = analysis_collection.find_one({"original_log_id": ObjectId(log_id)})
            
        # Strategy 3: Search by document's own _id
        if not doc and ObjectId.is_valid(log_id):
            doc = analysis_collection.find_one({"_id": ObjectId(log_id)})
            
        if not doc:
            return JSONResponse(status_code=404, content={"error": "Analysis not found"})
            
        # Convert ObjectIds to strings for JSON serialization
        doc["_id"] = str(doc["_id"])
        if "original_log_id" in doc:
            doc["original_log_id"] = str(doc["original_log_id"])
            
        return doc
        
    except Exception as e:
        logger.error(f"Error in get_log_analysis: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

@app.get("/api/health")
async def health_check():
    try:
        # Test MongoDB connection
        mongo_client.admin.command('ping')
        return {
            "status": "healthy", 
            "timestamp": datetime.utcnow(),
            "mongodb": "connected"
        }
    except Exception as e:
        return {
            "status": "degraded",
            "timestamp": datetime.utcnow(),
            "mongodb": "disconnected",
            "error": str(e)
        }

@app.get("/api/combined_logs")
async def get_combined_logs(limit: int = 50, environment: Optional[str] = None, 
                           server: Optional[str] = None, app_name: Optional[str] = None):
    try:
        # Build query filters
        query = {}
        if environment:
            query["environment"] = environment
        if server:
            query["server"] = server
        if app_name:
            query["app_name"] = app_name

        # Get app logs
        app_logs = list(app_col.find(query).sort("createdAt", -1).limit(limit))
        
        # Format logs for frontend
        formatted_logs = []
        for log in app_logs:
            formatted_log = {
                "log": {
                    "id": str(log.get("_id")),
                    "timestamp": log.get("timestamp", ""),
                    "level": log.get("level", "INFO"),
                    "message": log.get("message", ""),
                    "source": f"{log.get('environment', 'Unknown')}/{log.get('server', 'Unknown')}/{log.get('app_name', 'Unknown')}",
                    "environment": log.get("environment"),
                    "server": log.get("server"),
                    "app_name": log.get("app_name"),
                    "logger": log.get("logger"),
                    "thread": log.get("thread"),
                    "pid": log.get("pid"),
                    "exception_type": log.get("exception_type"),
                    "exception_message": log.get("exception_message"),
                    "stacktrace": log.get("stacktrace"),
                    "createdAt": log.get("createdAt")
                }
            }
            formatted_logs.append(formatted_log)
        
        return formatted_logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/debug/app_deployment/{app_name}")
async def debug_app_deployment(app_name: str, environment: Optional[str] = None):
    try:
        pipeline = [{"$match": {"app_name": app_name}}]
        
        if environment:
            pipeline["$match"]["environment"] = environment
            
        pipeline.append({
            "$group": {
                "_id": {"server": "$server", "environment": "$environment"},
                "log_count": {"$sum": 1}
            }
        })
        
        result = list(app_col.aggregate(pipeline))
        return {
            "app": app_name,
            "environment": environment,
            "deployed_on": result
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/debug/all_servers")
async def debug_all_servers():
    try:
        # Get all unique server combinations
        pipeline = [
            {"$match": {
                "server": {"$ne": None, "$ne": "", "$exists": True},
                "environment": {"$ne": None, "$ne": "", "$exists": True}
            }},
            {"$group": {
                "_id": {"server": "$server", "environment": "$environment"},
                "last_seen": {"$max": "$createdAt"},
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id.environment": 1, "_id.server": 1}}
        ]
        
        all_servers = list(cpu_col.aggregate(pipeline))
        return {
            "total_unique_servers": len(all_servers),
            "servers": all_servers
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/server_metrics")
async def get_server_metrics(request: Request, environment: Optional[str] = None, app_name: Optional[str] = None):
    try:
        logger.info(f"DEBUG: Received params - environment: {environment}, app_name: {app_name}")
        
        # If app_name is specified, get servers where this app is deployed
        if app_name and app_name.strip():
            logger.info(f"DEBUG: Filtering by app: {app_name}")
            
            # First, find servers where this app runs
            app_match = {"app_name": app_name}
            if environment:
                app_match["environment"] = environment
                
            app_servers = list(app_col.aggregate([
                {"$match": app_match},
                {"$group": {"_id": {"server": "$server", "environment": "$environment"}}}
            ]))
            
            if not app_servers:
                logger.info(f"DEBUG: No servers found for app {app_name}")
                return []
            
            # Build server filter conditions
            server_conditions = []
            for server_combo in app_servers:
                server_conditions.append({
                    "server": server_combo["_id"]["server"],
                    "environment": server_combo["_id"]["environment"]
                })
            
            logger.info(f"DEBUG: Server conditions: {server_conditions}")
            
            # Query server metrics ONLY for these specific servers
            pipeline = [
                {
                    "$match": {
                        "$or": server_conditions,
                        "server": {"$ne": None, "$ne": "", "$exists": True},
                        "environment": {"$ne": None, "$ne": "", "$exists": True}
                    }
                }
            ]
        else:
            logger.info(f"DEBUG: No app filter, using environment only: {environment}")
            # No app filter, use environment filter if provided
            match_condition = {
                "server": {"$ne": None, "$ne": "", "$exists": True},
                "environment": {"$ne": None, "$ne": "", "$exists": True}
            }
            if environment:
                match_condition["environment"] = environment
                
            pipeline = [{"$match": match_condition}]
        
        logger.info(f"DEBUG: Pipeline match stage: {pipeline}")
        
        # Add grouping to get latest metric per server
        pipeline.extend([
            {"$sort": {"createdAt": -1}},
            {
                "$group": {
                    "_id": {"server": "$server", "environment": "$environment"},
                    "latest_metric": {"$first": "$$ROOT"}
                }
            },
            {"$replaceRoot": {"newRoot": "$latest_metric"}},
            {"$sort": {"environment": 1, "server": 1}},
            {"$limit": 50}
        ])
        
        server_logs = list(cpu_col.aggregate(pipeline))
        logger.info(f"DEBUG: Found {len(server_logs)} server logs after aggregation")
        
        # Safe float conversion function
        def safe_float(value, default=0.0):
            if value is None or value == '':
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        # Process metrics
        metrics = []
        for log in server_logs:
            try:
                metric = {
                    "id": str(log.get("_id")),
                    "timestamp": log.get("timestamp"),
                    "environment": log.get("environment"),
                    "server": log.get("server"),
                    "cpu_usage": safe_float(log.get("cpu_usage")),
                    "cpu_temp": safe_float(log.get("cpu_temp")),
                    "memory_usage": safe_float(log.get("memory_usage")),
                    "disk_utilization": safe_float(log.get("disk_utilization")),
                    "server_health": log.get("server_health"),
                    "ip_address": log.get("ip_address"),
                    "createdAt": log.get("createdAt")
                }
                metrics.append(metric)
                logger.info(f"DEBUG: Added metric for {metric['environment']}/{metric['server']}")
            except Exception as e:
                logger.error(f"ERROR: Processing server log {log.get('_id')}: {e}")
                continue
        
        logger.info(f"DEBUG: Returning {len(metrics)} server metrics for app: {app_name}, env: {environment}")
        return metrics
        
    except Exception as e:
        logger.error(f"ERROR: Server metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/environments")
async def get_environments():
    try:
        environments = app_col.distinct("environment")
        return {"environments": environments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/applications")
async def get_applications(environment: Optional[str] = None):
    try:
        query = {}
        if environment:
            query["environment"] = environment
            
        apps = app_col.distinct("app_name", query)
        servers = app_col.distinct("server", query)
        
        app_server_mapping = {}
        for app in apps:
            app_query = {"app_name": app}
            if environment:
                app_query["environment"] = environment
            app_servers = app_col.distinct("server", app_query)
            app_server_mapping[app] = app_servers
        
        return {
            "applications": apps,
            "servers": servers,
            "app_server_mapping": app_server_mapping
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dashboard_stats")
async def get_dashboard_stats():
    try:
        # Get error counts by environment
        error_pipeline = [
            {"$match": {"level": "ERROR"}},
            {"$group": {"_id": "$environment", "count": {"$sum": 1}}}
        ]
        error_stats = list(app_col.aggregate(error_pipeline))
        
        # Get UNIQUE server health stats (fixed)
        health_pipeline = [
            {
                "$group": {
                    "_id": {
                        "server": "$server",
                        "environment": "$environment",
                        "server_health": "$server_health"
                    }
                }
            },
            {
                "$group": {
                    "_id": "$_id.server_health",
                    "count": {"$sum": 1}
                }
            }
        ]
        health_stats = list(cpu_col.aggregate(health_pipeline))
        
        # Alternative: Get latest health status per server
        latest_health_pipeline = [
            {"$sort": {"createdAt": -1}},
            {
                "$group": {
                    "_id": {"server": "$server", "environment": "$environment"},
                    "latest_health": {"$first": "$server_health"}
                }
            },
            {
                "$group": {
                    "_id": "$latest_health",
                    "count": {"$sum": 1}
                }
            }
        ]
        latest_health_stats = list(cpu_col.aggregate(latest_health_pipeline))
        
        # Get application distribution
        app_pipeline = [
            {"$group": {"_id": {"environment": "$environment", "app_name": "$app_name"}, "count": {"$sum": 1}}}
        ]
        app_stats = list(app_col.aggregate(app_pipeline))
        
        return {
            "error_stats": error_stats,
            "health_stats": health_stats,  # Unique server count
            "latest_health_stats": latest_health_stats,  # Latest status per server
            "app_stats": app_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Static file serving for production
frontend_build_path = "../frontend/build"
static_path = "../frontend/build/static"

if os.path.exists(frontend_build_path) and os.path.exists(static_path):
    # Mount static files for production
    app.mount("/static", StaticFiles(directory=static_path), name="static")
    templates = Jinja2Templates(directory=frontend_build_path)
    
    @app.get("/{rest_of_path:path}")
    async def serve_react_app(request: Request, rest_of_path: str):
        return templates.TemplateResponse("index.html", {"request": request})
    
    logger.info("✅ Static file serving enabled")
else:
    logger.info("⚠️  Frontend build directory not found. Running in API-only mode.")
    logger.info("   Build your React app first: cd ../infraapp/infraapp && npm run build")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
