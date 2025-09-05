#!/usr/bin/env python3
"""
ARCIS API Service
Handles Raspberry Pi communication and web interface
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import redis
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from arcis_redis_integration import ARCISRedisManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models
class ThreatAlert(BaseModel):
    threat_id: str
    threat_type: str
    threat_level: str
    confidence: float
    distance: Optional[float] = None
    bearing: Optional[float] = None
    timestamp: str
    mission_id: str
    operator_id: str

class RaspberryPiStatus(BaseModel):
    pi_id: str
    status: str
    last_seen: str
    location: Optional[Dict] = None

class SystemStats(BaseModel):
    active_threats: int
    total_detections: int
    system_uptime: str
    redis_status: str

# FastAPI app
app = FastAPI(
    title="ARCIS API Service",
    description="API for ARCIS Tactical Weapon Detection System",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ARCISAPIService:
    """API service for ARCIS system"""
    
    def __init__(self):
        # Initialize Redis connection
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        
        self.redis_manager = ARCISRedisManager(
            redis_host=redis_host,
            redis_port=redis_port
        )
        
        # WebSocket connections for real-time updates
        self.websocket_connections: List[WebSocket] = []
        
        # Raspberry Pi registry
        self.raspberry_pis: Dict[str, RaspberryPiStatus] = {}
        
        logger.info("ARCIS API Service initialized")
        logger.info(f"Redis: {redis_host}:{redis_port}")

# Global service instance
api_service = ARCISAPIService()

@app.on_event("startup")
async def startup_event():
    """Initialize background tasks on startup"""
    logger.info("ARCIS API Service starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("ARCIS API Service shutting down...")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Redis connection
        api_service.redis_manager.redis_client.ping()
        redis_status = "connected"
    except:
        redis_status = "disconnected"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "redis_status": redis_status,
        "service": "arcis_api"
    }

# Raspberry Pi endpoints
@app.post("/api/raspberry-pi/register")
async def register_raspberry_pi(pi_status: RaspberryPiStatus):
    """Register a Raspberry Pi device"""
    api_service.raspberry_pis[pi_status.pi_id] = pi_status
    
    # Store in Redis
    pi_key = f"arcis:raspberry_pi:{pi_status.pi_id}"
    api_service.redis_manager.redis_client.setex(
        pi_key,
        3600,  # 1 hour TTL
        json.dumps(pi_status.dict())
    )
    
    logger.info(f"Registered Raspberry Pi: {pi_status.pi_id}")
    return {"status": "registered", "pi_id": pi_status.pi_id}

@app.get("/api/raspberry-pi/threats")
async def get_threats_for_pi(pi_id: str, limit: int = 10):
    """Get recent threats for a specific Raspberry Pi"""
    try:
        # Get threats from Redis
        threat_keys = api_service.redis_manager.redis_client.keys(f"arcis:pi_threats:*")
        threats = []
        
        for key in sorted(threat_keys, reverse=True)[:limit]:
            threat_data = api_service.redis_manager.redis_client.get(key)
            if threat_data:
                threats.append(json.loads(threat_data))
        
        return {
            "pi_id": pi_id,
            "threats": threats,
            "count": len(threats),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting threats for Pi {pi_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/raspberry-pi/heartbeat")
async def raspberry_pi_heartbeat(pi_id: str):
    """Receive heartbeat from Raspberry Pi"""
    if pi_id in api_service.raspberry_pis:
        api_service.raspberry_pis[pi_id].last_seen = datetime.now().isoformat()
        
        # Update in Redis
        pi_key = f"arcis:raspberry_pi:{pi_id}"
        api_service.redis_manager.redis_client.setex(
            pi_key,
            3600,
            json.dumps(api_service.raspberry_pis[pi_id].dict())
        )
    
    return {"status": "acknowledged", "timestamp": datetime.now().isoformat()}

# Threat monitoring endpoints
@app.get("/api/threats/active")
async def get_active_threats():
    """Get currently active threats"""
    try:
        # Get recent threats from Redis
        threat_keys = api_service.redis_manager.redis_client.keys("arcis:pi_threats:*")
        active_threats = []
        
        current_time = datetime.now()
        
        for key in threat_keys:
            threat_data = api_service.redis_manager.redis_client.get(key)
            if threat_data:
                threat = json.loads(threat_data)
                # Consider threats active if they're less than 5 minutes old
                threat_time = datetime.fromisoformat(threat['timestamp'].replace('Z', '+00:00').replace('+00:00', ''))
                if (current_time - threat_time).total_seconds() < 300:  # 5 minutes
                    active_threats.append(threat)
        
        return {
            "active_threats": active_threats,
            "count": len(active_threats),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting active threats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/threats/statistics")
async def get_threat_statistics():
    """Get threat statistics"""
    try:
        stats = api_service.redis_manager.get_threat_statistics()
        
        # Add additional statistics
        stats['api_service'] = {
            'registered_pis': len(api_service.raspberry_pis),
            'websocket_connections': len(api_service.websocket_connections),
            'timestamp': datetime.now().isoformat()
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mission management endpoints
@app.get("/api/missions/{mission_id}")
async def get_mission_data(mission_id: str):
    """Get mission data"""
    try:
        mission_data = api_service.redis_manager.get_mission_data(mission_id)
        
        if not mission_data:
            raise HTTPException(status_code=404, detail="Mission not found")
        
        return mission_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting mission data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/missions")
async def list_missions():
    """List all missions"""
    try:
        mission_keys = api_service.redis_manager.redis_client.keys("arcis:mission_data:*")
        missions = []
        
        for key in mission_keys:
            mission_data = api_service.redis_manager.redis_client.get(key)
            if mission_data:
                data = json.loads(mission_data)
                missions.append({
                    'mission_id': data.get('mission_id'),
                    'start_time': data.get('start_time'),
                    'operator': data.get('operator'),
                    'total_detections': len(data.get('detections', []))
                })
        
        return {
            "missions": missions,
            "count": len(missions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error listing missions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time updates
@app.websocket("/ws/threats")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time threat updates"""
    await websocket.accept()
    api_service.websocket_connections.append(websocket)
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive and listen for Redis pub/sub
        pubsub = api_service.redis_manager.redis_client.pubsub()
        pubsub.subscribe('arcis:pi_channel')
        
        while True:
            # Check for new messages from Redis
            message = pubsub.get_message(timeout=1)
            if message and message['type'] == 'message':
                threat_data = json.loads(message['data'])
                await websocket.send_json({
                    "type": "threat_update",
                    "data": threat_data,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Send periodic heartbeat
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in api_service.websocket_connections:
            api_service.websocket_connections.remove(websocket)
        pubsub.close()

# System monitoring endpoints
@app.get("/api/system/status")
async def get_system_status():
    """Get overall system status"""
    try:
        # Get Redis statistics
        redis_stats = api_service.redis_manager.get_threat_statistics()
        
        # Get cloud service statistics if available
        cloud_stats_data = api_service.redis_manager.redis_client.get('arcis:cloud_service_stats')
        cloud_stats = json.loads(cloud_stats_data) if cloud_stats_data else {}
        
        return {
            "system_status": "operational",
            "redis_stats": redis_stats,
            "cloud_service_stats": cloud_stats,
            "api_service": {
                "registered_pis": len(api_service.raspberry_pis),
                "websocket_connections": len(api_service.websocket_connections)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task to clean up expired data
async def cleanup_expired_data():
    """Background task to clean up expired data"""
    while True:
        try:
            api_service.redis_manager.cleanup_expired_data()
            await asyncio.sleep(300)  # Run every 5 minutes
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(60)

def main():
    """Main entry point"""
    port = int(os.getenv('API_PORT', 8080))
    host = os.getenv('API_HOST', '0.0.0.0')
    
    logger.info(f"Starting ARCIS API Service on {host}:{port}")
    
    uvicorn.run(
        "arcis_api_service:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main() 