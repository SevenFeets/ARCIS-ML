import redis
import json
import time
import base64
import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading
import logging
from dataclasses import dataclass, asdict
import asyncio
import aioredis
from google.cloud import vision
import requests
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ThreatDetection:
    """Data class for threat detection information"""
    timestamp: str
    threat_type: str
    confidence: float
    threat_level: str
    distance: float
    bearing: float
    bbox: tuple
    frame_id: str
    mission_id: str
    operator_id: str
    gps_coords: Optional[Dict] = None
    engagement_rec: Optional[Dict] = None

class ARCISRedisManager:
    """Redis manager for ARCIS system with caching, messaging, and cloud integration"""
    
    def __init__(self, 
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 google_cloud_project: str = None):
        
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=False  # Keep binary for images
        )
        
        # Test Redis connection
        try:
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
        except redis.ConnectionError:
            logger.error(f"Failed to connect to Redis at {redis_host}:{redis_port}")
            raise
        
        # Google Cloud Vision client
        self.vision_client = None
        if google_cloud_project:
            try:
                self.vision_client = vision.ImageAnnotatorClient()
                logger.info("Google Cloud Vision client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Google Cloud Vision: {e}")
        
        # Redis keys
        self.CRITICAL_FRAMES_KEY = "arcis:critical_frames"
        self.THREAT_QUEUE_KEY = "arcis:threat_queue"
        self.YELLOW_THREATS_KEY = "arcis:yellow_threats"
        self.ORANGE_RED_THREATS_KEY = "arcis:orange_red_threats"
        self.RASPBERRY_PI_QUEUE = "arcis:raspberry_pi_queue"
        self.WEBSITE_QUEUE = "arcis:website_queue"
        self.MISSION_DATA_KEY = "arcis:mission_data"
        
        # TTL settings (in seconds)
        self.CRITICAL_FRAME_TTL = 600  # 10 minutes
        self.THREAT_DATA_TTL = 3600    # 1 hour
        self.MISSION_DATA_TTL = 86400  # 24 hours
        
        # Start background workers
        self._start_background_workers()
    
    def _start_background_workers(self):
        """Start background worker threads"""
        # Worker for processing yellow threats with Google Cloud Vision
        threading.Thread(target=self._process_yellow_threats_worker, daemon=True).start()
        
        # Worker for sending data to website
        threading.Thread(target=self._website_sender_worker, daemon=True).start()
        
        # Worker for Raspberry Pi communication
        threading.Thread(target=self._raspberry_pi_worker, daemon=True).start()
        
        logger.info("Background workers started")
    
    def cache_critical_frame(self, frame: np.ndarray, threat_data: ThreatDetection) -> str:
        """
        Cache critical frames with 10-minute TTL
        
        Args:
            frame: OpenCV frame (numpy array)
            threat_data: Threat detection information
            
        Returns:
            frame_id: Unique identifier for the cached frame
        """
        frame_id = f"frame_{int(time.time() * 1000)}_{threat_data.threat_level}"
        
        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        # Create frame data
        frame_data = {
            'frame_bytes': base64.b64encode(frame_bytes).decode('utf-8'),
            'threat_data': asdict(threat_data),
            'cached_at': datetime.now().isoformat()
        }
        
        # Cache with TTL
        key = f"{self.CRITICAL_FRAMES_KEY}:{frame_id}"
        self.redis_client.setex(
            key,
            self.CRITICAL_FRAME_TTL,
            json.dumps(frame_data)
        )
        
        logger.info(f"Cached critical frame {frame_id} with {threat_data.threat_level} threat")
        return frame_id
    
    def get_cached_frame(self, frame_id: str) -> Optional[tuple]:
        """
        Retrieve cached frame and threat data
        
        Returns:
            (frame, threat_data) or None if not found
        """
        key = f"{self.CRITICAL_FRAMES_KEY}:{frame_id}"
        data = self.redis_client.get(key)
        
        if not data:
            return None
        
        frame_data = json.loads(data)
        
        # Decode frame
        frame_bytes = base64.b64decode(frame_data['frame_bytes'])
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        return frame, frame_data['threat_data']
    
    def publish_threat_detection(self, threat_data: ThreatDetection, frame: np.ndarray = None):
        """
        Publish threat detection to appropriate queues based on threat level
        """
        threat_message = {
            'detection': asdict(threat_data),
            'timestamp': datetime.now().isoformat(),
            'frame_id': None
        }
        
        # Cache frame for CRITICAL and HIGH threats
        if threat_data.threat_level in ['CRITICAL', 'HIGH'] and frame is not None:
            frame_id = self.cache_critical_frame(frame, threat_data)
            threat_message['frame_id'] = frame_id
        
        # Route based on threat level
        if threat_data.threat_level == 'MEDIUM':
            # Send YELLOW threats to Google Cloud Vision queue
            self.redis_client.lpush(self.YELLOW_THREATS_KEY, json.dumps(threat_message))
            logger.info(f"Queued YELLOW threat for Cloud Vision analysis")
            
        elif threat_data.threat_level in ['CRITICAL', 'HIGH']:
            # Send ORANGE/RED threats to website immediately
            self.redis_client.lpush(self.ORANGE_RED_THREATS_KEY, json.dumps(threat_message))
            logger.info(f"Queued {threat_data.threat_level} threat for immediate website notification")
        
        # Always send to Raspberry Pi queue for local processing
        self.redis_client.lpush(self.RASPBERRY_PI_QUEUE, json.dumps(threat_message))
        
        # General threat queue
        self.redis_client.lpush(self.THREAT_QUEUE_KEY, json.dumps(threat_message))
    
    def _process_yellow_threats_worker(self):
        """Background worker to process YELLOW threats with Google Cloud Vision"""
        while True:
            try:
                # Block and wait for yellow threats
                _, message = self.redis_client.brpop(self.YELLOW_THREATS_KEY, timeout=1)
                if not message:
                    continue
                
                threat_message = json.loads(message)
                frame_id = threat_message.get('frame_id')
                
                if frame_id and self.vision_client:
                    # Get cached frame
                    frame_data = self.get_cached_frame(frame_id)
                    if frame_data:
                        frame, threat_data = frame_data
                        
                        # Analyze with Google Cloud Vision
                        vision_result = self._analyze_with_cloud_vision(frame)
                        
                        # Update threat data with Cloud Vision results
                        enhanced_threat = threat_message.copy()
                        enhanced_threat['cloud_vision_analysis'] = vision_result
                        enhanced_threat['processed_at'] = datetime.now().isoformat()
                        
                        # If Cloud Vision confirms threat, escalate
                        if self._should_escalate_threat(vision_result):
                            enhanced_threat['detection']['threat_level'] = 'HIGH'
                            enhanced_threat['escalated'] = True
                            
                            # Send to website queue
                            self.redis_client.lpush(self.WEBSITE_QUEUE, json.dumps(enhanced_threat))
                            logger.info("Escalated YELLOW threat to HIGH based on Cloud Vision")
                        else:
                            # Still send to website but as confirmed MEDIUM
                            enhanced_threat['confirmed_medium'] = True
                            self.redis_client.lpush(self.WEBSITE_QUEUE, json.dumps(enhanced_threat))
                
            except Exception as e:
                logger.error(f"Error in yellow threats worker: {e}")
                time.sleep(1)
    
    def _analyze_with_cloud_vision(self, frame: np.ndarray) -> Dict:
        """Analyze frame with Google Cloud Vision API"""
        try:
            # Encode frame for Cloud Vision
            _, buffer = cv2.imencode('.jpg', frame)
            image_bytes = buffer.tobytes()
            
            image = vision.Image(content=image_bytes)
            
            # Perform object detection
            objects = self.vision_client.object_localization(image=image).localized_object_annotations
            
            # Perform label detection
            labels = self.vision_client.label_detection(image=image).label_annotations
            
            # Perform safe search (for violence detection)
            safe_search = self.vision_client.safe_search_detection(image=image).safe_search_annotation
            
            return {
                'objects': [{'name': obj.name, 'score': obj.score} for obj in objects],
                'labels': [{'description': label.description, 'score': label.score} for label in labels[:10]],
                'safe_search': {
                    'violence': safe_search.violence.name,
                    'violence_score': safe_search.violence
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Cloud Vision analysis failed: {e}")
            return {'error': str(e)}
    
    def _should_escalate_threat(self, vision_result: Dict) -> bool:
        """Determine if threat should be escalated based on Cloud Vision results"""
        if 'error' in vision_result:
            return False
        
        # Check for weapon-related objects
        weapon_objects = ['weapon', 'gun', 'rifle', 'knife', 'firearm']
        for obj in vision_result.get('objects', []):
            if any(weapon in obj['name'].lower() for weapon in weapon_objects):
                if obj['score'] > 0.7:  # High confidence
                    return True
        
        # Check for weapon-related labels
        weapon_labels = ['weapon', 'gun', 'firearm', 'rifle', 'pistol', 'knife']
        for label in vision_result.get('labels', []):
            if any(weapon in label['description'].lower() for weapon in weapon_labels):
                if label['score'] > 0.8:  # Very high confidence
                    return True
        
        # Check violence detection
        safe_search = vision_result.get('safe_search', {})
        if safe_search.get('violence') in ['LIKELY', 'VERY_LIKELY']:
            return True
        
        return False
    
    def _website_sender_worker(self):
        """Background worker to send data to website"""
        while True:
            try:
                # Process orange/red threats first (higher priority)
                _, message = self.redis_client.brpop([self.ORANGE_RED_THREATS_KEY, self.WEBSITE_QUEUE], timeout=1)
                if not message:
                    continue
                
                threat_data = json.loads(message)
                self._send_to_website(threat_data)
                
            except Exception as e:
                logger.error(f"Error in website sender worker: {e}")
                time.sleep(1)
    
    def _send_to_website(self, threat_data: Dict):
        """Send threat data to website API"""
        try:
            website_url = os.getenv('WEBSITE_API_URL')
            api_key = os.getenv('WEBSITE_API_KEY')
            
            if not website_url:
                logger.warning("WEBSITE_API_URL not configured")
                return
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}' if api_key else None
            }
            
            # Remove None values from headers
            headers = {k: v for k, v in headers.items() if v is not None}
            
            response = requests.post(
                f"{website_url}/api/threats",
                json=threat_data,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully sent threat data to website")
            else:
                logger.error(f"Website API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Failed to send data to website: {e}")
    
    def _raspberry_pi_worker(self):
        """Background worker for Raspberry Pi communication"""
        while True:
            try:
                _, message = self.redis_client.brpop(self.RASPBERRY_PI_QUEUE, timeout=1)
                if not message:
                    continue
                
                threat_data = json.loads(message)
                
                # Store in a specific key for Raspberry Pi to consume
                pi_key = f"arcis:pi_threats:{int(time.time())}"
                self.redis_client.setex(pi_key, 3600, message)  # 1 hour TTL
                
                # Publish to Pi channel
                self.redis_client.publish('arcis:pi_channel', message)
                
                logger.info("Sent threat data to Raspberry Pi")
                
            except Exception as e:
                logger.error(f"Error in Raspberry Pi worker: {e}")
                time.sleep(1)
    
    def store_mission_data(self, mission_id: str, mission_data: Dict):
        """Store mission data with 24-hour TTL"""
        key = f"{self.MISSION_DATA_KEY}:{mission_id}"
        self.redis_client.setex(
            key,
            self.MISSION_DATA_TTL,
            json.dumps(mission_data)
        )
        logger.info(f"Stored mission data for {mission_id}")
    
    def get_mission_data(self, mission_id: str) -> Optional[Dict]:
        """Retrieve mission data"""
        key = f"{self.MISSION_DATA_KEY}:{mission_id}"
        data = self.redis_client.get(key)
        return json.loads(data) if data else None
    
    def get_threat_statistics(self) -> Dict:
        """Get real-time threat statistics"""
        stats = {
            'critical_frames_cached': len(self.redis_client.keys(f"{self.CRITICAL_FRAMES_KEY}:*")),
            'yellow_threats_pending': self.redis_client.llen(self.YELLOW_THREATS_KEY),
            'orange_red_threats_pending': self.redis_client.llen(self.ORANGE_RED_THREATS_KEY),
            'website_queue_size': self.redis_client.llen(self.WEBSITE_QUEUE),
            'raspberry_pi_queue_size': self.redis_client.llen(self.RASPBERRY_PI_QUEUE),
            'total_threats_processed': self.redis_client.llen(self.THREAT_QUEUE_KEY)
        }
        return stats
    
    def cleanup_expired_data(self):
        """Manual cleanup of expired data (Redis handles TTL automatically)"""
        # This is mainly for logging purposes
        expired_count = 0
        for key in self.redis_client.keys(f"{self.CRITICAL_FRAMES_KEY}:*"):
            ttl = self.redis_client.ttl(key)
            if ttl == -2:  # Key doesn't exist (expired)
                expired_count += 1
        
        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired critical frames")

# Integration with existing ARCIS system
class ARCISRedisIntegration:
    """Integration layer for existing ARCIS detection system"""
    
    def __init__(self, redis_manager: ARCISRedisManager):
        self.redis_manager = redis_manager
    
    def process_detection(self, detection_data: Dict, frame: np.ndarray, mission_id: str, operator_id: str):
        """Process detection from main ARCIS system"""
        
        # Convert to ThreatDetection object
        threat = ThreatDetection(
            timestamp=datetime.now().isoformat(),
            threat_type=detection_data['class_name'],
            confidence=detection_data['confidence'],
            threat_level=detection_data['threat_level'],
            distance=detection_data.get('distance', 0),
            bearing=detection_data.get('bearing', 0),
            bbox=detection_data['bbox'],
            frame_id='',  # Will be set by Redis manager
            mission_id=mission_id,
            operator_id=operator_id,
            gps_coords=detection_data.get('gps_coords'),
            engagement_rec=detection_data.get('engagement')
        )
        
        # Publish to Redis
        self.redis_manager.publish_threat_detection(threat, frame)
        
        return threat 