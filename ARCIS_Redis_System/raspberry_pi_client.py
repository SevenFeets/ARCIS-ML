#!/usr/bin/env python3
"""
ARCIS Raspberry Pi Client
Connects to ARCIS system and receives threat notifications
"""

import os
import json
import time
import logging
import requests
import redis
from datetime import datetime
from typing import Dict, Optional
import threading
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ARCISRaspberryPiClient:
    """Raspberry Pi client for ARCIS system"""
    
    def __init__(self, 
                 pi_id: str,
                 arcis_api_url: str,
                 redis_host: str = 'localhost',
                 redis_port: int = 6379):
        
        self.pi_id = pi_id
        self.arcis_api_url = arcis_api_url.rstrip('/')
        self.redis_host = redis_host
        self.redis_port = redis_port
        
        # Initialize Redis connection
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
        except redis.ConnectionError:
            logger.error(f"Failed to connect to Redis at {redis_host}:{redis_port}")
            self.redis_client = None
        
        # Client state
        self.running = False
        self.last_heartbeat = None
        self.threat_count = 0
        
        # Threat handlers
        self.threat_handlers = {
            'CRITICAL': self._handle_critical_threat,
            'HIGH': self._handle_high_threat,
            'MEDIUM': self._handle_medium_threat,
            'LOW': self._handle_low_threat
        }
        
        logger.info(f"ARCIS Raspberry Pi Client initialized")
        logger.info(f"Pi ID: {pi_id}")
        logger.info(f"API URL: {arcis_api_url}")
    
    def register_with_api(self) -> bool:
        """Register this Pi with the ARCIS API service"""
        try:
            registration_data = {
                "pi_id": self.pi_id,
                "status": "online",
                "last_seen": datetime.now().isoformat(),
                "location": {
                    "description": "Field deployment",
                    "coordinates": None  # Add GPS coordinates if available
                }
            }
            
            response = requests.post(
                f"{self.arcis_api_url}/api/raspberry-pi/register",
                json=registration_data,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Successfully registered with ARCIS API")
                return True
            else:
                logger.error(f"Registration failed: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Registration request failed: {e}")
            return False
    
    def send_heartbeat(self):
        """Send heartbeat to API service"""
        try:
            response = requests.post(
                f"{self.arcis_api_url}/api/raspberry-pi/heartbeat",
                params={"pi_id": self.pi_id},
                timeout=5
            )
            
            if response.status_code == 200:
                self.last_heartbeat = datetime.now()
                logger.debug("Heartbeat sent successfully")
            else:
                logger.warning(f"Heartbeat failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Heartbeat request failed: {e}")
    
    def listen_for_threats(self):
        """Listen for threat notifications from Redis"""
        if not self.redis_client:
            logger.error("Redis not available, cannot listen for threats")
            return
        
        logger.info("Starting threat listener...")
        
        # Subscribe to threat channel
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe('arcis:pi_channel')
        
        try:
            while self.running:
                message = pubsub.get_message(timeout=1)
                
                if message and message['type'] == 'message':
                    try:
                        threat_data = json.loads(message['data'])
                        self._process_threat(threat_data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode threat message: {e}")
                    except Exception as e:
                        logger.error(f"Error processing threat: {e}")
                
        except Exception as e:
            logger.error(f"Error in threat listener: {e}")
        finally:
            pubsub.close()
    
    def _process_threat(self, threat_data: Dict):
        """Process incoming threat notification"""
        try:
            detection = threat_data.get('detection', {})
            threat_level = detection.get('threat_level', 'UNKNOWN')
            threat_type = detection.get('threat_type', 'unknown')
            confidence = detection.get('confidence', 0.0)
            
            self.threat_count += 1
            
            logger.info(f"Received {threat_level} threat: {threat_type} (confidence: {confidence:.2f})")
            
            # Call appropriate handler
            handler = self.threat_handlers.get(threat_level, self._handle_unknown_threat)
            handler(threat_data)
            
        except Exception as e:
            logger.error(f"Error processing threat data: {e}")
    
    def _handle_critical_threat(self, threat_data: Dict):
        """Handle CRITICAL threat level"""
        detection = threat_data['detection']
        
        logger.critical(f"🚨 CRITICAL THREAT DETECTED 🚨")
        logger.critical(f"Type: {detection['threat_type']}")
        logger.critical(f"Confidence: {detection['confidence']:.2f}")
        logger.critical(f"Distance: {detection.get('distance', 'Unknown')}m")
        logger.critical(f"Bearing: {detection.get('bearing', 'Unknown')}°")
        
        # Implement critical threat response
        # Examples:
        # - Activate emergency lighting
        # - Send emergency alerts
        # - Trigger alarm systems
        # - Log to emergency systems
        
        self._log_threat_to_file(threat_data, "CRITICAL")
        self._trigger_visual_alert("CRITICAL")
    
    def _handle_high_threat(self, threat_data: Dict):
        """Handle HIGH threat level"""
        detection = threat_data['detection']
        
        logger.warning(f"⚠️  HIGH THREAT DETECTED")
        logger.warning(f"Type: {detection['threat_type']}")
        logger.warning(f"Confidence: {detection['confidence']:.2f}")
        
        # Implement high threat response
        self._log_threat_to_file(threat_data, "HIGH")
        self._trigger_visual_alert("HIGH")
    
    def _handle_medium_threat(self, threat_data: Dict):
        """Handle MEDIUM threat level"""
        detection = threat_data['detection']
        
        logger.info(f"⚡ MEDIUM THREAT DETECTED")
        logger.info(f"Type: {detection['threat_type']}")
        logger.info(f"Confidence: {detection['confidence']:.2f}")
        
        # Implement medium threat response
        self._log_threat_to_file(threat_data, "MEDIUM")
    
    def _handle_low_threat(self, threat_data: Dict):
        """Handle LOW threat level"""
        detection = threat_data['detection']
        
        logger.info(f"ℹ️  LOW THREAT DETECTED")
        logger.info(f"Type: {detection['threat_type']}")
        logger.info(f"Confidence: {detection['confidence']:.2f}")
        
        # Implement low threat response
        self._log_threat_to_file(threat_data, "LOW")
    
    def _handle_unknown_threat(self, threat_data: Dict):
        """Handle unknown threat level"""
        logger.warning(f"❓ UNKNOWN THREAT LEVEL")
        logger.warning(f"Data: {threat_data}")
        
        self._log_threat_to_file(threat_data, "UNKNOWN")
    
    def _log_threat_to_file(self, threat_data: Dict, level: str):
        """Log threat to local file"""
        try:
            log_dir = "threat_logs"
            os.makedirs(log_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d")
            log_file = os.path.join(log_dir, f"threats_{timestamp}.json")
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "pi_id": self.pi_id,
                "threat_level": level,
                "threat_data": threat_data
            }
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to log threat to file: {e}")
    
    def _trigger_visual_alert(self, level: str):
        """Trigger visual alert (LED, display, etc.)"""
        try:
            # Example: Control GPIO pins for LED indicators
            # This would require RPi.GPIO library
            
            if level == "CRITICAL":
                # Flash red LED rapidly
                logger.info("🔴 Triggering CRITICAL visual alert")
                # GPIO implementation here
                
            elif level == "HIGH":
                # Solid red LED
                logger.info("🟠 Triggering HIGH visual alert")
                # GPIO implementation here
                
            # Add more visual alert implementations as needed
            
        except Exception as e:
            logger.error(f"Failed to trigger visual alert: {e}")
    
    def heartbeat_worker(self):
        """Background worker for sending heartbeats"""
        while self.running:
            try:
                self.send_heartbeat()
                time.sleep(30)  # Send heartbeat every 30 seconds
            except Exception as e:
                logger.error(f"Heartbeat worker error: {e}")
                time.sleep(5)
    
    def status_monitor(self):
        """Monitor and log system status"""
        while self.running:
            try:
                logger.info(f"Status: Running | Threats received: {self.threat_count} | Last heartbeat: {self.last_heartbeat}")
                time.sleep(300)  # Log status every 5 minutes
            except Exception as e:
                logger.error(f"Status monitor error: {e}")
                time.sleep(60)
    
    def run(self):
        """Run the Raspberry Pi client"""
        logger.info("Starting ARCIS Raspberry Pi Client...")
        
        # Register with API
        if not self.register_with_api():
            logger.error("Failed to register with API, continuing anyway...")
        
        self.running = True
        
        # Start background workers
        heartbeat_thread = threading.Thread(target=self.heartbeat_worker, daemon=True)
        status_thread = threading.Thread(target=self.status_monitor, daemon=True)
        threat_thread = threading.Thread(target=self.listen_for_threats, daemon=True)
        
        heartbeat_thread.start()
        status_thread.start()
        threat_thread.start()
        
        logger.info("All workers started. Client is operational.")
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown the client gracefully"""
        logger.info("Shutting down ARCIS Raspberry Pi Client...")
        self.running = False
        
        # Close Redis connection
        if self.redis_client:
            self.redis_client.close()
        
        logger.info("Client shutdown complete")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

def main():
    """Main entry point"""
    # Configuration from environment variables
    pi_id = os.getenv('ARCIS_PI_ID', f'pi_{int(time.time())}')
    api_url = os.getenv('ARCIS_API_URL', 'http://localhost:8080')
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_port = int(os.getenv('REDIS_PORT', 6379))
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run client
    client = ARCISRaspberryPiClient(
        pi_id=pi_id,
        arcis_api_url=api_url,
        redis_host=redis_host,
        redis_port=redis_port
    )
    
    try:
        client.run()
    except Exception as e:
        logger.error(f"Client failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 