#!/usr/bin/env python3
"""
ARCIS Cloud Integration Service
Handles Google Cloud Vision processing and website communication
"""

import os
import time
import json
import logging
import asyncio
from datetime import datetime
from arcis_redis_integration import ARCISRedisManager
import requests
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ARCISCloudService:
    """Cloud integration service for ARCIS system"""
    
    def __init__(self):
        # Initialize Redis connection
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        google_project = os.getenv('GOOGLE_CLOUD_PROJECT')
        
        self.redis_manager = ARCISRedisManager(
            redis_host=redis_host,
            redis_port=redis_port,
            google_cloud_project=google_project
        )
        
        # Website configuration
        self.website_url = os.getenv('WEBSITE_API_URL')
        self.website_api_key = os.getenv('WEBSITE_API_KEY')
        
        # Service statistics
        self.stats = {
            'threats_processed': 0,
            'cloud_vision_calls': 0,
            'website_uploads': 0,
            'errors': 0,
            'start_time': datetime.now().isoformat()
        }
        
        logger.info("ARCIS Cloud Service initialized")
        logger.info(f"Redis: {redis_host}:{redis_port}")
        logger.info(f"Google Cloud Project: {google_project}")
        logger.info(f"Website URL: {self.website_url}")
    
    async def process_yellow_threats(self):
        """Process YELLOW threats with Google Cloud Vision"""
        logger.info("Starting Yellow Threats processor...")
        
        while True:
            try:
                # Check for yellow threats
                result = self.redis_manager.redis_client.brpop(
                    self.redis_manager.YELLOW_THREATS_KEY, 
                    timeout=5
                )
                
                if result:
                    _, message = result
                    threat_data = json.loads(message)
                    
                    logger.info(f"Processing YELLOW threat: {threat_data['detection']['threat_type']}")
                    
                    # Get cached frame if available
                    frame_id = threat_data.get('frame_id')
                    if frame_id and self.redis_manager.vision_client:
                        frame_data = self.redis_manager.get_cached_frame(frame_id)
                        if frame_data:
                            frame, _ = frame_data
                            
                            # Analyze with Cloud Vision
                            vision_result = self.redis_manager._analyze_with_cloud_vision(frame)
                            self.stats['cloud_vision_calls'] += 1
                            
                            # Update threat data
                            enhanced_threat = threat_data.copy()
                            enhanced_threat['cloud_vision_analysis'] = vision_result
                            enhanced_threat['processed_at'] = datetime.now().isoformat()
                            
                            # Check if threat should be escalated
                            if self.redis_manager._should_escalate_threat(vision_result):
                                enhanced_threat['detection']['threat_level'] = 'HIGH'
                                enhanced_threat['escalated'] = True
                                logger.warning("Escalated YELLOW threat to HIGH based on Cloud Vision")
                            else:
                                enhanced_threat['confirmed_medium'] = True
                                logger.info("Confirmed YELLOW threat as MEDIUM")
                            
                            # Send to website queue
                            self.redis_manager.redis_client.lpush(
                                self.redis_manager.WEBSITE_QUEUE, 
                                json.dumps(enhanced_threat)
                            )
                    
                    self.stats['threats_processed'] += 1
                
                await asyncio.sleep(0.1)  # Small delay to prevent CPU spinning
                
            except Exception as e:
                logger.error(f"Error processing yellow threats: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(1)
    
    async def process_website_uploads(self):
        """Process website uploads for all threat levels"""
        logger.info("Starting Website Upload processor...")
        
        while True:
            try:
                # Process both orange/red threats and general website queue
                result = self.redis_manager.redis_client.brpop([
                    self.redis_manager.ORANGE_RED_THREATS_KEY,
                    self.redis_manager.WEBSITE_QUEUE
                ], timeout=5)
                
                if result:
                    queue_name, message = result
                    threat_data = json.loads(message)
                    
                    # Determine priority based on queue
                    is_critical = queue_name.decode() == self.redis_manager.ORANGE_RED_THREATS_KEY
                    
                    logger.info(f"Uploading {'CRITICAL' if is_critical else 'STANDARD'} threat to website")
                    
                    # Send to website
                    success = await self._send_to_website(threat_data, is_critical)
                    
                    if success:
                        self.stats['website_uploads'] += 1
                        logger.info("Successfully uploaded threat data to website")
                    else:
                        # Retry logic for failed uploads
                        retry_count = threat_data.get('retry_count', 0)
                        if retry_count < 3:
                            threat_data['retry_count'] = retry_count + 1
                            # Put back in queue for retry
                            queue_key = (self.redis_manager.ORANGE_RED_THREATS_KEY 
                                       if is_critical 
                                       else self.redis_manager.WEBSITE_QUEUE)
                            self.redis_manager.redis_client.lpush(queue_key, json.dumps(threat_data))
                            logger.warning(f"Retrying upload (attempt {retry_count + 1})")
                        else:
                            logger.error("Max retries exceeded for website upload")
                            self.stats['errors'] += 1
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing website uploads: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(1)
    
    async def _send_to_website(self, threat_data: Dict[Any, Any], is_critical: bool = False) -> bool:
        """Send threat data to website API"""
        try:
            if not self.website_url:
                logger.warning("Website URL not configured")
                return False
            
            # Prepare headers
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'ARCIS-Cloud-Service/1.0'
            }
            
            if self.website_api_key:
                headers['Authorization'] = f'Bearer {self.website_api_key}'
            
            # Add priority flag for critical threats
            if is_critical:
                headers['X-ARCIS-Priority'] = 'CRITICAL'
            
            # Prepare payload
            payload = {
                'threat_data': threat_data,
                'service_info': {
                    'service': 'arcis_cloud',
                    'version': '1.0',
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            # Send request
            response = requests.post(
                f"{self.website_url}/api/threats",
                json=payload,
                headers=headers,
                timeout=30 if is_critical else 10
            )
            
            if response.status_code in [200, 201, 202]:
                return True
            else:
                logger.error(f"Website API error: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            logger.error("Website upload timeout")
            return False
        except requests.exceptions.ConnectionError:
            logger.error("Website connection error")
            return False
        except Exception as e:
            logger.error(f"Website upload error: {e}")
            return False
    
    async def monitor_system(self):
        """Monitor system health and log statistics"""
        logger.info("Starting System Monitor...")
        
        while True:
            try:
                # Log statistics every 60 seconds
                await asyncio.sleep(60)
                
                # Get Redis statistics
                redis_stats = self.redis_manager.get_threat_statistics()
                
                # Log combined statistics
                logger.info("=== ARCIS CLOUD SERVICE STATISTICS ===")
                logger.info(f"Threats Processed: {self.stats['threats_processed']}")
                logger.info(f"Cloud Vision Calls: {self.stats['cloud_vision_calls']}")
                logger.info(f"Website Uploads: {self.stats['website_uploads']}")
                logger.info(f"Errors: {self.stats['errors']}")
                logger.info("=== REDIS QUEUE STATUS ===")
                for key, value in redis_stats.items():
                    logger.info(f"{key}: {value}")
                
                # Store statistics in Redis for monitoring
                stats_data = {
                    'cloud_service_stats': self.stats,
                    'redis_stats': redis_stats,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.redis_manager.redis_client.setex(
                    'arcis:cloud_service_stats',
                    300,  # 5 minutes TTL
                    json.dumps(stats_data)
                )
                
            except Exception as e:
                logger.error(f"Error in system monitor: {e}")
                self.stats['errors'] += 1
    
    async def run(self):
        """Run all service components"""
        logger.info("Starting ARCIS Cloud Service...")
        
        # Create tasks for all components
        tasks = [
            asyncio.create_task(self.process_yellow_threats()),
            asyncio.create_task(self.process_website_uploads()),
            asyncio.create_task(self.monitor_system())
        ]
        
        try:
            # Run all tasks concurrently
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Shutting down ARCIS Cloud Service...")
            for task in tasks:
                task.cancel()
        except Exception as e:
            logger.error(f"Service error: {e}")
            raise

def main():
    """Main entry point"""
    service = ARCISCloudService()
    
    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service failed: {e}")
        exit(1)

if __name__ == "__main__":
    main() 