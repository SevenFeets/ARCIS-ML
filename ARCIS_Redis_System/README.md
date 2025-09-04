# ARCIS Redis Integration System

This folder contains the distributed, Redis-integrated version of the ARCIS (Advanced Reconnaissance and Combat Intelligence System) weapon detection system. This enhanced version adds cloud processing, real-time messaging, and multi-device communication capabilities.

## 🚀 Key Features

- **Redis Message Broker**: Real-time threat distribution and caching
- **Google Cloud Vision Integration**: Enhanced threat analysis for MEDIUM threats
- **Distributed Architecture**: Multi-container deployment with Docker Compose
- **Raspberry Pi Communication**: Field alert system with real-time notifications
- **Website Integration**: Automatic data upload and dashboard connectivity
- **10-minute Critical Frame Caching**: Automatic cleanup with TTL
- **Threat Level Routing**: Intelligent queue management based on threat severity

## 📁 File Structure

### Core Services
- `train_weapon_detection_redis.py` - Main detection service with Redis integration
- `arcis_redis_integration.py` - Redis manager and integration layer
- `arcis_cloud_service.py` - Google Cloud Vision processing service
- `arcis_api_service.py` - FastAPI service for Raspberry Pi communication
- `raspberry_pi_client.py` - Field alert client for Raspberry Pi devices

### Docker Configuration
- `docker-compose.yml` - Complete multi-container deployment
- `Dockerfile.jetson` - Jetson Nano detection service container
- `Dockerfile.cloud` - Cloud processing service container
- `Dockerfile.api` - API service container

### Requirements
- `requirements_jetson.txt` - Dependencies for Jetson Nano
- `requirements_cloud.txt` - Dependencies for cloud service
- `requirements_api.txt` - Dependencies for API service

### Configuration
- `env.example` - Environment variables template
- `DEPLOYMENT_GUIDE.md` - Complete deployment instructions

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Jetson Nano   │    │      Redis      │    │  Cloud Service  │
│  (Detection)    │◄──►│   (Message      │◄──►│ (Google Vision) │
│                 │    │    Broker)      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Service   │    │  Raspberry Pi   │    │    Website      │
│  (Monitoring)   │◄──►│   (Alerts)      │    │  (Dashboard)    │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔄 Data Flow

### Threat Detection Flow
1. **Camera** captures frame on Jetson Nano
2. **YOLO Detection** processes frame and classifies threats
3. **Redis Queues** route threats based on severity:
   - **CRITICAL/HIGH**: Immediate website upload + frame caching
   - **MEDIUM**: Google Cloud Vision analysis queue
   - **ALL**: Raspberry Pi notification queue

### Cloud Processing Flow
1. **MEDIUM threats** queued for Cloud Vision analysis
2. **Google Cloud Vision** analyzes cached frames
3. **Escalation logic** determines if threat should be upgraded
4. **Enhanced data** sent to website with Cloud Vision results

### Field Alert Flow
1. **All threats** published to Raspberry Pi channel
2. **Pi clients** receive real-time notifications
3. **Threat handlers** trigger appropriate responses:
   - CRITICAL: Emergency alerts, logging
   - HIGH: Warning alerts, logging
   - MEDIUM/LOW: Standard logging

## 🚀 Quick Start

### 1. Setup Environment
```bash
cd ARCIS_Redis_System
cp env.example .env
# Edit .env with your configuration
```

### 2. Deploy with Docker Compose
```bash
# Start complete system
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### 3. Deploy Raspberry Pi Client
```bash
# On Raspberry Pi
export ARCIS_PI_ID="pi_field_01"
export ARCIS_API_URL="http://jetson-ip:8080"
export REDIS_HOST="jetson-ip"
python raspberry_pi_client.py
```

## 🔧 Configuration

### Environment Variables
- `GOOGLE_CLOUD_PROJECT` - GCP project for Cloud Vision
- `WEBSITE_API_URL` - Your website API endpoint
- `WEBSITE_API_KEY` - API authentication key
- `REDIS_HOST/PORT` - Redis connection details

### Threat Level Routing
- **CRITICAL/HIGH** → Immediate website upload + Pi alerts
- **MEDIUM** → Cloud Vision analysis → Website upload + Pi alerts  
- **LOW** → Pi alerts only

### Caching Strategy
- **Critical frames**: 10-minute TTL with automatic cleanup
- **Threat data**: 1-hour TTL for Pi consumption
- **Mission data**: 24-hour TTL for reporting

## 📊 Monitoring

### Health Checks
```bash
# API Service health
curl http://localhost:8080/health

# System statistics
curl http://localhost:8080/api/threats/statistics

# Redis statistics
curl http://localhost:8080/api/system/status
```

### Real-time Monitoring
- **Redis Commander**: http://localhost:8081 (if enabled)
- **WebSocket**: ws://localhost:8080/ws/threats
- **API Endpoints**: http://localhost:8080/api/

## 🔒 Security Features

- **Redis TTL**: Automatic data expiration
- **API Authentication**: Bearer token support
- **Encrypted Communication**: HTTPS/WSS in production
- **Access Logging**: Comprehensive audit trails
- **Credential Management**: Secure environment variables

## 🆚 Differences from Original ARCIS

### Original ARCIS (`train_weapon_detection.py`)
- ✅ Standalone operation
- ✅ Local processing only
- ✅ Direct camera interface
- ✅ Mission logging
- ✅ Tactical interface

### Redis-Integrated ARCIS
- ✅ **All original features PLUS:**
- 🆕 Distributed processing
- 🆕 Cloud intelligence integration
- 🆕 Real-time multi-device communication
- 🆕 Automatic website data upload
- 🆕 Raspberry Pi field alerts
- 🆕 Redis caching and messaging
- 🆕 Docker containerization
- 🆕 Scalable architecture

## 📚 Documentation

- `DEPLOYMENT_GUIDE.md` - Complete deployment instructions
- `env.example` - Configuration template
- Docker Compose comments - Service-specific configuration

## 🔧 Troubleshooting

### Common Issues
1. **Redis Connection Failed**: Check Redis container status
2. **Google Cloud Vision Errors**: Verify credentials and API access
3. **Camera Not Detected**: Ensure proper device permissions
4. **High Memory Usage**: Adjust batch sizes and image resolution

### Debug Commands
```bash
# Check container logs
docker-compose logs [service_name]

# Test Redis connection
docker-compose exec redis redis-cli ping

# Monitor system resources
docker stats
```

## 🤝 Support

For deployment assistance or troubleshooting:
1. Check the `DEPLOYMENT_GUIDE.md`
2. Review container logs
3. Verify environment configuration
4. Test individual components

---

**Note**: This Redis-integrated system is designed for distributed military field operations. The original standalone ARCIS system (`train_weapon_detection.py`) remains available in the parent directory for single-device deployments. 