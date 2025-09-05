# ARCIS System Deployment Guide

## Overview

The ARCIS (Advanced Reconnaissance and Combat Intelligence System) is a distributed weapon detection system designed for military field operations. This guide covers the complete deployment of the system with Redis integration, cloud services, and Raspberry Pi communication.

## System Architecture

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

## Components

1. **ARCIS Detection Service** (Jetson Nano) - Main object detection
2. **Redis** - Message broker and caching
3. **Cloud Service** - Google Cloud Vision integration
4. **API Service** - Raspberry Pi communication and monitoring
5. **Raspberry Pi Client** - Field alert system
6. **Website Integration** - Data upload and dashboard

## Prerequisites

### Hardware Requirements

- **Jetson Nano** (4GB recommended)
- **IMX415 Camera** with 2.8mm lens
- **Raspberry Pi** (3B+ or 4B)
- **Network connectivity** between all components

### Software Requirements

- Docker and Docker Compose
- NVIDIA Container Toolkit (for Jetson)
- Python 3.8+
- Redis 7+

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository>
cd arcis-system
cp env.example .env
# Edit .env with your configuration
```

### 2. Deploy with Docker Compose

```bash
# Start the complete system
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### 3. Verify Deployment

```bash
# Check Redis
curl http://localhost:8081  # Redis Commander (if enabled)

# Check API Service
curl http://localhost:8080/health

# Check system status
curl http://localhost:8080/api/system/status
```

## Detailed Configuration

### Environment Variables

Copy `env.example` to `.env` and configure:

```bash
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/gcp-key.json

# Website Integration
WEBSITE_API_URL=https://your-website.com
WEBSITE_API_KEY=your-api-key-here

# Database Configuration (optional)
DATABASE_URL=postgresql://user:password@host:port/database
```

### Google Cloud Setup

1. **Create GCP Project**
   ```bash
   gcloud projects create arcis-detection
   gcloud config set project arcis-detection
   ```

2. **Enable APIs**
   ```bash
   gcloud services enable vision.googleapis.com
   ```

3. **Create Service Account**
   ```bash
   gcloud iam service-accounts create arcis-service
   gcloud iam service-accounts keys create credentials/gcp-key.json \
     --iam-account=arcis-service@arcis-detection.iam.gserviceaccount.com
   ```

4. **Grant Permissions**
   ```bash
   gcloud projects add-iam-policy-binding arcis-detection \
     --member="serviceAccount:arcis-service@arcis-detection.iam.gserviceaccount.com" \
     --role="roles/vision.admin"
   ```

### Redis Configuration

Redis is automatically configured via Docker Compose with:
- **Memory limit**: 512MB
- **Persistence**: Enabled with AOF
- **TTL policies**: Automatic cleanup
- **Pub/Sub**: Real-time messaging

### Jetson Nano Setup

1. **Install Docker**
   ```bash
   # Install Docker for Jetson
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   ```

2. **Install NVIDIA Container Toolkit**
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

3. **Deploy ARCIS**
   ```bash
   # Copy files to Jetson
   scp -r arcis-system/ jetson@jetson-ip:~/
   
   # SSH to Jetson and deploy
   ssh jetson@jetson-ip
   cd arcis-system
   docker-compose up -d arcis_detection redis
   ```

## Service Configuration

### 1. Detection Service (Jetson Nano)

**Features:**
- Real-time object detection
- Redis integration for threat publishing
- IMX415 distance estimation
- Military threat classification
- Audio alerts for critical threats

**Configuration:**
```yaml
# docker-compose.yml
arcis_detection:
  build:
    dockerfile: Dockerfile.jetson
  privileged: true  # For camera access
  volumes:
    - /dev:/dev  # Device access
    - ./models:/app/models
  environment:
    - REDIS_HOST=redis
    - GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT}
```

### 2. Cloud Service

**Features:**
- Google Cloud Vision analysis for YELLOW threats
- Automatic threat escalation
- Website data upload
- Retry logic for failed uploads

**Queues Processed:**
- `arcis:yellow_threats` → Cloud Vision analysis
- `arcis:orange_red_threats` → Immediate website upload
- `arcis:website_queue` → Standard website upload

### 3. API Service

**Features:**
- Raspberry Pi registration and heartbeat
- Real-time WebSocket updates
- Mission data management
- System monitoring endpoints

**Endpoints:**
- `GET /health` - Health check
- `POST /api/raspberry-pi/register` - Pi registration
- `GET /api/threats/active` - Active threats
- `GET /api/system/status` - System status
- `WebSocket /ws/threats` - Real-time updates

### 4. Raspberry Pi Client

**Features:**
- Real-time threat notifications
- Threat level-based responses
- Local logging
- Visual/audio alerts
- Heartbeat monitoring

**Setup:**
```bash
# On Raspberry Pi
pip install -r requirements_pi.txt
export ARCIS_PI_ID="pi_field_01"
export ARCIS_API_URL="http://jetson-ip:8080"
export REDIS_HOST="jetson-ip"
python raspberry_pi_client.py
```

## Data Flow

### 1. Threat Detection Flow

```
Camera → Jetson Detection → Redis Queues → Cloud/API Services → Website/Pi
```

1. **Camera captures frame**
2. **Jetson runs YOLO detection**
3. **Threat classified by level**
4. **Frame cached in Redis (CRITICAL/HIGH only)**
5. **Threat published to appropriate queues**:
   - YELLOW → Cloud Vision analysis
   - ORANGE/RED → Immediate website upload
   - ALL → Raspberry Pi notifications

### 2. Cloud Vision Flow

```
YELLOW Threat → Cloud Vision API → Escalation Decision → Website Upload
```

1. **YELLOW threat queued**
2. **Frame retrieved from cache**
3. **Google Cloud Vision analysis**
4. **Escalation decision based on results**
5. **Enhanced data sent to website**

### 3. Raspberry Pi Flow

```
Redis Pub/Sub → Pi Client → Threat Handler → Local Response
```

1. **Threat published to Pi channel**
2. **Pi client receives notification**
3. **Threat level determines response**:
   - CRITICAL: Emergency alerts, logging
   - HIGH: Warning alerts, logging
   - MEDIUM: Standard logging
   - LOW: Basic logging

## Monitoring and Maintenance

### System Health Checks

```bash
# Check all services
docker-compose ps

# Check Redis statistics
curl http://localhost:8080/api/threats/statistics

# Check system status
curl http://localhost:8080/api/system/status

# View real-time logs
docker-compose logs -f arcis_detection
docker-compose logs -f arcis_cloud
```

### Redis Monitoring

```bash
# Connect to Redis CLI
docker-compose exec redis redis-cli

# Check queue sizes
LLEN arcis:yellow_threats
LLEN arcis:orange_red_threats
LLEN arcis:website_queue

# Check cached frames
KEYS arcis:critical_frames:*

# Monitor pub/sub
MONITOR
```

### Performance Tuning

**Jetson Nano Optimization:**
```bash
# Set performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Monitor GPU usage
tegrastats

# Check memory usage
free -h
```

**Redis Optimization:**
```bash
# Increase memory if needed
# Edit docker-compose.yml
command: redis-server --maxmemory 1gb --maxmemory-policy allkeys-lru
```

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   ```bash
   # Check Redis container
   docker-compose logs redis
   
   # Test connection
   docker-compose exec redis redis-cli ping
   ```

2. **Google Cloud Vision Errors**
   ```bash
   # Check credentials
   ls -la credentials/gcp-key.json
   
   # Test authentication
   gcloud auth application-default print-access-token
   ```

3. **Camera Not Detected**
   ```bash
   # Check camera connection
   ls /dev/video*
   
   # Test camera
   v4l2-ctl --list-devices
   ```

4. **High Memory Usage**
   ```bash
   # Check container memory
   docker stats
   
   # Reduce batch size in detection
   # Edit train_weapon_detection_redis.py
   ```

### Log Analysis

```bash
# Detection service logs
docker-compose logs arcis_detection | grep "CRITICAL\|HIGH"

# Cloud service logs
docker-compose logs arcis_cloud | grep "escalated\|failed"

# API service logs
docker-compose logs arcis_api | grep "ERROR\|WARNING"

# Redis logs
docker-compose logs redis
```

## Security Considerations

### Network Security

1. **Firewall Configuration**
   ```bash
   # Allow only necessary ports
   sudo ufw allow 6379/tcp  # Redis (internal only)
   sudo ufw allow 8080/tcp  # API service
   ```

2. **Redis Security**
   ```bash
   # Add authentication (production)
   command: redis-server --requirepass your-password
   ```

3. **API Security**
   - Use HTTPS in production
   - Implement API key authentication
   - Rate limiting for endpoints

### Data Security

1. **Encrypt sensitive data**
2. **Secure credential storage**
3. **Regular security updates**
4. **Access logging and monitoring**

## Production Deployment

### High Availability Setup

1. **Redis Cluster**
   ```yaml
   # Use Redis Sentinel or Cluster mode
   redis-sentinel:
     image: redis:7-alpine
     command: redis-sentinel /etc/redis/sentinel.conf
   ```

2. **Load Balancing**
   ```yaml
   # Add nginx load balancer
   nginx:
     image: nginx:alpine
     ports:
       - "80:80"
       - "443:443"
   ```

3. **Health Monitoring**
   ```yaml
   # Add monitoring stack
   prometheus:
     image: prom/prometheus
   grafana:
     image: grafana/grafana
   ```

### Backup Strategy

1. **Redis Data Backup**
   ```bash
   # Automated backup script
   docker-compose exec redis redis-cli BGSAVE
   ```

2. **Mission Logs Backup**
   ```bash
   # Sync mission logs to cloud storage
   rsync -av mission_logs/ backup-location/
   ```

## Support and Maintenance

### Regular Maintenance Tasks

1. **Weekly:**
   - Check system health
   - Review error logs
   - Update threat statistics

2. **Monthly:**
   - Update Docker images
   - Clean old mission logs
   - Performance review

3. **Quarterly:**
   - Security audit
   - System optimization
   - Hardware maintenance

### Getting Help

- Check logs first: `docker-compose logs`
- Review this documentation
- Check Redis statistics
- Monitor system resources

For additional support, contact the ARCIS development team. 