# L76K GPS Module Setup Guide for ARCIS System

## Overview
This guide covers the integration of the L76K GPS module with Jetson Nano for the ARCIS tactical weapon detection system. The L76K provides precise GPS positioning for military field operations.

## Hardware Requirements

### L76K GPS Module Specifications
- **Chipset**: Quectel L76K
- **Channels**: 33 tracking / 99 acquisition
- **Sensitivity**: -165 dBm tracking, -148 dBm acquisition
- **Accuracy**: <2.5m CEP (without SA)
- **Update Rate**: 1Hz (default), up to 10Hz
- **Power**: 3.3V, ~20mA tracking
- **Interface**: UART (9600 baud default)
- **Protocols**: NMEA 0183, PMTK

### Jetson Nano Connections
```
L76K Module    →    Jetson Nano
VCC (3.3V)     →    Pin 1 (3.3V)
GND            →    Pin 6 (GND)
TX             →    Pin 10 (GPIO 15, UART RX)
RX             →    Pin 8 (GPIO 14, UART TX)
```

## Software Installation

### 1. Install Required Dependencies
```bash
# Install GPS-specific packages
pip install -r requirements_gps.txt

# Or install individually:
pip install pyserial>=3.5
pip install pynmea2>=1.19.0
pip install mgrs>=1.4.0
```

### 2. Enable UART on Jetson Nano
```bash
# Add to /boot/config.txt (if using Jetson Nano with SD card)
sudo nano /boot/config.txt

# Add these lines:
enable_uart=1
dtoverlay=uart2

# For Jetson Nano Developer Kit, UART is usually available at:
# /dev/ttyTHS1 (hardware UART)
# /dev/ttyUSB0 (if using USB-to-Serial adapter)
```

### 3. Set Permissions
```bash
# Add user to dialout group for serial access
sudo usermod -a -G dialout $USER

# Set permissions for UART device
sudo chmod 666 /dev/ttyTHS1
# or
sudo chmod 666 /dev/ttyUSB0
```

## Hardware Setup

### Option 1: Direct UART Connection
1. Connect L76K directly to Jetson Nano UART pins
2. Use `/dev/ttyTHS1` as the port in software

### Option 2: USB-to-Serial Adapter (Recommended for Testing)
1. Connect L76K to USB-to-Serial adapter (FTDI, CP2102, etc.)
2. Connect adapter to Jetson Nano USB port
3. Use `/dev/ttyUSB0` as the port in software

### Wiring Diagram
```
L76K GPS Module
┌─────────────────┐
│  VCC  GND  TX  RX │
└──┬────┬────┬───┬──┘
   │    │    │   │
   │    │    │   └── Pin 8 (GPIO 14, UART TX)
   │    │    └────── Pin 10 (GPIO 15, UART RX)
   │    └─────────── Pin 6 (GND)
   └──────────────── Pin 1 (3.3V)

Jetson Nano GPIO Header
```

## Software Configuration

### 1. Test GPS Connection
```bash
# Run the GPS test function
python train_weapon_detection_gps.py

# Select option 4: "Test L76K GPS connection only"
# Enter the correct port (e.g., /dev/ttyUSB0 or /dev/ttyTHS1)
```

### 2. Configure GPS Port
In `train_weapon_detection_gps.py`, the default port is `/dev/ttyUSB0`. 
Modify if using different connection:

```python
# For direct UART connection:
gps_port = '/dev/ttyTHS1'

# For USB adapter:
gps_port = '/dev/ttyUSB0'

# For specific USB device:
gps_port = '/dev/ttyACM0'
```

## GPS Features in ARCIS System

### Real-time GPS Data
- **Coordinates**: Latitude/Longitude with 6 decimal precision
- **MGRS**: Military Grid Reference System coordinates
- **Altitude**: Height above sea level
- **Satellites**: Number of satellites in view
- **HDOP**: Horizontal Dilution of Precision
- **Speed**: Ground speed in knots
- **Course**: True course over ground

### Tactical Integration
- **Threat Logging**: GPS coordinates saved with each detection
- **Mission Tracking**: Start/end positions logged
- **SITREP Generation**: Position data in situation reports
- **Screenshot Geotagging**: GPS coordinates in filenames
- **Bearing Calculation**: Uses GPS course for accurate threat bearing

### Controls During Operation
- **'g'**: Show detailed GPS status
- **'m'**: Display MGRS coordinates
- **'r'**: Generate SITREP with GPS data
- **'s'**: Save screenshot with GPS coordinates

## Troubleshooting

### Common Issues

#### 1. Permission Denied
```bash
# Error: Permission denied: '/dev/ttyUSB0'
sudo chmod 666 /dev/ttyUSB0
sudo usermod -a -G dialout $USER
# Logout and login again
```

#### 2. No GPS Lock
- **Check antenna**: Ensure GPS antenna is connected and has clear sky view
- **Wait time**: GPS cold start can take 30+ seconds
- **Location**: Move to open area away from buildings
- **Check wiring**: Verify TX/RX connections are correct

#### 3. Device Not Found
```bash
# List available serial devices
ls -la /dev/tty*

# Check USB devices
lsusb

# Check dmesg for connection messages
dmesg | grep tty
```

#### 4. NMEA Parsing Errors
- **Baud rate**: Ensure 9600 baud (L76K default)
- **Data format**: Check NMEA sentence format
- **Module health**: Verify L76K is receiving power

### GPS Status Indicators

#### Good GPS Status
```
GPS: 8SAT HDOP:1.2
LAT: 40.123456 LON: -74.123456
MGRS: 18TWL1234567890
```

#### Poor GPS Status
```
GPS: NO LOCK (3SAT)
LAT: 0.000000 LON: 0.000000
MGRS: N/A
```

## Performance Optimization

### For Jetson Nano
- **Update Rate**: Keep at 1Hz for power efficiency
- **Satellite Systems**: Enable GPS + GLONASS for better coverage
- **Power Management**: Use sleep mode when not in use

### NMEA Configuration
```python
# Optional: Configure L76K for better performance
# Send PMTK commands for:
# - Update rate: PMTK_SET_NMEA_UPDATE_1HZ
# - Output format: PMTK_SET_NMEA_OUTPUT_RMCGGA
# - Satellite systems: PMTK_API_SET_GPS_GLONASS
```

## Field Deployment Tips

### 1. Antenna Placement
- Mount GPS antenna with clear sky view
- Avoid metal obstructions
- Consider external antenna for vehicle mounting

### 2. Power Management
- L76K draws ~20mA during tracking
- Consider backup power for extended operations
- Use sleep mode during system standby

### 3. Accuracy Considerations
- HDOP < 2.0 for good accuracy
- Wait for 4+ satellites before mission start
- Consider DGPS/RTK for sub-meter accuracy

### 4. Military Grid Reference System (MGRS)
- Provides standardized military coordinates
- Format: Zone + Grid Square + Coordinates
- Example: 18TWL1234567890

## Integration with ARCIS Features

### Mission Logging
```json
{
  "timestamp": "2024-01-15T14:30:45.123Z",
  "threat_type": "weapon",
  "confidence": 0.85,
  "threat_level": "HIGH",
  "distance": 25.3,
  "bearing": 045.2,
  "gps_coordinates": {
    "latitude": 40.123456,
    "longitude": -74.123456,
    "altitude": 125.5,
    "valid": true,
    "satellites": 8,
    "hdop": 1.2
  },
  "mgrs_coordinates": "18TWL1234567890"
}
```

### Tactical Display
- Real-time GPS status in top-left corner
- Coordinates displayed when GPS locked
- Color-coded GPS status (green=locked, red=no lock)
- MGRS coordinates available on demand

## Security Considerations

### 1. OPSEC (Operational Security)
- GPS data contains sensitive location information
- Secure mission logs and screenshots
- Consider GPS jamming/spoofing in hostile environments

### 2. Data Encryption
- Encrypt mission logs containing GPS data
- Use secure channels for SITREP transmission
- Implement data sanitization procedures

### 3. Backup Navigation
- Don't rely solely on GPS
- Maintain traditional navigation skills
- Have backup positioning methods

## Maintenance

### Regular Checks
- Verify GPS antenna connections
- Check for firmware updates
- Monitor GPS accuracy and satellite count
- Clean antenna contacts

### Calibration
- Perform static position tests
- Verify MGRS coordinate accuracy
- Check bearing calculations against known references

## Support and Resources

### Documentation
- L76K Hardware Design Manual
- NMEA 0183 Protocol Reference
- MGRS Coordinate System Guide

### Tools
- GPS test utilities
- NMEA sentence analyzers
- MGRS coordinate converters

This setup guide ensures proper integration of the L76K GPS module with your ARCIS tactical system for enhanced military field operations. 