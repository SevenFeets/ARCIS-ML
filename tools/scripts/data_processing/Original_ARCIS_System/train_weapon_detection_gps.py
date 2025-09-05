from ultralytics import YOLO
import torch
import os
import cv2
import matplotlib.pyplot as plt
import random
from pathlib import Path
import numpy as np
import pygame
import threading
import time
import yaml
import json
from datetime import datetime
import math
import serial
import pynmea2

# Jetson Nano deployment configurations
JETSON_CONFIGS = {
    "2K_square": {
        "display_size": (1440, 1440),
        "inference_size": 416,  # Smaller for Jetson Nano performance
        "batch_size": 1,
        "font_scale": 0.4,
        "line_thickness": 1,
        "description": "2.9 inch 2K 1440x1440 IPS display"
    },
    "1080p": {
        "display_size": (1920, 1080),
        "inference_size": 640,
        "batch_size": 1,
        "font_scale": 0.6,
        "line_thickness": 2,
        "description": "1080p laptop screen"
    }
}

# Military threat classification
THREAT_LEVELS = {
    "CRITICAL": {"color": (0, 0, 255), "priority": 1, "code": "RED"},      # Immediate danger
    "HIGH": {"color": (0, 100, 255), "priority": 2, "code": "ORANGE"},    # Significant threat
    "MEDIUM": {"color": (0, 255, 255), "priority": 3, "code": "YELLOW"},  # Potential threat
    "LOW": {"color": (0, 255, 0), "priority": 4, "code": "GREEN"},        # Minimal threat
    "UNKNOWN": {"color": (128, 128, 128), "priority": 5, "code": "GRAY"}  # Unclassified
}

# Engagement recommendations
ENGAGEMENT_RECOMMENDATIONS = {
    "weapon": {"action": "ENGAGE", "method": "SMALL_ARMS", "priority": "IMMEDIATE"},
    "gun": {"action": "ENGAGE", "method": "SMALL_ARMS", "priority": "IMMEDIATE"},
    "military_tank": {"action": "AVOID/CALL_SUPPORT", "method": "ANTI_ARMOR", "priority": "CRITICAL"},
    "military_aircraft": {"action": "TAKE_COVER", "method": "AIR_DEFENSE", "priority": "CRITICAL"},
    "high_warning": {"action": "ASSESS", "method": "OBSERVE", "priority": "HIGH"}
}

class L76K_GPS:
    """L76K GPS module interface for Jetson Nano"""
    
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600, timeout=1):
        """
        Initialize L76K GPS module
        
        Args:
            port: Serial port (usually /dev/ttyUSB0 or /dev/ttyACM0 on Jetson)
            baudrate: Communication speed (9600 for L76K)
            timeout: Read timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None
        self.last_valid_position = None
        self.gps_lock = False
        self.satellites = 0
        self.hdop = 99.9  # Horizontal dilution of precision
        
        # GPS data
        self.latitude = 0.0
        self.longitude = 0.0
        self.altitude = 0.0
        self.speed = 0.0
        self.course = 0.0
        self.timestamp = None
        
        self.connect()
    
    def connect(self):
        """Connect to L76K GPS module"""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
            print(f"L76K GPS connected on {self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to L76K GPS: {e}")
            print("GPS features will be disabled")
            return False
    
    def read_gps_data(self):
        """Read and parse GPS data from L76K module"""
        if not self.serial_conn or not self.serial_conn.is_open:
            return False
        
        try:
            # Read line from GPS
            line = self.serial_conn.readline().decode('ascii', errors='replace').strip()
            
            if line.startswith('$'):
                try:
                    msg = pynmea2.parse(line)
                    
                    # Process different NMEA message types
                    if isinstance(msg, pynmea2.GGA):  # Global Positioning System Fix Data
                        if msg.latitude and msg.longitude:
                            self.latitude = float(msg.latitude)
                            self.longitude = float(msg.longitude)
                            self.altitude = float(msg.altitude) if msg.altitude else 0.0
                            self.satellites = int(msg.num_sats) if msg.num_sats else 0
                            self.hdop = float(msg.horizontal_dil) if msg.horizontal_dil else 99.9
                            self.gps_lock = msg.gps_qual > 0
                            self.last_valid_position = datetime.now()
                            return True
                    
                    elif isinstance(msg, pynmea2.RMC):  # Recommended Minimum Course
                        if msg.latitude and msg.longitude:
                            self.latitude = float(msg.latitude)
                            self.longitude = float(msg.longitude)
                            self.speed = float(msg.spd_over_grnd) if msg.spd_over_grnd else 0.0
                            self.course = float(msg.true_course) if msg.true_course else 0.0
                            self.timestamp = msg.timestamp
                            return True
                    
                except pynmea2.ParseError:
                    pass  # Ignore malformed sentences
                    
        except Exception as e:
            print(f"GPS read error: {e}")
        
        return False
    
    def get_position(self):
        """Get current GPS position"""
        self.read_gps_data()
        
        if self.gps_lock and self.latitude != 0.0 and self.longitude != 0.0:
            return {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "altitude": self.altitude,
                "timestamp": datetime.now().isoformat(),
                "satellites": self.satellites,
                "hdop": self.hdop,
                "valid": True
            }
        else:
            return {
                "latitude": 0.0,
                "longitude": 0.0,
                "altitude": 0.0,
                "timestamp": datetime.now().isoformat(),
                "satellites": self.satellites,
                "hdop": self.hdop,
                "valid": False
            }
    
    def get_mgrs_coordinates(self):
        """Convert GPS coordinates to Military Grid Reference System (MGRS)"""
        try:
            import mgrs
            m = mgrs.MGRS()
            if self.gps_lock and self.latitude != 0.0 and self.longitude != 0.0:
                mgrs_coord = m.toMGRS(self.latitude, self.longitude)
                return mgrs_coord
        except ImportError:
            print("MGRS library not installed. Install with: pip install mgrs")
        except Exception as e:
            print(f"MGRS conversion error: {e}")
        
        return "N/A"
    
    def calculate_bearing_to_point(self, target_lat, target_lon):
        """Calculate bearing from current position to target coordinates"""
        if not self.gps_lock:
            return 0.0
        
        lat1 = math.radians(self.latitude)
        lat2 = math.radians(target_lat)
        diff_lon = math.radians(target_lon - self.longitude)
        
        x = math.sin(diff_lon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diff_lon))
        
        bearing = math.atan2(x, y)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360
        
        return bearing
    
    def calculate_distance_to_point(self, target_lat, target_lon):
        """Calculate distance from current position to target coordinates (in meters)"""
        if not self.gps_lock:
            return 0.0
        
        # Haversine formula
        R = 6371000  # Earth's radius in meters
        
        lat1 = math.radians(self.latitude)
        lat2 = math.radians(target_lat)
        diff_lat = math.radians(target_lat - self.latitude)
        diff_lon = math.radians(target_lon - self.longitude)
        
        a = (math.sin(diff_lat/2) * math.sin(diff_lat/2) + 
             math.cos(lat1) * math.cos(lat2) * 
             math.sin(diff_lon/2) * math.sin(diff_lon/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        distance = R * c
        return distance
    
    def get_status_string(self):
        """Get GPS status for display"""
        if self.gps_lock:
            return f"GPS: {self.satellites}SAT HDOP:{self.hdop:.1f}"
        else:
            return f"GPS: NO LOCK ({self.satellites}SAT)"
    
    def close(self):
        """Close GPS connection"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("L76K GPS connection closed")

class MissionLogger:
    """Handles mission data logging and intelligence gathering with GPS support"""
    
    def __init__(self, mission_id=None, gps_module=None):
        self.mission_id = mission_id or f"MISSION_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = Path(f"mission_logs/{self.mission_id}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.detections = []
        self.threat_count = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        self.start_time = datetime.now()
        self.gps = gps_module
        
        # Get initial GPS position
        start_position = self.gps.get_position() if self.gps else {"latitude": 0, "longitude": 0, "valid": False}
        
        # Initialize mission log
        self.mission_data = {
            "mission_id": self.mission_id,
            "start_time": self.start_time.isoformat(),
            "start_position": start_position,
            "operator": os.getenv("USER", "SOLDIER_01"),
            "system_version": "ARCIS_v1.0_GPS",
            "detections": [],
            "summary": {}
        }
    
    def log_detection(self, detection_data):
        """Log a threat detection with full tactical data including GPS"""
        timestamp = datetime.now()
        
        # Get current GPS position
        gps_data = self.gps.get_position() if self.gps else {"latitude": 0, "longitude": 0, "valid": False}
        mgrs_coord = self.gps.get_mgrs_coordinates() if self.gps else "N/A"
        
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "threat_type": detection_data["class_name"],
            "confidence": detection_data["confidence"],
            "threat_level": detection_data["threat_level"],
            "distance": detection_data.get("distance", 0),
            "bearing": detection_data.get("bearing", "N/A"),
            "gps_coordinates": gps_data,
            "mgrs_coordinates": mgrs_coord,
            "engagement_rec": detection_data.get("engagement", {}),
            "frame_saved": detection_data.get("frame_path", "")
        }
        
        self.detections.append(log_entry)
        self.mission_data["detections"].append(log_entry)
        
        # Update threat counts
        threat_level = detection_data["threat_level"]
        if threat_level in self.threat_count:
            self.threat_count[threat_level] += 1
        
        # Save to file immediately for mission continuity
        self.save_mission_log()
    
    def save_mission_log(self):
        """Save current mission data to file"""
        current_position = self.gps.get_position() if self.gps else {"latitude": 0, "longitude": 0, "valid": False}
        
        self.mission_data["summary"] = {
            "duration_minutes": (datetime.now() - self.start_time).total_seconds() / 60,
            "total_detections": len(self.detections),
            "threat_breakdown": self.threat_count,
            "current_position": current_position,
            "last_updated": datetime.now().isoformat()
        }
        
        log_file = self.log_dir / "mission_log.json"
        with open(log_file, 'w') as f:
            json.dump(self.mission_data, f, indent=2)
    
    def generate_sitrep(self):
        """Generate situation report for command with GPS data"""
        duration = (datetime.now() - self.start_time).total_seconds() / 60
        current_pos = self.gps.get_position() if self.gps else {"latitude": 0, "longitude": 0, "valid": False}
        mgrs = self.gps.get_mgrs_coordinates() if self.gps else "N/A"
        
        sitrep = f"""
=== ARCIS SITUATION REPORT ===
Mission ID: {self.mission_id}
Duration: {duration:.1f} minutes
Operator: {self.mission_data['operator']}

POSITION:
- GPS: {current_pos['latitude']:.6f}, {current_pos['longitude']:.6f}
- MGRS: {mgrs}
- Altitude: {current_pos.get('altitude', 0):.1f}m
- GPS Status: {'LOCKED' if current_pos['valid'] else 'NO LOCK'}

THREAT SUMMARY:
- CRITICAL: {self.threat_count['CRITICAL']}
- HIGH: {self.threat_count['HIGH']}
- MEDIUM: {self.threat_count['MEDIUM']}
- LOW: {self.threat_count['LOW']}

Total Detections: {len(self.detections)}
Status: OPERATIONAL
=== END SITREP ===
"""
        return sitrep

def analyze_dataset(dataset_path):
    """Analyze the dataset structure and content"""
    print("\nDataset Analysis:")
    splits = ["train", "val", "test"]
    for split in splits:
        images_path = os.path.join(dataset_path, split, "images")
        labels_path = os.path.join(dataset_path, split, "labels")
        if os.path.exists(images_path) and os.path.exists(labels_path):
            num_images = len(os.listdir(images_path))
            num_labels = len(os.listdir(labels_path))
            print(f"{split}: {num_images} images, {num_labels} label files")
        else:
            print(f"{split}: Directory not found")

def load_class_names(data_yaml_path):
    """Load class names from data.yaml file"""
    with open(data_yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data.get('names', [])

def get_display_name(class_name, class_names):
    """Get display name with fallback to category defaults"""
    if isinstance(class_name, int):
        if class_name < len(class_names):
            original_name = class_names[class_name]
        else:
            original_name = f"class_{class_name}"
    else:
        original_name = class_name
    
    # Convert to lowercase for comparison
    name_lower = original_name.lower()
    
    # Weapon categories
    weapon_keywords = ['gun', 'rifle', 'pistol', 'knife', 'sword', 'handgun', 'automatic_rifle', 
                      'bazooka', 'grenade_launcher', 'weapon_holding']
    
    # Military vehicle categories  
    vehicle_keywords = ['tank', 'truck', 'military_tank', 'military_truck', 'armored']
    
    # Aircraft categories
    aircraft_keywords = ['aircraft', 'plane', 'helicopter', 'drone', 'military_aircraft', 
                        'military_helicopter', 'civilian_aircraft']
    
    # Check for specific matches first
    if any(keyword in name_lower for keyword in weapon_keywords):
        if 'unknown' in name_lower or name_lower == 'weapon':
            return 'weapon'
        return original_name
    elif any(keyword in name_lower for keyword in vehicle_keywords):
        if 'unknown' in name_lower:
            return 'vehicle'
        return original_name
    elif any(keyword in name_lower for keyword in aircraft_keywords):
        if 'unknown' in name_lower:
            return 'aircraft'
        return original_name
    
    return original_name

def get_threat_level(class_name, confidence, distance=None):
    """Determine military threat level based on class, confidence, and distance"""
    name_lower = class_name.lower()
    
    # Critical threats (immediate danger)
    critical_threats = ['military_tank', 'military_aircraft', 'military_helicopter', 'bazooka', 'grenade_launcher']
    high_threats = ['gun', 'rifle', 'automatic_rifle', 'weapon', 'high_warning']
    medium_threats = ['handgun', 'knife', 'weapon_holding']
    
    # Base threat level
    if any(threat in name_lower for threat in critical_threats):
        base_level = "CRITICAL"
    elif any(threat in name_lower for threat in high_threats):
        base_level = "HIGH"
    elif any(threat in name_lower for threat in medium_threats):
        base_level = "MEDIUM"
    else:
        base_level = "LOW"
    
    # Adjust based on confidence
    if confidence < 0.4:
        # Lower confidence reduces threat level
        if base_level == "CRITICAL":
            return "HIGH"
        elif base_level == "HIGH":
            return "MEDIUM"
        elif base_level == "MEDIUM":
            return "LOW"
    elif confidence >= 0.8:
        # High confidence maintains or elevates threat
        return base_level
    
    # Adjust based on distance (closer = higher threat)
    if distance and distance < 50:  # Within 50 meters
        if base_level == "HIGH":
            return "CRITICAL"
        elif base_level == "MEDIUM":
            return "HIGH"
    
    return base_level

def get_engagement_recommendation(class_name, distance=None):
    """Get tactical engagement recommendation"""
    name_lower = class_name.lower()
    
    # Default recommendation
    default_rec = {"action": "OBSERVE", "method": "ASSESS", "priority": "LOW"}
    
    # Find specific recommendation
    for threat_type, rec in ENGAGEMENT_RECOMMENDATIONS.items():
        if threat_type in name_lower:
            recommendation = rec.copy()
            
            # Adjust based on distance
            if distance:
                if distance < 25:  # Very close
                    recommendation["priority"] = "IMMEDIATE"
                elif distance > 200:  # Far away
                    if recommendation["action"] == "ENGAGE":
                        recommendation["action"] = "MONITOR"
            
            return recommendation
    
    return default_rec

def estimate_distance_imx415(bbox_height, focal_length_mm=2.8, sensor_height_mm=2.74, 
                           real_object_height_m=1.7, image_height_px=1080):
    """
    Estimate distance to object using IMX415 sensor specifications
    
    Args:
        bbox_height: Height of bounding box in pixels
        focal_length_mm: Focal length of IMX415 lens (typically 2.8mm)
        sensor_height_mm: Physical sensor height (2.74mm for IMX415)
        real_object_height_m: Assumed real height of object (1.7m for person, adjust for vehicles)
        image_height_px: Image height in pixels (1080p)
    
    Returns:
        Estimated distance in meters
    """
    if bbox_height <= 0:
        return 0
    
    # Calculate distance using similar triangles
    # distance = (real_height * focal_length * image_height) / (bbox_height * sensor_height)
    distance = (real_object_height_m * focal_length_mm * image_height_px) / (bbox_height * sensor_height_mm)
    
    # Convert from mm to meters and apply calibration factor
    distance_m = distance / 1000.0
    
    # Apply empirical calibration factor (may need adjustment based on real-world testing)
    calibration_factor = 0.8
    distance_m *= calibration_factor
    
    return max(0.1, min(distance_m, 100.0))  # Clamp between 0.1m and 100m

def calculate_bearing_with_gps(bbox_center_x, frame_width, gps_module=None, compass_heading=0):
    """Calculate bearing to detected object using GPS and camera FOV"""
    # Assume 60-degree horizontal field of view for IMX415
    fov_degrees = 60
    
    # Calculate offset from center
    center_offset = (bbox_center_x - frame_width/2) / (frame_width/2)
    relative_bearing = center_offset * (fov_degrees/2)
    
    # Use GPS course if available, otherwise use compass heading
    if gps_module and gps_module.gps_lock and gps_module.course > 0:
        base_heading = gps_module.course
    else:
        base_heading = compass_heading
    
    # Calculate absolute bearing
    absolute_bearing = (base_heading + relative_bearing) % 360
    
    return absolute_bearing

def train_yolo_for_jetson(
    data_yaml_path: str,
    epochs: int = 100,
    model_type: str = "yolov8n.pt",  # Nano model for Jetson
    run_name: str = None,
    target_device: str = "jetson"
):
    """Train YOLO model optimized for Jetson Nano deployment"""
    # Initialize model
    model = YOLO(model_type)
    
    # Print GPU information
    device_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f"\nTraining on device: {device_info}")
    
    # Jetson-optimized training parameters
    if target_device == "jetson":
        imgsz = 416  # Smaller image size for Jetson performance
        batch_size = 4  # Smaller batch size for memory constraints
        workers = 2  # Fewer workers for Jetson
        print("Using Jetson Nano optimized settings:")
        print(f"  Image size: {imgsz}")
        print(f"  Batch size: {batch_size}")
        print(f"  Workers: {workers}")
    else:
        imgsz = 640
        batch_size = 16 if torch.cuda.is_available() else 4
        workers = 4
    
    # Set run name if provided
    project = "runs/detect"
    name = run_name if run_name else "arcis_jetson_gps"
    
    # Train the model
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=0 if torch.cuda.is_available() else 'cpu',
        patience=30,  # Reduced patience for faster training
        save=True,
        plots=True,
        workers=workers,
        cache=False,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        project=project,
        name=name,
        # Jetson-specific optimizations
        amp=False,  # Disable mixed precision for compatibility
        fraction=0.8 if target_device == "jetson" else 1.0  # Use subset for faster training
    )
    
    print("\nTraining completed!")
    best_model_path = Path(f"{project}/{name}/weights/best.pt")
    print(f"Best model saved at: {best_model_path}")
    
    # Export optimized formats for Jetson deployment
    print("\nExporting model for Jetson deployment...")
    export_model = YOLO(best_model_path)
    
    # Export to ONNX (primary format for Jetson)
    onnx_path = export_model.export(format="onnx", imgsz=imgsz, simplify=True, dynamic=False)
    print(f"ONNX model exported: {onnx_path}")
    
    # Export to TensorRT for maximum Jetson performance (if available)
    try:
        trt_path = export_model.export(format="engine", imgsz=imgsz, half=True, device=0)
        print(f"TensorRT model exported: {trt_path}")
    except Exception as e:
        print(f"TensorRT export failed (normal on non-Jetson systems): {e}")
    
    return str(best_model_path)

def play_danger_alert():
    """Play a danger alert sound using pygame"""
    try:
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Look for sound file
        sound_files = ["danger_alert.mp3", "alert.mp3", "danger_alert.wav"]
        sound_file = None
        
        for file in sound_files:
            if os.path.exists(file):
                sound_file = file
                break
        
        if not sound_file:
            print("Warning: No alert sound file found. Using system beep.")
            for _ in range(3):
                print("\a")  # System beep
                time.sleep(0.3)
            return
        
        # Load and play sound
        sound = pygame.mixer.Sound(sound_file)
        sound.play()
        time.sleep(1)  # Allow time for sound to play
            
    except Exception as e:
        print(f"Could not play alert sound: {e}")
        # Fallback to system beep
        for _ in range(3):
            print("\a")
            time.sleep(0.3)

def resize_for_display(frame, target_size, maintain_aspect=True):
    """Resize frame for specific display while maintaining aspect ratio"""
    if maintain_aspect:
        h, w = frame.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w/w, target_h/h)
        new_w, new_h = int(w*scale), int(h*scale)
        
        # Resize frame
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Create black canvas and center the image
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas, scale, (x_offset, y_offset)
    else:
        return cv2.resize(frame, target_size), 1.0, (0, 0)

def run_tactical_inference_with_gps(
    model_path="runs/detect/arcis_jetson_gps/weights/best.pt",
    data_yaml_path="ARCIS_Dataset_80_10_10/data.yaml",
    source=0,  # 0 for webcam, or video file path
    display_config="2K_square",  # "2K_square" or "1080p"
    conf_thres=0.25,
    enable_sound=True,
    enable_distance=True,
    enable_logging=True,
    enable_gps=True,
    gps_port='/dev/ttyUSB0',
    mission_id=None,
    operator_id="SOLDIER_01"
):
    """
    Run tactical inference with L76K GPS integration for full military field operations
    """
    # Get display configuration
    config = JETSON_CONFIGS.get(display_config, JETSON_CONFIGS["2K_square"])
    print(f"=== ARCIS TACTICAL SYSTEM WITH L76K GPS ===")
    print(f"Display: {config['description']}")
    print(f"Operator: {operator_id}")
    print(f"Mission: {mission_id or 'PATROL'}")
    
    # Initialize GPS module
    gps_module = None
    if enable_gps:
        print(f"Initializing L76K GPS on {gps_port}...")
        try:
            gps_module = L76K_GPS(port=gps_port)
            if gps_module.serial_conn:
                print("GPS module initialized successfully")
                # Wait for GPS lock
                print("Waiting for GPS lock...")
                for i in range(30):  # Wait up to 30 seconds
                    gps_module.read_gps_data()
                    if gps_module.gps_lock:
                        print(f"GPS lock acquired! Position: {gps_module.latitude:.6f}, {gps_module.longitude:.6f}")
                        break
                    time.sleep(1)
                    print(f"GPS status: {gps_module.get_status_string()}")
                    
                if not gps_module.gps_lock:
                    print("Warning: GPS lock not acquired within 30 seconds. Continuing with GPS enabled...")
            else:
                print("GPS initialization failed - continuing without GPS")
                gps_module = None
        except Exception as e:
            print(f"GPS initialization error: {e}")
            print("Continuing without GPS...")
            gps_module = None
    
    # Initialize mission logger with GPS
    logger = MissionLogger(mission_id, gps_module) if enable_logging else None
    
    # Load model and class names
    model = YOLO(model_path)
    class_names = load_class_names(data_yaml_path)
    
    print(f"Loaded {len(class_names)} threat classes")
    
    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Set camera resolution for IMX415 (if using webcam)
    if source == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Alert management
    alert_thread = None
    last_alert_time = 0
    alert_cooldown = 3  # seconds between alerts
    
    # Tactical tracking
    active_threats = []
    threat_history = []
    frame_count = 0
    
    # FPS calculation
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    
    print("=== SYSTEM OPERATIONAL ===")
    print("Controls:")
    print("  'q' - Quit system")
    print("  's' - Save screenshot with GPS coordinates")
    print("  'r' - Generate SITREP with GPS data")
    print("  'g' - Show detailed GPS status")
    print("  'c' - Clear threat history")
    print("  'm' - Show MGRS coordinates")
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Error: Failed to read from video source")
                break
            
            frame_count += 1
            
            # Update GPS data
            if gps_module:
                gps_module.read_gps_data()
            
            # FPS calculation
            fps_counter += 1
            if fps_counter % 30 == 0:  # Update FPS every 30 frames
                current_fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
            
            # Run inference with optimized size
            results = model(frame, imgsz=config['inference_size'], conf=conf_thres)
            
            # Process results
            annotated_frame = frame.copy()
            current_threats = []
            highest_threat_level = "LOW"
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Get confidence and class
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        
                        # Get display name and tactical information
                        display_name = get_display_name(cls, class_names)
                        
                        # Calculate distance and bearing
                        bbox_height = y2 - y1
                        distance = estimate_distance_imx415(bbox_height) if enable_distance else None
                        bearing = calculate_bearing_with_gps((x1+x2)/2, frame.shape[1], gps_module)
                        
                        # Determine threat level
                        threat_level = get_threat_level(display_name, conf, distance)
                        threat_info = THREAT_LEVELS[threat_level]
                        
                        # Get engagement recommendation
                        engagement = get_engagement_recommendation(display_name, distance)
                        
                        # Track threat with GPS data
                        threat_data = {
                            "class_name": display_name,
                            "confidence": conf,
                            "threat_level": threat_level,
                            "distance": distance,
                            "bearing": bearing,
                            "engagement": engagement,
                            "bbox": (x1, y1, x2, y2),
                            "timestamp": datetime.now(),
                            "gps_coords": gps_module.get_position() if gps_module else None
                        }
                        current_threats.append(threat_data)
                        
                        # Update highest threat level
                        if threat_info["priority"] < THREAT_LEVELS[highest_threat_level]["priority"]:
                            highest_threat_level = threat_level
                        
                        # Draw tactical bounding box
                        color = threat_info["color"]
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, config['line_thickness'])
                        
                        # Tactical label with threat code
                        font_scale = config['font_scale']
                        threat_code = threat_info["code"]
                        label = f"{threat_code}: {display_name} ({conf:.2f})"
                        
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
                        cv2.rectangle(annotated_frame, (x1, y1-label_size[1]-5), 
                                    (x1+label_size[0], y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1-3), 
                                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
                        
                        # Distance and bearing info
                        if distance:
                            dist_text = f"{distance:.0f}m"
                            bearing_text = f"{bearing:.0f}°"
                            info_text = f"{dist_text} | {bearing_text}"
                            
                            info_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.7, 1)[0]
                            cv2.rectangle(annotated_frame, (x2-info_size[0]-3, y1), 
                                        (x2, y1+info_size[1]+3), (0, 0, 255), -1)
                            cv2.putText(annotated_frame, info_text, (x2-info_size[0], y1+info_size[1]), 
                                       cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.7, (255, 255, 255), 1)
                        
                        # Engagement recommendation
                        if engagement["priority"] in ["IMMEDIATE", "CRITICAL"]:
                            rec_text = f"ACTION: {engagement['action']}"
                            cv2.putText(annotated_frame, rec_text, (x1, y2+15), 
                                       cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, color, 1)
                        
                        # Log detection if enabled
                        if logger and threat_level in ["CRITICAL", "HIGH"]:
                            logger.log_detection(threat_data)
            
            # Update active threats
            active_threats = current_threats
            
            # Play alert for critical/high threats
            current_time = time.time()
            critical_threat = any(t["threat_level"] in ["CRITICAL", "HIGH"] for t in current_threats)
            
            if critical_threat and enable_sound and (current_time - last_alert_time) > alert_cooldown:
                if alert_thread is None or not alert_thread.is_alive():
                    alert_thread = threading.Thread(target=play_danger_alert)
                    alert_thread.daemon = True
                    alert_thread.start()
                    last_alert_time = current_time
            
            # Tactical status display
            status_y = 25
            font_scale = config['font_scale']
            
            # Threat level indicator
            threat_color = THREAT_LEVELS[highest_threat_level]["color"]
            status_text = f"THREAT LEVEL: {highest_threat_level}"
            cv2.putText(annotated_frame, status_text, (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale*1.2, threat_color, 2)
            status_y += 30
            
            # GPS status
            if gps_module:
                gps_status = gps_module.get_status_string()
                gps_color = (0, 255, 0) if gps_module.gps_lock else (0, 0, 255)
                cv2.putText(annotated_frame, gps_status, (10, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, gps_color, 1)
                status_y += 25
                
                # Show coordinates if GPS locked
                if gps_module.gps_lock:
                    coord_text = f"LAT: {gps_module.latitude:.6f} LON: {gps_module.longitude:.6f}"
                    cv2.putText(annotated_frame, coord_text, (10, status_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.6, (0, 255, 0), 1)
                    status_y += 20
            
            # Active threats count
            threat_count = len(current_threats)
            if threat_count > 0:
                count_text = f"ACTIVE THREATS: {threat_count}"
                cv2.putText(annotated_frame, count_text, (10, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
                status_y += 25
            
            # System status
            system_text = f"ARCIS GPS | FPS: {current_fps:.1f} | CONF: {conf_thres}"
            cv2.putText(annotated_frame, system_text, (10, annotated_frame.shape[0]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, (255, 255, 255), 1)
            
            # Mission time
            if logger:
                mission_time = (datetime.now() - logger.start_time).total_seconds() / 60
                time_text = f"MISSION TIME: {mission_time:.1f}min"
                cv2.putText(annotated_frame, time_text, (10, annotated_frame.shape[0]-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, (255, 255, 255), 1)
            
            # Resize for target display
            display_frame, scale, offset = resize_for_display(annotated_frame, config['display_size'])
            
            # Display the frame
            cv2.imshow("ARCIS TACTICAL SYSTEM - L76K GPS ENABLED", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save tactical screenshot with GPS data
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                gps_info = ""
                if gps_module and gps_module.gps_lock:
                    gps_info = f"_GPS_{gps_module.latitude:.6f}_{gps_module.longitude:.6f}"
                filename = f"tactical_capture_{timestamp}{gps_info}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"Tactical screenshot saved: {filename}")
            elif key == ord('r') and logger:
                # Generate and print SITREP
                sitrep = logger.generate_sitrep()
                print(sitrep)
            elif key == ord('g') and gps_module:
                # Show detailed GPS status
                pos = gps_module.get_position()
                mgrs = gps_module.get_mgrs_coordinates()
                print(f"\n=== L76K GPS STATUS ===")
                print(f"Lock: {'YES' if pos['valid'] else 'NO'}")
                print(f"Coordinates: {pos['latitude']:.6f}, {pos['longitude']:.6f}")
                print(f"MGRS: {mgrs}")
                print(f"Altitude: {pos['altitude']:.1f}m")
                print(f"Satellites: {pos['satellites']}")
                print(f"HDOP: {pos['hdop']:.1f}")
                print(f"Speed: {gps_module.speed:.1f} knots")
                print(f"Course: {gps_module.course:.1f}°")
                print(f"======================")
            elif key == ord('m') and gps_module:
                # Show MGRS coordinates
                mgrs = gps_module.get_mgrs_coordinates()
                print(f"MGRS Coordinates: {mgrs}")
            elif key == ord('c'):
                # Clear threat history
                threat_history.clear()
                print("Threat history cleared")
    
    except KeyboardInterrupt:
        print("\nSystem shutdown requested...")
    
    finally:
        # Mission debrief
        if logger:
            print("\n=== MISSION DEBRIEF ===")
            print(logger.generate_sitrep())
            logger.save_mission_log()
            print(f"Mission log saved to: {logger.log_dir}")
        
        # Close GPS connection
        if gps_module:
            gps_module.close()
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # ARCIS Dataset configuration
    dataset_path = Path("ARCIS_Dataset_80_10_10")
    data_yaml = str(dataset_path / "data.yaml")
    
    print("=== ARCIS TACTICAL WEAPON DETECTION SYSTEM WITH L76K GPS ===")
    print("1. Train model for Jetson Nano deployment")
    print("2. Run tactical inference with L76K GPS (2.9\" 2K display)")
    print("3. Run tactical inference with L76K GPS (1080p display)")
    print("4. Test L76K GPS connection only")
    print("5. Analyze ARCIS dataset")
    
    choice = input("Enter your choice (1-5): ")
    
    if choice == '1':
        # Train for Jetson deployment
        print("\nTraining model optimized for Jetson Nano with GPS support...")
        analyze_dataset(dataset_path)
        
        run_name = input("Enter training run name (press Enter for 'arcis_jetson_gps'): ") or "arcis_jetson_gps"
        
        # Jetson-optimized training
        model_path = train_yolo_for_jetson(
            data_yaml_path=data_yaml,
            epochs=100,  # Reasonable for Jetson
            model_type="yolov8n.pt",  # Nano model for performance
            run_name=run_name,
            target_device="jetson"
        )
        
        print(f"\nJetson GPS model training completed!")
        print(f"Model saved at: {model_path}")
        print("ONNX and TensorRT exports created for deployment.")
        
    elif choice in ['2', '3']:
        # Run tactical inference with GPS
        display_config = "2K_square" if choice == '2' else "1080p"
        model_path = input("Enter model path (press Enter for default): ") or "runs/detect/arcis_jetson_gps/weights/best.pt"
        
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            print("Please train a model first or use existing model.")
        else:
            # Get mission and GPS details
            mission_id = input("Enter mission ID (press Enter for auto): ") or None
            operator_id = input("Enter operator ID (press Enter for SOLDIER_01): ") or "SOLDIER_01"
            gps_port = input("Enter L76K GPS port (press Enter for /dev/ttyUSB0): ") or "/dev/ttyUSB0"
            enable_gps = input("Enable L76K GPS? (y/n, default: y): ").lower() != 'n'
            
            print(f"Starting tactical inference with L76K GPS for {JETSON_CONFIGS[display_config]['description']}...")
            run_tactical_inference_with_gps(
                model_path=model_path, 
                data_yaml_path=data_yaml, 
                display_config=display_config,
                mission_id=mission_id,
                operator_id=operator_id,
                enable_gps=enable_gps,
                gps_port=gps_port
            )
    
    elif choice == '4':
        # Test GPS connection only
        gps_port = input("Enter L76K GPS port (press Enter for /dev/ttyUSB0): ") or "/dev/ttyUSB0"
        print(f"Testing L76K GPS connection on {gps_port}...")
        
        gps = L76K_GPS(port=gps_port)
        if gps.serial_conn:
            print("GPS connected successfully!")
            print("Waiting for GPS data...")
            
            for i in range(60):  # Test for 60 seconds
                if gps.read_gps_data():
                    pos = gps.get_position()
                    mgrs = gps.get_mgrs_coordinates()
                    print(f"GPS Data: {pos['latitude']:.6f}, {pos['longitude']:.6f}")
                    print(f"MGRS: {mgrs}")
                    print(f"Status: {gps.get_status_string()}")
                    if gps.gps_lock:
                        print("GPS lock acquired!")
                        break
                time.sleep(1)
            
            gps.close()
        else:
            print("Failed to connect to GPS module")
    
    elif choice == '5':
        # Analyze dataset only
        analyze_dataset(dataset_path)
        
        class_names = load_class_names(data_yaml)
        print(f"\nARCIS Dataset Classes ({len(class_names)}):")
        for i, name in enumerate(class_names):
            print(f"  {i}: {name}")
    
    else:
        print("Invalid choice. Exiting.") 