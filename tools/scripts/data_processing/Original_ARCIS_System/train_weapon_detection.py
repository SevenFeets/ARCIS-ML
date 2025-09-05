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

class MissionLogger:
    """Handles mission data logging and intelligence gathering"""
    
    def __init__(self, mission_id=None):
        self.mission_id = mission_id or f"MISSION_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = Path(f"mission_logs/{self.mission_id}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.detections = []
        self.threat_count = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        self.start_time = datetime.now()
        
        # Initialize mission log
        self.mission_data = {
            "mission_id": self.mission_id,
            "start_time": self.start_time.isoformat(),
            "operator": os.getenv("USER", "SOLDIER_01"),
            "system_version": "ARCIS_v1.0",
            "detections": [],
            "summary": {}
        }
    
    def log_detection(self, detection_data):
        """Log a threat detection with full tactical data"""
        timestamp = datetime.now()
        
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "threat_type": detection_data["class_name"],
            "confidence": detection_data["confidence"],
            "threat_level": detection_data["threat_level"],
            "distance": detection_data.get("distance", 0),
            "coordinates": detection_data.get("gps_coords", "N/A"),
            "bearing": detection_data.get("bearing", "N/A"),
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
        self.mission_data["summary"] = {
            "duration_minutes": (datetime.now() - self.start_time).total_seconds() / 60,
            "total_detections": len(self.detections),
            "threat_breakdown": self.threat_count,
            "last_updated": datetime.now().isoformat()
        }
        
        log_file = self.log_dir / "mission_log.json"
        with open(log_file, 'w') as f:
            json.dump(self.mission_data, f, indent=2)
    
    def generate_sitrep(self):
        """Generate situation report for command"""
        duration = (datetime.now() - self.start_time).total_seconds() / 60
        
        sitrep = f"""
=== ARCIS SITUATION REPORT ===
Mission ID: {self.mission_id}
Duration: {duration:.1f} minutes
Operator: {self.mission_data['operator']}

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

def is_danger_class(class_name, class_names):
    """Check if the detected class is considered dangerous"""
    if isinstance(class_name, int):
        if class_name < len(class_names):
            original_name = class_names[class_name]
        else:
            return False
    else:
        original_name = class_name
    
    name_lower = original_name.lower()
    
    # Define dangerous categories
    danger_keywords = ['weapon', 'gun', 'rifle', 'pistol', 'knife', 'sword', 'handgun', 
                      'automatic_rifle', 'bazooka', 'grenade_launcher', 'weapon_holding',
                      'military_tank', 'military_truck', 'military_aircraft', 'military_helicopter',
                      'high_warning']
    
    return any(keyword in name_lower for keyword in danger_keywords)

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

def calculate_bearing(bbox_center_x, frame_width, compass_heading=0):
    """Calculate relative bearing to detected object"""
    # Assume 60-degree horizontal field of view
    fov_degrees = 60
    
    # Calculate offset from center
    center_offset = (bbox_center_x - frame_width/2) / (frame_width/2)
    relative_bearing = center_offset * (fov_degrees/2)
    
    # Add to compass heading (if available)
    absolute_bearing = (compass_heading + relative_bearing) % 360
    
    return absolute_bearing

def plot_sample_images(dataset_path, split="train", num_samples=5):
    """Plot sample images with their bounding boxes"""
    image_dir = os.path.join(dataset_path, split, "images")
    label_dir = os.path.join(dataset_path, split, "labels")
    
    if not os.path.exists(image_dir):
        print(f"Image directory not found: {image_dir}")
        return
    
    image_files = os.listdir(image_dir)
    if not image_files:
        print("No images found in directory")
        return
        
    image_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    plt.figure(figsize=(15, 3*len(image_files)))
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, img_file.replace(".jpg", ".txt").replace(".jpeg", ".txt").replace(".png", ".txt"))
        
        # Load and plot image
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.subplot(len(image_files), 1, idx + 1)
        plt.imshow(image)
        
        # Draw bounding boxes if label exists
        if os.path.exists(label_path):
            h, w, _ = image.shape
            with open(label_path, "r") as file:
                for line in file.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id, x_center, y_center, box_w, box_h = map(float, parts[:5])
                        x1 = int((x_center - box_w/2) * w)
                        y1 = int((y_center - box_h/2) * h)
                        x2 = int((x_center + box_w/2) * w)
                        y2 = int((y_center + box_h/2) * h)
                        
                        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                        edgecolor='red', linewidth=2, fill=False))
                        plt.text(x1, y1-5, f"Class {int(class_id)}", color='red', fontsize=12)
        
        plt.axis('off')
        plt.title(f"Sample {idx+1}: {img_file}")
    plt.tight_layout()
    plt.show()

def train_yolo_for_jetson(
    data_yaml_path: str,
    epochs: int = 100,
    model_type: str = "yolov8n.pt",  # Nano model for Jetson
    run_name: str = None,
    target_device: str = "jetson",
    hyperparameter_preset: str = "default"
):
    """Train YOLO model optimized for Jetson Nano deployment with hyperparameter options"""
    
    # Hyperparameter presets
    PRESETS = {
        "default": {
            "lr0": 0.01, "lrf": 0.1, "momentum": 0.937, "weight_decay": 0.0005,
            "warmup_epochs": 3, "warmup_momentum": 0.8, "warmup_bias_lr": 0.1,
            "box": 7.5, "cls": 0.5, "dfl": 1.5, "label_smoothing": 0.0,
            "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "degrees": 0.0,
            "translate": 0.1, "scale": 0.5, "fliplr": 0.5, "mosaic": 1.0, "mixup": 0.0
        },
        "weapon_optimized": {
            "lr0": 0.008, "lrf": 0.05, "momentum": 0.95, "weight_decay": 0.001,
            "warmup_epochs": 5, "warmup_momentum": 0.9, "warmup_bias_lr": 0.05,
            "box": 10.0, "cls": 0.8, "dfl": 2.0, "label_smoothing": 0.1,
            "hsv_h": 0.01, "hsv_s": 0.5, "hsv_v": 0.3, "degrees": 5.0,
            "translate": 0.05, "scale": 0.3, "fliplr": 0.3, "mosaic": 0.8, "mixup": 0.1
        },
        "jetson_optimized": {
            "lr0": 0.012, "lrf": 0.15, "momentum": 0.9, "weight_decay": 0.0003,
            "warmup_epochs": 2, "warmup_momentum": 0.7, "warmup_bias_lr": 0.15,
            "box": 6.0, "cls": 0.4, "dfl": 1.0, "label_smoothing": 0.05,
            "hsv_h": 0.02, "hsv_s": 0.6, "hsv_v": 0.4, "degrees": 10.0,
            "translate": 0.15, "scale": 0.6, "fliplr": 0.5, "mosaic": 1.0, "mixup": 0.0
        },
        "high_accuracy": {
            "lr0": 0.005, "lrf": 0.01, "momentum": 0.98, "weight_decay": 0.002,
            "warmup_epochs": 10, "warmup_momentum": 0.95, "warmup_bias_lr": 0.01,
            "box": 15.0, "cls": 1.0, "dfl": 3.0, "label_smoothing": 0.15,
            "hsv_h": 0.005, "hsv_s": 0.3, "hsv_v": 0.2, "degrees": 2.0,
            "translate": 0.02, "scale": 0.2, "fliplr": 0.2, "mosaic": 0.5, "mixup": 0.2
        }
    }
    
    # Get hyperparameters
    hyperparams = PRESETS.get(hyperparameter_preset, PRESETS["default"])
    
    # Initialize model
    model = YOLO(model_type)
    
    # Print GPU information
    device_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f"\nTraining on device: {device_info}")
    print(f"Hyperparameter preset: {hyperparameter_preset}")
    
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
    name = run_name if run_name else f"arcis_{hyperparameter_preset}"
    
    # Train the model with selected hyperparameters
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
        project=project,
        name=name,
        # Jetson-specific optimizations
        amp=False,  # Disable mixed precision for compatibility
        fraction=0.8 if target_device == "jetson" else 1.0,  # Use subset for faster training
        **hyperparams  # Unpack selected hyperparameters
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
        
        # Look for sound file - updated paths for folder organization
        sound_files = [
            "../Audio_Assets/danger_alert.mp3", 
            "danger_alert.mp3", 
            "../Audio_Assets/alert.mp3",
            "alert.mp3", 
            "danger_alert.wav"
        ]
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

def get_confidence_color(confidence):
    """
    Returns color based on confidence level:
    - Green: 0.0-0.4 confidence
    - Yellow: 0.4-0.8 confidence  
    - Red: 0.8-1.0 confidence
    """
    if confidence < 0.4:
        return (0, 255, 0)  # Green (BGR format)
    elif confidence < 0.8:
        return (0, 255, 255)  # Yellow
    else:
        return (0, 0, 255)  # Red

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

def run_tactical_inference(
    model_path="runs/detect/arcis_jetson/weights/best.pt",
    data_yaml_path="../ARCIS_Dataset_80_10_10/data.yaml",  # Updated path for folder organization
    source=0,  # 0 for webcam, or video file path
    display_config="2K_square",  # "2K_square" or "1080p"
    conf_thres=0.25,
    enable_sound=True,
    enable_distance=True,
    enable_logging=True,
    mission_id=None,
    operator_id="SOLDIER_01"
):
    """
    Run tactical inference with full military field operation features
    """
    # Get display configuration
    config = JETSON_CONFIGS.get(display_config, JETSON_CONFIGS["2K_square"])
    print(f"=== ARCIS TACTICAL SYSTEM INITIALIZED ===")
    print(f"Display: {config['description']}")
    print(f"Operator: {operator_id}")
    print(f"Mission: {mission_id or 'PATROL'}")
    
    # Initialize mission logger
    logger = MissionLogger(mission_id) if enable_logging else None
    
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
    print("  's' - Save screenshot")
    print("  'r' - Generate SITREP")
    print("  'c' - Clear threat history")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read from video source")
            break
        
        frame_count += 1
        
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
                    bearing = calculate_bearing((x1+x2)/2, frame.shape[1])
                    
                    # Determine threat level
                    threat_level = get_threat_level(display_name, conf, distance)
                    threat_info = THREAT_LEVELS[threat_level]
                    
                    # Get engagement recommendation
                    engagement = get_engagement_recommendation(display_name, distance)
                    
                    # Track threat
                    threat_data = {
                        "class_name": display_name,
                        "confidence": conf,
                        "threat_level": threat_level,
                        "distance": distance,
                        "bearing": bearing,
                        "engagement": engagement,
                        "bbox": (x1, y1, x2, y2),
                        "timestamp": datetime.now()
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
        
        # Active threats count
        threat_count = len(current_threats)
        if threat_count > 0:
            count_text = f"ACTIVE THREATS: {threat_count}"
            cv2.putText(annotated_frame, count_text, (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
            status_y += 25
        
        # System status
        system_text = f"ARCIS | FPS: {current_fps:.1f} | CONF: {conf_thres}"
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
        cv2.imshow("ARCIS TACTICAL SYSTEM", display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save tactical screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tactical_capture_{timestamp}.jpg"
            cv2.imwrite(filename, display_frame)
            print(f"Tactical screenshot saved: {filename}")
        elif key == ord('r') and logger:
            # Generate and print SITREP
            sitrep = logger.generate_sitrep()
            print(sitrep)
        elif key == ord('c'):
            # Clear threat history
            threat_history.clear()
            print("Threat history cleared")
    
    # Mission debrief
    if logger:
        print("\n=== MISSION DEBRIEF ===")
        print(logger.generate_sitrep())
        logger.save_mission_log()
        print(f"Mission log saved to: {logger.log_dir}")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("=== ARCIS TACTICAL WEAPON DETECTION SYSTEM ===")
    
    # Dataset selection
    print("\n📊 AVAILABLE DATASETS:")
    datasets = {
        "1": {"path": "../ARCIS_Dataset_80_10_10", "name": "80/10/10 Split", "desc": "80% train, 10% val, 10% test (Recommended for most training)"},
        "2": {"path": "../ARCIS_Dataset_70_15_15", "name": "70/15/15 Split", "desc": "70% train, 15% val, 15% test (More validation data)"},
        "3": {"path": "../ARCIS_Dataset_75_12_12", "name": "75/12.5/12.5 Split", "desc": "75% train, 12.5% val, 12.5% test (Balanced approach)"}
    }
    
    for key, dataset in datasets.items():
        status = "✅" if os.path.exists(dataset["path"]) else "❌"
        print(f"{key}. {status} {dataset['name']} - {dataset['desc']}")
    
    dataset_choice = input("\nSelect dataset (1-3, press Enter for default 80/10/10): ") or "1"
    
    if dataset_choice not in datasets:
        print("Invalid choice, using default 80/10/10 split")
        dataset_choice = "1"
    
    selected_dataset = datasets[dataset_choice]
    dataset_path = Path(selected_dataset["path"])
    data_yaml = str(dataset_path / "data.yaml")
    
    print(f"\n🎯 Selected: {selected_dataset['name']}")
    print(f"📁 Path: {dataset_path}")
    
    # Verify dataset exists
    if not dataset_path.exists():
        print(f"❌ Error: Dataset not found at {dataset_path}")
        print("💡 Run the dataset merger first: cd ../Dataset_Tools && python dataset_merger.py")
        exit(1)
    
    print("\n🚀 TRAINING/INFERENCE OPTIONS:")
    print("1. Train model for Jetson Nano deployment")
    print("2. Run tactical inference (2.9\" 2K display)")
    print("3. Run tactical inference (1080p display)")
    print("4. Analyze selected dataset")
    print("5. Train full-size model (for desktop/server)")
    
    choice = input("Enter your choice (1-5): ")
    
    if choice == '1':
        # Train for Jetson deployment
        print(f"\nTraining model optimized for Jetson Nano using {selected_dataset['name']}...")
        analyze_dataset(dataset_path)
        
        run_name = input("Enter training run name (press Enter for 'arcis_jetson'): ") or "arcis_jetson"
        
        # Jetson-optimized training - updated model path
        model_path = train_yolo_for_jetson(
            data_yaml_path=data_yaml,
            epochs=100,  # Reasonable for Jetson
            model_type="../Models/yolov8n.pt",  # Updated path for folder organization
            run_name=run_name,
            target_device="jetson"
        )
        
        print(f"\nJetson model training completed!")
        print(f"Model saved at: {model_path}")
        print("ONNX and TensorRT exports created for deployment.")
        
    elif choice in ['2', '3']:
        # Run tactical inference
        display_config = "2K_square" if choice == '2' else "1080p"
        model_path = input("Enter model path (press Enter for default): ") or "runs/detect/arcis_jetson/weights/best.pt"
        
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            print("Please train a model first.")
        else:
            # Get mission details
            mission_id = input("Enter mission ID (press Enter for auto): ") or None
            operator_id = input("Enter operator ID (press Enter for SOLDIER_01): ") or "SOLDIER_01"
            
            print(f"Starting tactical inference for {JETSON_CONFIGS[display_config]['description']}...")
            print(f"Using dataset: {selected_dataset['name']}")
            run_tactical_inference(
                model_path=model_path, 
                data_yaml_path=data_yaml, 
                display_config=display_config,
                mission_id=mission_id,
                operator_id=operator_id
            )
    
    elif choice == '4':
        # Analyze dataset only
        print(f"\nAnalyzing {selected_dataset['name']}...")
        analyze_dataset(dataset_path)
        plot_sample_images(dataset_path)
        
        class_names = load_class_names(data_yaml)
        print(f"\nARCIS Dataset Classes ({len(class_names)}):")
        for i, name in enumerate(class_names):
            print(f"  {i}: {name}")
    
    elif choice == '5':
        # Train full-size model for desktop/server
        print(f"\nTraining full-size model for desktop/server using {selected_dataset['name']}...")
        analyze_dataset(dataset_path)
        
        run_name = input("Enter training run name (press Enter for 'arcis_full'): ") or "arcis_full"
        
        # Full training configuration - updated model path
        model_path = train_yolo_for_jetson(
            data_yaml_path=data_yaml,
            epochs=150,
            model_type="../Models/yolo11n.pt",  # Use available YOLO11 model for better accuracy
            run_name=run_name,
            target_device="desktop"
        )
        
        print(f"\nFull model training completed!")
        print(f"Model saved at: {model_path}")
    
    else:
        print("Invalid choice. Exiting.") 