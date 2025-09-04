# Import the original training script functionality
from train_weapon_detection import *
from arcis_redis_integration import ARCISRedisManager, ARCISRedisIntegration, ThreatDetection
import os

def run_tactical_inference_with_redis(
    model_path="runs/detect/arcis_jetson/weights/best.pt",
    data_yaml_path="ARCIS_Dataset_80_10_10/data.yaml",
    source=0,  # 0 for webcam, or video file path
    display_config="2K_square",  # "2K_square" or "1080p"
    conf_thres=0.25,
    enable_sound=True,
    enable_distance=True,
    enable_logging=True,
    mission_id=None,
    operator_id="SOLDIER_01",
    redis_host="localhost",
    redis_port=6379,
    google_cloud_project=None
):
    """
    Run tactical inference with Redis integration for distributed processing
    """
    # Initialize Redis integration
    try:
        redis_manager = ARCISRedisManager(
            redis_host=redis_host,
            redis_port=redis_port,
            google_cloud_project=google_cloud_project
        )
        redis_integration = ARCISRedisIntegration(redis_manager)
        print("✓ Redis integration initialized")
    except Exception as e:
        print(f"✗ Redis connection failed: {e}")
        print("Running without Redis integration...")
        redis_integration = None
    
    # Get display configuration
    config = JETSON_CONFIGS.get(display_config, JETSON_CONFIGS["2K_square"])
    print(f"=== ARCIS TACTICAL SYSTEM WITH REDIS INTEGRATION ===")
    print(f"Display: {config['description']}")
    print(f"Operator: {operator_id}")
    print(f"Mission: {mission_id or 'PATROL'}")
    print(f"Redis: {'ENABLED' if redis_integration else 'DISABLED'}")
    
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
    
    # Redis statistics display
    last_stats_update = 0
    stats_update_interval = 5  # Update stats every 5 seconds
    
    print("=== SYSTEM OPERATIONAL ===")
    print("Controls:")
    print("  'q' - Quit system")
    print("  's' - Save screenshot")
    print("  'r' - Generate SITREP")
    print("  'c' - Clear threat history")
    print("  't' - Show Redis statistics")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read from video source")
            break
        
        frame_count += 1
        current_time = time.time()
        
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
                    
                    # Send to Redis if enabled
                    if redis_integration:
                        try:
                            redis_integration.process_detection(
                                threat_data, 
                                frame, 
                                mission_id or "PATROL", 
                                operator_id
                            )
                        except Exception as e:
                            print(f"Redis processing error: {e}")
                    
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
        
        # Redis status indicator
        if redis_integration:
            redis_status = "REDIS: CONNECTED"
            cv2.putText(annotated_frame, redis_status, (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, (0, 255, 0), 1)
            status_y += 20
            
            # Show Redis statistics periodically
            if current_time - last_stats_update > stats_update_interval:
                try:
                    stats = redis_integration.redis_manager.get_threat_statistics()
                    print(f"\n=== REDIS STATISTICS ===")
                    for key, value in stats.items():
                        print(f"{key}: {value}")
                    last_stats_update = current_time
                except Exception as e:
                    print(f"Failed to get Redis stats: {e}")
        
        # System status
        system_text = f"ARCIS+REDIS | FPS: {current_fps:.1f} | CONF: {conf_thres}"
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
        cv2.imshow("ARCIS TACTICAL SYSTEM + REDIS", display_frame)
        
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
        elif key == ord('t') and redis_integration:
            # Show Redis statistics
            try:
                stats = redis_integration.redis_manager.get_threat_statistics()
                print(f"\n=== REDIS STATISTICS ===")
                for key, value in stats.items():
                    print(f"{key}: {value}")
            except Exception as e:
                print(f"Failed to get Redis stats: {e}")
    
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
    # ARCIS Dataset configuration
    dataset_path = Path("ARCIS_Dataset_80_10_10")
    data_yaml = str(dataset_path / "data.yaml")
    
    print("=== ARCIS TACTICAL WEAPON DETECTION SYSTEM WITH REDIS ===")
    print("1. Train model for Jetson Nano deployment")
    print("2. Run tactical inference with Redis (2.9\" 2K display)")
    print("3. Run tactical inference with Redis (1080p display)")
    print("4. Analyze ARCIS dataset")
    print("5. Train full-size model (for desktop/server)")
    print("6. Test Redis connection only")
    
    choice = input("Enter your choice (1-6): ")
    
    if choice == '1':
        # Train for Jetson deployment (same as original)
        print("\nTraining model optimized for Jetson Nano...")
        analyze_dataset(dataset_path)
        
        run_name = input("Enter training run name (press Enter for 'arcis_jetson'): ") or "arcis_jetson"
        
        # Jetson-optimized training
        model_path = train_yolo_for_jetson(
            data_yaml_path=data_yaml,
            epochs=100,
            model_type="yolov8n.pt",
            run_name=run_name,
            target_device="jetson"
        )
        
        print(f"\nJetson model training completed!")
        print(f"Model saved at: {model_path}")
        print("ONNX and TensorRT exports created for deployment.")
        
    elif choice in ['2', '3']:
        # Run tactical inference with Redis
        display_config = "2K_square" if choice == '2' else "1080p"
        model_path = input("Enter model path (press Enter for default): ") or "runs/detect/arcis_jetson/weights/best.pt"
        
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            print("Please train a model first.")
        else:
            # Get mission details
            mission_id = input("Enter mission ID (press Enter for auto): ") or None
            operator_id = input("Enter operator ID (press Enter for SOLDIER_01): ") or "SOLDIER_01"
            
            # Redis configuration
            redis_host = input("Enter Redis host (press Enter for localhost): ") or "localhost"
            redis_port = int(input("Enter Redis port (press Enter for 6379): ") or "6379")
            
            # Google Cloud configuration
            gcp_project = input("Enter Google Cloud Project ID (press Enter to skip): ") or None
            
            print(f"Starting tactical inference with Redis for {JETSON_CONFIGS[display_config]['description']}...")
            run_tactical_inference_with_redis(
                model_path=model_path, 
                data_yaml_path=data_yaml, 
                display_config=display_config,
                mission_id=mission_id,
                operator_id=operator_id,
                redis_host=redis_host,
                redis_port=redis_port,
                google_cloud_project=gcp_project
            )
    
    elif choice == '4':
        # Analyze dataset only (same as original)
        analyze_dataset(dataset_path)
        plot_sample_images(dataset_path)
        
        class_names = load_class_names(data_yaml)
        print(f"\nARCIS Dataset Classes ({len(class_names)}):")
        for i, name in enumerate(class_names):
            print(f"  {i}: {name}")
    
    elif choice == '5':
        # Train full-size model (same as original)
        print("\nTraining full-size model for desktop/server deployment...")
        analyze_dataset(dataset_path)
        
        run_name = input("Enter training run name (press Enter for 'arcis_full'): ") or "arcis_full"
        
        model_path = train_yolo_for_jetson(
            data_yaml_path=data_yaml,
            epochs=150,
            model_type="yolov8s.pt",
            run_name=run_name,
            target_device="desktop"
        )
        
        print(f"\nFull model training completed!")
        print(f"Model saved at: {model_path}")
    
    elif choice == '6':
        # Test Redis connection
        print("\nTesting Redis connection...")
        redis_host = input("Enter Redis host (press Enter for localhost): ") or "localhost"
        redis_port = int(input("Enter Redis port (press Enter for 6379): ") or "6379")
        gcp_project = input("Enter Google Cloud Project ID (press Enter to skip): ") or None
        
        try:
            redis_manager = ARCISRedisManager(
                redis_host=redis_host,
                redis_port=redis_port,
                google_cloud_project=gcp_project
            )
            
            print("✓ Redis connection successful!")
            
            # Test basic operations
            test_data = {"test": "data", "timestamp": datetime.now().isoformat()}
            redis_manager.redis_client.setex("arcis:test", 60, json.dumps(test_data))
            retrieved = redis_manager.redis_client.get("arcis:test")
            
            if retrieved:
                print("✓ Redis read/write test successful!")
                print(f"Test data: {json.loads(retrieved)}")
            
            # Show statistics
            stats = redis_manager.get_threat_statistics()
            print(f"\nRedis Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
                
        except Exception as e:
            print(f"✗ Redis connection failed: {e}")
    
    else:
        print("Invalid choice. Exiting.") 