import cv2
import time
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run YOLO weapon detection on a webcam')
    parser.add_argument('--model', type=str, default='runs/detect/train/weights/best.pt',
                        help='Path to YOLO model file (.pt)')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='Confidence threshold for detections')
    parser.add_argument('--imgsz', type=int, default=416,
                        help='Image size for inference')
    parser.add_argument('--save', action='store_true',
                        help='Save video with detections')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to run inference on (cuda device or cpu)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (usually 0 for built-in webcam)')
    return parser.parse_args()

def list_cameras():
    """Test the first 10 camera indices to find available cameras"""
    available_cameras = {}
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                available_cameras[i] = f"{int(width)}x{int(height)}"
            cap.release()
    
    if available_cameras:
        print("\nAvailable cameras:")
        for idx, resolution in available_cameras.items():
            print(f"Camera {idx}: {resolution}")
    else:
        print("\nNo cameras found. Check your webcam connection.")
    
    return available_cameras

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # List available cameras
    cameras = list_cameras()
    if not cameras and args.camera not in cameras:
        print(f"Camera index {args.camera} not available.")
        return
    
    # Load the model
    print(f"Loading model from {args.model}...")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model file exists and is valid.")
        return
    
    # Open webcam
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        print("Please check your webcam connection or try a different camera index.")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera properties: {width}x{height} @ {fps:.1f}fps")
    
    # Initialize video writer if saving is enabled
    out = None
    if args.save:
        output_path = f"webcam_output_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        print(f"Saving output to {output_path}")
    
    # Process frames
    print("\nStarting inference. Press 'q' to quit.")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame.")
                break
            
            # Run inference
            start_time = time.time()
            results = model(frame, conf=args.conf_thres, imgsz=args.imgsz)
            inference_time = time.time() - start_time
            
            # Visualize results on frame
            annotated_frame = results[0].plot()
            
            # Add FPS info
            fps_text = f"Inference: {1/inference_time:.1f} FPS"
            cv2.putText(annotated_frame, fps_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow("Weapon Detection", annotated_frame)
            
            # Save frame if enabled
            if out is not None:
                out.write(annotated_frame)
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        print("Inference complete.")

if __name__ == "__main__":
    main() 