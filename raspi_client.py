#!/usr/bin/env python3
import cv2
import requests
import json
import numpy as np
import time
import pygame
import threading
from io import BytesIO
import argparse

# Initialize pygame for audio
pygame.mixer.init()

# Load alarm sound
ALARM_SOUND_PATH = "/home/barvaz/Desktop/alarm.mp3"
alarm_sound = pygame.mixer.Sound(ALARM_SOUND_PATH)

# Global variables
alarm_playing = False
alarm_lock = threading.Lock()

def play_alarm():
    global alarm_playing
    with alarm_lock:
        if not alarm_playing:
            alarm_playing = True
            alarm_sound.play(-1)  # Play in loop

def stop_alarm():
    global alarm_playing
    with alarm_lock:
        if alarm_playing:
            alarm_sound.stop()
            alarm_playing = False

def process_frame(frame, server_url):
    # Send frame to server and process the response
    # Encode frame as JPEG
    _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    # Send to server
    try:
        files = {'image': BytesIO(img_encoded.tobytes())}
        response = requests.post(f"{server_url}/infer", files=files, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            
            # Check if we should alert
            if result.get('alert', False):
                play_alarm()
            else:
                stop_alarm()
            
            # Draw bounding boxes on the frame
            for detection in result.get('detections', []):
                box = detection['box']
                label = f"{detection['class']} {detection['confidence']:.2f}"
                
                # Draw rectangle
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                
                # Draw label background
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (box[0], box[1] - text_size[1] - 5), 
                              (box[0] + text_size[0], box[1]), (0, 255, 0), -1)
                
                # Draw text
                cv2.putText(frame, label, (box[0], box[1] - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Add server processing time
            process_time = result.get('process_time', 0)
            cv2.putText(frame, f"Server: {process_time*1000:.0f}ms", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        else:
            print(f"Error: Server returned status code {response.status_code}")
            cv2.putText(frame, "Server Error", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")
        cv2.putText(frame, "Connection Error", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame

def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi client for weapon detection")
    parser.add_argument("--server", type=str, default="http://34.0.85.5:8000",
                       help="Server URL (default: http://34.0.85.5:8000)")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=640,
                       help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480,
                       help="Frame height (default: 480)")
    parser.add_argument("--fps", type=int, default=15,
                       help="Target FPS (default: 15)")
    args = parser.parse_args()
    
    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print(f"Connected to camera. Sending frames to {args.server}")
    
    # Set target time between frames based on FPS
    frame_delay = 1.0 / args.fps
    
    try:
        # Check server status
        try:
            response = requests.get(f"{args.server}/status", timeout=5)
            if response.status_code == 200:
                print(f"Server status: {response.json()}")
            else:
                print(f"Warning: Server status check failed with code {response.status_code}")
        except requests.exceptions.RequestException:
            print("Warning: Could not connect to server for status check")
    
        while True:
            start_time = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Process frame (send to server and draw results)
            processed_frame = process_frame(frame.copy(), args.server)
            
            # Display frame
            cv2.imshow('Weapon Detection', processed_frame)
            
            # Calculate time to wait to maintain target FPS
            elapsed = time.time() - start_time
            wait_time = max(1, int((frame_delay - elapsed) * 1000))
            
            # Exit on 'q' key press
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break
    
    finally:
        # Clean up
        stop_alarm()
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

if __name__ == "__main__":
    main()
