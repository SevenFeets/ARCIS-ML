import cv2
import os
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test YOLO model on test images')
    parser.add_argument('--model', type=str, default='runs/detect/train/weights/best.pt',
                        help='Path to YOLO model file (.pt)')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='Confidence threshold for detections')
    parser.add_argument('--test-dir', type=str, default='weapon_detection/test/images',
                        help='Directory containing test images')
    parser.add_argument('--save-dir', type=str, default='test_results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to run inference on (cuda device or cpu)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Create save directory if it doesn't exist
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Load the model
    print(f"Loading model from {args.model}...")
    model = YOLO(args.model)
    
    # Get list of test images
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        print(f"Error: Test directory {test_dir} does not exist")
        return
    
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"No image files found in {test_dir}")
        return
    
    print(f"Found {len(image_files)} test images")
    
    # Process each image
    total_time = 0
    for i, img_file in enumerate(image_files):
        img_path = test_dir / img_file
        print(f"Processing {i+1}/{len(image_files)}: {img_path}")

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Error: Could not read image {img_path}")
            continue
        
        # Run inference
        start_time = time.time()
        results = model(img, conf=args.conf_thres)
        inference_time = time.time() - start_time
        total_time += inference_time
        
        # Get results
        result = results[0]
        detections = []
        
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            cls_name = result.names[cls]
            
            detections.append({
                'class': cls_name,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2]
            })
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show detections in console
        print(f"  Found {len(detections)} detections")
        for det in detections:
            print(f"  - {det['class']}: {det['confidence']:.2f}")
        
        # Add inference time to image
        cv2.putText(img, f"Inference: {inference_time:.4f}s", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Save result
        output_path = save_dir / f"result_{img_file}"
        cv2.imwrite(str(output_path), img)
        print(f"  Saved result to {output_path}")
        
        # Display for 1 second (press any key to move to next image)
        cv2.imshow("Detection Result", img)
        key = cv2.waitKey(1000) & 0xFF
        if key == 27:  # ESC key
            break
    
    # Print summary
    avg_time = total_time / len(image_files) if image_files else 0
    print("\nTesting complete!")
    print(f"Processed {len(image_files)} images")
    print(f"Average inference time: {avg_time:.4f}s ({1/avg_time:.2f} FPS)")
    print(f"Results saved to {save_dir}")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 