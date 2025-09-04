#!/usr/bin/env python3
import cv2
import time
import numpy as np
import argparse
import os
from pathlib import Path

# TensorRT and CUDA imports - these will be available on Jetson
try:
    import tensorrt as trt  # type: ignore
    import pycuda.driver as cuda  # type: ignore
    import pycuda.autoinit  # type: ignore
    TENSORRT_AVAILABLE = True
    print("TensorRT is available")
except ImportError:
    TENSORRT_AVAILABLE = False
    print("TensorRT is not available, falling back to ONNX or TFLite")

# TFLite import
try:
    import tensorflow as tf  # type: ignore
    import tflite_runtime.interpreter as tflite  # type: ignore
    TFLITE_AVAILABLE = True
    print("TFLite is available")
except ImportError:
    TFLITE_AVAILABLE = False
    print("TFLite is not available")

# ONNX Runtime import
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    print("ONNX Runtime is available")
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX Runtime is not available")

class YOLOInference:
    def __init__(self, model_path, imgsz=256, conf_thres=0.25, iou_thres=0.45):
        """
        Initialize the YOLO inference engine
        
        Args:
            model_path: Path to the model file (.engine, .onnx, or .tflite)
            imgsz: Input image size
            conf_thres: Confidence threshold
            iou_thres: IoU threshold for NMS
        """
        self.imgsz = (imgsz, imgsz)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Determine model type from extension
        model_ext = Path(model_path).suffix.lower()
        
        if model_ext == '.engine' and TENSORRT_AVAILABLE:
            print(f"Loading TensorRT engine: {model_path}")
            self.model_type = 'tensorrt'
            self._load_tensorrt(model_path)
        elif model_ext == '.tflite' and TFLITE_AVAILABLE:
            print(f"Loading TFLite model: {model_path}")
            self.model_type = 'tflite'
            self._load_tflite(model_path)
        elif model_ext == '.onnx' and ONNX_AVAILABLE:
            print(f"Loading ONNX model: {model_path}")
            self.model_type = 'onnx'
            self._load_onnx(model_path)
        else:
            raise ValueError(f"Unsupported model format: {model_ext} or required runtime not available")
        
        # Load class names if available
        self.class_names = ['object']  # Default
        try:
            yaml_path = str(Path(model_path).parent.parent.parent / 'data.yaml')
            if os.path.exists(yaml_path):
                import yaml
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    if 'names' in data:
                        self.class_names = data['names']
                        print(f"Loaded {len(self.class_names)} class names")
        except Exception as e:
            print(f"Warning: Could not load class names: {e}")
    
    def _load_tensorrt(self, model_path):
        """Load TensorRT engine"""
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        with open(model_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Allocate memory for inputs and outputs
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for binding in range(self.engine.num_bindings):
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def _load_tflite(self, model_path):
        """Load TFLite model"""
        # Use TFLite runtime if available, otherwise use TensorFlow
        if hasattr(tf, 'lite'):
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
        else:
            self.interpreter = tflite.Interpreter(model_path=model_path)
        
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input shape
        self.tflite_input_shape = self.input_details[0]['shape']
    
    def _load_onnx(self, model_path):
        """Load ONNX model"""
        # Use CUDA execution provider if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Get metadata
        self.input_shape = self.session.get_inputs()[0].shape
        self.onnx_input_width = self.input_shape[2]
        self.onnx_input_height = self.input_shape[3]
    
    def preprocess(self, img):
        """Preprocess the image for inference"""
        # Resize
        img = cv2.resize(img, self.imgsz)
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize and transpose
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def infer(self, img):
        """Run inference on the image"""
        # Preprocess image
        input_data = self.preprocess(img)
        
        start_time = time.time()
        
        if self.model_type == 'tensorrt':
            # Copy input data to input memory
            np.copyto(self.inputs[0]['host'], input_data.ravel())
            
            # Transfer input data to device
            cuda.memcpy_htod(self.inputs[0]['device'], self.inputs[0]['host'])
            
            # Run inference
            self.context.execute_v2(self.bindings)
            
            # Transfer predictions back from device
            for out in self.outputs:
                cuda.memcpy_dtoh(out['host'], out['device'])
            
            # Process output
            output = self.outputs[0]['host']
            
            # Reshape output to [num_boxes, num_classes + 5]
            output = output.reshape((-1, len(self.class_names) + 5))
            
        elif self.model_type == 'tflite':
            # Prepare input
            if self.tflite_input_shape[3] == 3:  # NHWC format
                input_data = np.transpose(input_data, (0, 2, 3, 1))  # NCHW to NHWC
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data.astype(np.float32))
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output tensor
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            
        elif self.model_type == 'onnx':
            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: input_data.astype(np.float32)})
            
            # Get output
            output = outputs[0]
        
        inference_time = time.time() - start_time
        
        # Process detections
        return self.process_output(output, img.shape), inference_time
    
    def process_output(self, output, img_shape):
        """Process the raw output to get detections"""
        # Filter by confidence threshold
        conf_mask = output[:, 4] >= self.conf_thres
        output = output[conf_mask]
        
        # If no detections, return empty list
        if len(output) == 0:
            return []
        
        # Get boxes, scores, and class ids
        boxes = output[:, :4]  # x1, y1, x2, y2 or x, y, w, h
        scores = output[:, 4]
        class_ids = output[:, 5:].argmax(1)
        
        # Convert boxes to x1, y1, x2, y2 format if in xywh format
        if boxes.shape[1] == 4 and boxes[0, 2] < 1 and boxes[0, 3] < 1:
            # xywh format, convert to xyxy
            x = boxes[:, 0]
            y = boxes[:, 1]
            w = boxes[:, 2]
            h = boxes[:, 3]
            
            x1 = x - w/2
            y1 = y - h/2
            x2 = x + w/2
            y2 = y + h/2
            
            boxes = np.stack((x1, y1, x2, y2), axis=1)
        
        # Scale boxes to original image size
        img_h, img_w = img_shape[:2]
        scale_x = img_w / self.imgsz[0]
        scale_y = img_h / self.imgsz[1]
        
        boxes[:, 0] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 2] *= scale_x
        boxes[:, 3] *= scale_y
        
        # Apply NMS (Non-Maximum Suppression)
        keep_indices = self._nms(boxes, scores, self.iou_thres)
        
        detections = []
        for i in keep_indices:
            detections.append({
                'box': boxes[i].astype(int).tolist(),
                'score': float(scores[i]),
                'class_id': int(class_ids[i]),
                'class_name': self.class_names[int(class_ids[i])]
            })
        
        return detections
    
    def _nms(self, boxes, scores, iou_threshold):
        """Simple implementation of Non-Maximum Suppression"""
        # Sort by score
        indices = np.argsort(-scores)
        
        # Apply NMS
        keep = []
        while indices.size > 0:
            # Keep the index with highest score
            i = indices[0]
            keep.append(i)
            
            # Get IoU of the current box with the rest
            iou = self._box_iou(boxes[i:i+1], boxes[indices[1:]])
            
            # Remove boxes with IoU > threshold
            mask = iou[0] <= iou_threshold
            indices = indices[1:][mask]
        
        return keep
    
    def _box_iou(self, box1, box2):
        """Calculate IoU between boxes"""
        # Calculate intersection area
        x1 = np.maximum(box1[:, 0, None], box2[:, 0])
        y1 = np.maximum(box1[:, 1, None], box2[:, 1])
        x2 = np.minimum(box1[:, 2, None], box2[:, 2])
        y2 = np.minimum(box1[:, 3, None], box2[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate union area
        box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        
        union = box1_area[:, None] + box2_area - intersection
        
        # Calculate IoU
        iou = intersection / union
        
        return iou
    
    def draw_detections(self, img, detections):
        """Draw detections on the image"""
        for det in detections:
            box = det['box']
            score = det['score']
            class_name = det['class_name']
            
            # Draw box
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {score:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (box[0], box[1] - label_height - baseline), 
                         (box[0] + label_width, box[1]), (0, 255, 0), -1)
            cv2.putText(img, label, (box[0], box[1] - baseline), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return img

def main():
    parser = argparse.ArgumentParser(description="YOLO inference on Jetson Nano")
    parser.add_argument("--model", type=str, required=True, help="Path to model file (.engine, .onnx, or .tflite)")
    parser.add_argument("--source", type=str, default="0", help="Source (0 for webcam, or video file path)")
    parser.add_argument("--imgsz", type=int, default=256, help="Input image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--show", action="store_true", help="Show detection results")
    parser.add_argument("--save", action="store_true", help="Save detection results")
    parser.add_argument("--skip", type=int, default=0, help="Skip frames for higher FPS (0 = no skip)")
    
    args = parser.parse_args()
    
    # Initialize model
    model = YOLOInference(args.model, args.imgsz, args.conf, args.iou)
    
    # Open video source
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
        source_name = f"camera_{args.source}"
    else:
        cap = cv2.VideoCapture(args.source)
        source_name = Path(args.source).stem
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create video writer if saving
    if args.save:
        output_path = f"output_{source_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Performance metrics
    frame_count = 0
    total_fps = 0
    skip_count = 0
    
    print(f"Starting inference on {args.source} with model {args.model}")
    print(f"Press 'q' to quit")
    
    # Main loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames if needed
        if args.skip > 0:
            skip_count += 1
            if skip_count % (args.skip + 1) != 0:
                continue
        
        # Run inference
        detections, inference_time = model.infer(frame)
        
        # Calculate FPS
        fps = 1 / inference_time
        total_fps += fps
        frame_count += 1
        
        # Draw detections
        frame = model.draw_detections(frame, detections)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        if args.show:
            cv2.imshow("Detection", frame)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Save frame
        if args.save:
            out.write(frame)
    
    # Release resources
    cap.release()
    if args.save:
        out.release()
    cv2.destroyAllWindows()
    
    # Print performance summary
    if frame_count > 0:
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Processed {frame_count} frames")
        if args.save:
            print(f"Output saved to {output_path}")

if __name__ == "__main__":
    main() 