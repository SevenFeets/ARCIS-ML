tflite_info.py:
Purpose: Analyzes and displays detailed information about a TensorFlow Lite model
Key functions:
Shows model size, input/output details, tensor information
Tests the model with dummy input to verify it works
Reports quantization settings if present
Usage: python tflite_info.py --model path/to/model.tflite [--quantized]



inference_tflite.py:
Purpose: Performs object detection using a TensorFlow Lite model
Key functions:
Loads and processes images to the required format (320×320)
Runs inference using the TFLite interpreter
Extracts bounding boxes and class probabilities
Filters results based on confidence threshold
Draws detection boxes on images and saves/displays results
Output: Annotated image with weapon detections (saved as detection_result_tflite.jpg)
Usage: python inference_tflite.py --image test.jpg --model model.tflite --threshold 0.3



test_inference.py:
Purpose: Similar to inference_tflite.py but uses the full Keras model (.h5) instead
Key differences:
Uses tf.keras.models.load_model instead of TFLite interpreter
Designed for desktop/server use rather than edge devices
Otherwise has similar image processing and output visualization functionality
Output: Annotated image with weapon detections (saved as detection_result.jpg)
Usage: python test_inference.py --image test.jpg --model model.h5 --threshold 0.3