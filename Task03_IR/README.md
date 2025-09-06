# Video Payload Detection System(IR)

## Overview
This project implements a video payload detection system using the YOLOv8 model to classify payloads in video frames as either "harmful" or "normal". The system processes a video file, detects payloads in each frame, and outputs the results to a CSV file with detailed detection information.

## Requirements
- Python 3.8+
- PyTorch
- OpenCV (`cv2`)
- NumPy
- Ultralytics YOLO (`ultralytics`)
- A trained YOLOv8 model file (`yolov8_payload_ir.pt`)

## Installation
1. Install required Python packages:
```
pip install torch opencv-python numpy ultralytics
```

2. Ensure you have a trained YOLOv8 model file (`yolov8_payload_ir.pt`) available.

## Usage
1. Update the `VIDEO_PATH` variable in the script to point to your input video file:
```python
VIDEO_PATH = '/path/to/your/video/file.mp4'
```

2. Run the script:
```bash
python inference_t3_ir.py
```

3. The script will:
   - Load the YOLOv8 model
   - Process the video frame by frame
   - Detect and classify payloads
   - Save results to a CSV file in the `/VIP_cup/payload_detection_results` directory
   - Print a summary of the processing results

## Output
- A CSV file named `<video_name>_payload_detection.csv` containing detection details for each frame, including:
  - Frame name
  - Detection ID
  - Normalized bounding box coordinates
  - Payload label (harmful/normal/none)
  - Confidence score
  - Probability of harmful payload
  - Inference time (ms)

## Notes
- The script uses CUDA if available; otherwise, it falls back to CPU.
- The confidence threshold for detections is set to 0.25 by default.
- Ensure the model file (`yolov8_payload_ir.pt`) and video file exist before running.
- The script includes an optional visualization function (`visualize_detections`) for debugging, which can be enabled by modifying the code.

## Troubleshooting
- If the video or model file is not found, update the paths in the script.
- Clear the CUDA cache if memory issues occur (handled automatically in the script).
- Check the console output for detailed error messages and progress updates.
