# Drone Tracking System README(IR)

## Overview
This project provides an optimized drone tracking system using YOLOv8 for object detection and a custom tracking algorithm to monitor drones and birds in video footage. The system processes videos, tracks objects, and generates CSV output with detailed tracking information.

## Requirements
- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- Ultralytics YOLOv8
- CUDA-enabled GPU (optional, for faster processing)

## Setup
1. Install dependencies:
```
pip install torch opencv-python numpy ultralytics
```

2. Ensure you have a trained YOLOv8 model file (`yolov8_ir.pt`).

3. Update the video path in the script:
   - Modify `video_path` in the `if __name__ == "__main__":` block to point to your video file.

## Usage
Run the script with:
```
python inference_ir.py
```

### Configuration
- **Video Path**: Set `video_path` to the location of your input video (e.g., `/path/to/your/video/file.mp4`).
- **Model Path**: Ensure `model_path` points to your YOLOv8 model file (`yolov8_rgb.pt`).
- **Output Directory**: Set `output_dir` to the desired location for CSV output (e.g., `/VIP_cup/tracked_videos`).

### Output
The script generates a CSV file in the output directory with the following columns:
- Frame_name: Unique frame identifier
- track_id: Class-specific track ID
- x_min_norm, y_min_norm, x_max_norm, y_max_norm: Normalized bounding box coordinates
- class_label: Object class (drone, bird, none, or error)
- direction: Movement direction for drones (approaching, receding, or unknown)
- confidence_detection: Detection confidence score
- inference_time_detection (ms): Detection processing time
- FPS (CPU/GPU): Frames per second for CPU/GPU processing
- confidence_track: Tracking confidence score
- inference_time_track (ms): Tracking processing time
- payload_label: Payload status (currently 'unknown')
- prob_harmful: Probability of harmful payload (currently 0.0)

## Notes
- The system uses both CPU and GPU for processing (GPU is optional).
- Ensure sufficient disk space for output CSV files.
- The script automatically clears CUDA cache after processing.
- For issues, check the console output for error details.
