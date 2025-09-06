# Drone Tracking System with Late Fusion README

## Overview
This project implements an optimized drone tracking system using YOLOv8 for object detection, supporting both RGB and IR video inputs with late fusion capabilities. It tracks drones and birds, fuses detections from multiple models, and generates detailed CSV output.

## Requirements
- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- Ultralytics YOLOv8
- Torchvision
- CUDA-enabled GPU (optional, for faster processing)

## Setup
1. Install dependencies:
```
pip install torch opencv-python numpy ultralytics torchvision
```

2. Ensure you have trained YOLOv8 model files:
   - `yolov8_rgb.pt` for RGB data
   - `yolov8_ir.pt` for IR data

3. Update the video path in the script:
   - Modify `video_path` in the `if __name__ == "__main__":` block to point to your video file.

## Usage
Run the script with:
```
python inference_fusion.py
```

### Configuration
- **Video Path**: Set `video_path` to your input video (e.g., `/path/to/your/video/file.mp4`).
- **Model Paths**: Ensure `model_path_rgb` (`T1_T2_RGB/yolov8_rgb.pt`) and `model_path_ir` (`T1_T2_IR/yolov8_ir.pt`) point to your model files.
- **Output Directory**: Set `output_dir` to the desired CSV output location (e.g., `/content/drive/MyDrive/VIP_cup/tracked_videos`).
- **Late Fusion**: Enable by setting `use_late_fusion=True` in `OptimizedDroneTrackingSystem` initialization.

## Output
The script generates a CSV file in the output directory with the following columns:
- Frame_name: Unique frame identifier
- track_id: Class-specific track ID
- x_min_norm, y_min_norm, x_max_norm, y_max_norm: Normalized bounding box coordinates
- class_label: Object class (drone, bird, none, or error)
- direction: Movement direction for drones (approaching, receding, or unknown)
- confidence_detection: Detection confidence score
- inference_time_detection (ms): Detection processing time
- FPS (CPU/GPU): Frames per second for CPU/GPU
- confidence_track: Tracking confidence score
- inference_time_track (ms): Tracking processing time
- payload_label: Payload status (currently 'unknown')
- prob_harmful: Probability of harmful payload (currently 0.0)
- fusion_method: Detection method used (late_fusion or single_model)

## Notes
- The system supports late fusion of RGB and IR detections using Non-Maximum Suppression (NMS).
- Both CPU and GPU processing are utilized (GPU is optional).
- Ensure sufficient disk space for output CSV files.
- The script clears CUDA cache automatically after processing.
- Check console output for error details if issues occur.
