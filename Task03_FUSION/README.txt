# Video Payload Detection System README

## Overview
This project implements a video payload detection system using YOLOv8 for detecting and classifying payloads (harmful or normal) in RGB and optionally IR video inputs. It supports late fusion of RGB and IR detections and generates detailed CSV output.

## Requirements
- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- Pandas
- Ultralytics YOLOv8
- Torchvision
- CUDA-enabled GPU (optional, for faster processing)

## Setup
1. Install dependencies:
```
pip install torch opencv-python numpy pandas ultralytics torchvision
```

2. Ensure you have trained YOLOv8 model files:
   - `yolov8_payload_rgb.pt` for RGB payload detection
   - `yolov8_payload_ir.pt` for IR payload detection (optional)

3. Update the video path in the script:
   - Modify `VIDEO_PATH` in the `if __name__ == "__main__":` block to point to your video file.

## Usage
Run the script with:
```
python inference_t3_fusion.py
```

### Configuration
- **Video Path**: Set `VIDEO_PATH` to your input video (e.g., `/path/to/your/video/file.mp4`).
- **Model Paths**: Ensure `model_path_rgb` (`yolov8_payload_rgb.pt`) and `model_path_ir` (`yolov8_payload_ir.pt`) point to your model files.
- **Output Directory**: Set `output_dir` to the desired CSV output location (e.g., `/VIP_cup/payload_detection_results`).
- **Confidence Threshold**: Set `conf_threshold` (default: 0.25) for detection filtering.
- **Fusion Mode**: Automatically enabled if IR model path is provided and IR video is available.

## Output
The script generates a CSV file in the output directory with the following columns:
- Frame_name: Unique frame identifier
- detection_id: Detection ID within the frame
- x_min_norm, y_min_norm, x_max_norm, y_max_norm: Normalized bounding box coordinates
- payload_label: Payload classification (harmful, normal, none, or error)
- confidence: Detection confidence score
- prob_harmful: Probability of harmful payload
- inference_time (ms): Detection processing time
- fusion_mode: Detection mode (RGB or RGB+IR)

## Notes
- The system supports late fusion of RGB and IR detections using Non-Maximum Suppression (NMS).
- GPU processing is used if available; otherwise, CPU is used.
- Ensure sufficient disk space for output CSV files.
- The script clears CUDA cache automatically after processing.
- Check console output for error details if issues occur.
