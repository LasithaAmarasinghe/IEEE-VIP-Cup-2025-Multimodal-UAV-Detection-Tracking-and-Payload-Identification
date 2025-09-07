import torch
import time
import os
import cv2
import numpy as np
import csv
import glob
from collections import defaultdict
from ultralytics import YOLO
from pathlib import Path

class VideoPayloadDetector:
    """Video payload detection system for frame-by-frame harmful/normal payload classification"""

    def __init__(self, model_path, device='cuda'):
        """
        Initialize the video payload detection system

        Args:
            model_path (str): Path to the payload detection model
            device (str): Device to use ('cuda' or 'cpu')
        """
        self.device = device
        self.model_path = model_path

        # Load payload detection model
        print(f"🤖 Loading payload detection model: {model_path}")
        self.model = YOLO(model_path)

        # Payload classification mapping (0: harmful, 1: normal)
        self.payload_labels = {0: 'harmful', 1: 'normal'}

        # Detection counter for frame processing
        self.frame_count = 0
        self.detection_count = 0

        print(f"✅ Payload detection model loaded successfully on {device}")

    def detect_payload_in_frame(self, frame, conf_threshold=0.25):
        """
        Detect payload in a single frame

        Args:
            frame (numpy.ndarray): Input frame
            conf_threshold (float): Confidence threshold for detection

        Returns:
            list: List of detection results with payload classification
        """
        # Get frame dimensions
        h, w = frame.shape[:2]

        # Run payload detection
        start_time = time.time()
        results = self.model.predict(
            source=frame,
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )
        inference_time = time.time() - start_time

        detections = []

        # Process detection results
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                detection_id = 0
                for box in result.boxes:
                    detection_id += 1
                    
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Extract confidence and class
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Get payload label (0: harmful, 1: normal)
                    payload_label = self.payload_labels.get(class_id, 'unknown')
                    
                    # Calculate probability of harmful payload
                    if class_id == 0:  # harmful
                        prob_harmful = confidence
                    elif class_id == 1:  # normal
                        prob_harmful = 1.0 - confidence
                    else:
                        prob_harmful = 0.0

                    # Normalize coordinates
                    x_min_norm = x1 / w
                    y_min_norm = y1 / h
                    x_max_norm = x2 / w
                    y_max_norm = y2 / h

                    detection = {
                        'detection_id': detection_id,
                        'x_min_norm': x_min_norm,
                        'y_min_norm': y_min_norm,
                        'x_max_norm': x_max_norm,
                        'y_max_norm': y_max_norm,
                        'payload_label': payload_label,
                        'confidence': confidence,
                        'prob_harmful': prob_harmful,
                        'inference_time': inference_time,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    }
                    
                    detections.append(detection)

        return detections, inference_time

    def process_video(self, video_path, output_csv, conf_threshold=0.25):
        """
        Process video file for payload detection

        Args:
            video_path (str): Path to input video file
            output_csv (str): Path to output CSV file
            conf_threshold (float): Confidence threshold for detection
        """
        print(f"🎬 Processing video: {os.path.basename(video_path)}")
        
        # Reset counters
        self.frame_count = 0
        self.detection_count = 0
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error opening video: {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 Video specs: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Prepare CSV data
        csv_data = []
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Progress tracking
        progress_interval = max(int(fps), 30)
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            frame_name = f"{video_name}_{self.frame_count:06d}"
            
            try:
                # Detect payload in current frame
                detections, inference_time = self.detect_payload_in_frame(frame, conf_threshold)
                
                if detections:
                    # Add each detection to CSV data
                    for detection in detections:
                        csv_row = {
                            'Frame_name': frame_name,
                            'detection_id': detection['detection_id'],
                            'x_min_norm': round(detection['x_min_norm'], 6),
                            'y_min_norm': round(detection['y_min_norm'], 6),
                            'x_max_norm': round(detection['x_max_norm'], 6),
                            'y_max_norm': round(detection['y_max_norm'], 6),
                            'payload_label': detection['payload_label'],
                            'confidence': round(detection['confidence'], 4),
                            'prob_harmful': round(detection['prob_harmful'], 4),
                            'inference_time (ms)': round(inference_time * 1000, 2)
                        }
                        csv_data.append(csv_row)
                        self.detection_count += 1
                else:
                    # No detections in this frame
                    csv_row = {
                        'Frame_name': frame_name,
                        'detection_id': 0,
                        'x_min_norm': 0.0,
                        'y_min_norm': 0.0,
                        'x_max_norm': 0.0,
                        'y_max_norm': 0.0,
                        'payload_label': 'none',
                        'confidence': 0.0,
                        'prob_harmful': 0.0,
                        'inference_time (ms)': round(inference_time * 1000, 2)
                    }
                    csv_data.append(csv_row)
                    
            except Exception as e:
                print(f"⚠️ Error processing frame {self.frame_count}: {str(e)}")
                # Add error row
                csv_row = {
                    'Frame_name': frame_name,
                    'detection_id': 0,
                    'x_min_norm': 0.0,
                    'y_min_norm': 0.0,
                    'x_max_norm': 0.0,
                    'y_max_norm': 0.0,
                    'payload_label': 'error',
                    'confidence': 0.0,
                    'prob_harmful': 0.0,
                    'inference_time (ms)': 0.0
                }
                csv_data.append(csv_row)
            
            # Progress reporting
            if self.frame_count % progress_interval == 0:
                progress = (self.frame_count / total_frames) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / self.frame_count) * (total_frames - self.frame_count)
                print(f"⏳ Progress: {progress:.1f}% ({self.frame_count}/{total_frames}) - ETA: {eta:.1f}s")
        
        cap.release()
        
        # Write CSV file
        self.write_csv(csv_data, output_csv)
        
        # Print summary
        total_time = time.time() - start_time
        self.print_summary(csv_data, total_time, video_name)

    def write_csv(self, csv_data, output_csv):
        """Write CSV data to file"""
        csv_headers = [
            'Frame_name',
            'detection_id',
            'x_min_norm',
            'y_min_norm',
            'x_max_norm',
            'y_max_norm',
            'payload_label',
            'confidence',
            'prob_harmful',
            'inference_time (ms)'
        ]

        try:
            with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_headers, delimiter='\t')
                writer.writeheader()
                writer.writerows(csv_data)

            print(f"✅ CSV file written successfully: {output_csv}")
            print(f"📝 Total rows written: {len(csv_data)}")

        except Exception as e:
            print(f"❌ Error writing CSV file: {str(e)}")

    def print_summary(self, csv_data, total_time, video_name):
        """Print processing summary"""
        # Calculate statistics
        total_detections = len([row for row in csv_data if row['payload_label'] not in ['none', 'error']])
        harmful_detections = len([row for row in csv_data if row['payload_label'] == 'harmful'])
        normal_detections = len([row for row in csv_data if row['payload_label'] == 'normal'])
        
        # Calculate average processing times
        inference_times = [row['inference_time (ms)'] for row in csv_data if row['inference_time (ms)'] > 0]
        avg_inference_time = np.mean(inference_times) if inference_times else 0
        
        # Calculate average confidences
        confidences = [row['confidence'] for row in csv_data if row['confidence'] > 0]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # Calculate average prob_harmful
        prob_harmful_values = [row['prob_harmful'] for row in csv_data if row['prob_harmful'] > 0]
        avg_prob_harmful = np.mean(prob_harmful_values) if prob_harmful_values else 0
        
        # Calculate FPS
        fps = self.frame_count / total_time if total_time > 0 else 0

        print(f"\n📊 Processing Summary for {video_name}:")
        print(f"   🎬 Total Frames Processed: {self.frame_count}")
        print(f"   ⏱️ Total Processing Time: {total_time:.2f}s")
        print(f"   🚀 Average FPS: {fps:.2f}")
        print(f"   📝 CSV Rows Generated: {len(csv_data)}")
        print(f"   🎯 Total Payload Detections: {total_detections}")
        print(f"   ☠️ Harmful Payload Detections: {harmful_detections}")
        print(f"   ✅ Normal Payload Detections: {normal_detections}")
        print(f"   ⏱️ Average Inference Time: {avg_inference_time:.2f}ms")
        print(f"   🎯 Average Confidence: {avg_confidence:.3f}")
        print(f"   ☠️ Average Harm Probability: {avg_prob_harmful:.3f}")

    def visualize_detections(self, video_path, output_path=None, conf_threshold=0.25, max_frames=100):
        """
        Visualize payload detections in video (optional, for debugging)

        Args:
            video_path (str): Path to input video
            output_path (str): Path to save annotated video (optional)
            conf_threshold (float): Confidence threshold for detection
            max_frames (int): Maximum number of frames to process for visualization
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error opening video: {video_path}")
            return

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_num = 0
        while cap.isOpened() and frame_num < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            
            # Detect payload in frame
            detections, _ = self.detect_payload_in_frame(frame, conf_threshold)

            # Draw detections
            for detection in detections:
                bbox = detection['bbox']
                payload_label = detection['payload_label']
                confidence = detection['confidence']
                prob_harmful = detection['prob_harmful']

                # Color based on payload type
                if payload_label == 'harmful':
                    color = (0, 0, 255)  # Red for harmful
                elif payload_label == 'normal':
                    color = (0, 255, 0)  # Green for normal
                else:
                    color = (128, 128, 128)  # Gray for unknown

                # Draw bounding box
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

                # Draw label
                label = f"{payload_label} {confidence:.2f} (harm: {prob_harmful:.2f})"
                
                # Add text background
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (bbox[0], bbox[1] - text_height - 10),
                             (bbox[0] + text_width, bbox[1]), color, -1)

                # Add text
                cv2.putText(frame, label, (bbox[0], bbox[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Add frame info
            cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if output_path:
                out.write(frame)
            else:
                cv2.imshow('Payload Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if output_path:
            out.release()
            print(f"✅ Annotated video saved: {output_path}")
        else:
            cv2.destroyAllWindows()


def main(video_path):
    """Main function to process video for payload detection"""
    
    # Model path - T3_IR payload detection model
    model_path = 'yolov8_payload_ir.pt'
    
    # Output directory and CSV file
    output_dir = '/VIP_cup/payload_detection_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output CSV filename based on video name
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_csv = os.path.join(output_dir, f"{video_name}_payload_detection.csv")
    
    # Detection confidence threshold
    conf_threshold = 0.25
    
    # Device to use
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🚀 Starting Video Payload Detection System")
    print(f"🎬 Video Path: {video_path}")
    print(f"🤖 Model Path: {model_path}")
    print(f"💾 Output CSV: {output_csv}")
    print(f"🖥️ Device: {device}")
    print(f"🎯 Confidence Threshold: {conf_threshold}")
    
    # Verify video file exists
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        print("   Please update the video_path variable with the correct path to your video file")
        return
    
    # Verify model file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("   Please check if the model file exists at the specified path")
        return
    
    try:
        # Initialize the payload detector
        detector = VideoPayloadDetector(
            model_path=model_path,
            device=device
        )
        
        # Remove existing CSV file to start fresh
        if os.path.exists(output_csv):
            os.remove(output_csv)
            print(f"🗑️ Removed existing CSV file: {output_csv}")
        
        # Process the video
        print(f"\n🎬 Starting video processing...")
        start_time = time.time()
        
        detector.process_video(
            video_path=video_path,
            output_csv=output_csv,
            conf_threshold=conf_threshold
        )
        
        total_time = time.time() - start_time
        print(f"\n🎉 Video processing completed successfully!")
        print(f"📄 Results saved to: {output_csv}")
        print(f"⏱️ Total processing time: {total_time:.2f}s")
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found error: {str(e)}")
        print("   Please check if the video file and model file exist at the specified paths")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        print("   Please check the error details above and try again")
    
    finally:
        # Clean up CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 CUDA cache cleared")


if __name__ == "__main__":
    # TODO: Update this path to point to your video file
    VIDEO_PATH = '/path/to/your/video/file.mp4'  # ← CHANGE THIS TO YOUR VIDEO PATH
    main(VIDEO_PATH)
