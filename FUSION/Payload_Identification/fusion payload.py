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

class PayloadDetectionSystem:
    """Payload detection system for harmfulness classification"""

    def __init__(self, payload_model_path, device='cuda'):
        """
        Initialize the payload detection system

        Args:
            payload_model_path (str): Path to the payload classification model (payload.pt)
            device (str): Device to use ('cuda' or 'cpu')
        """
        self.device = device

        # Load payload model
        print(f"🤖 Loading payload model: {payload_model_path}")
        self.payload_model = YOLO(payload_model_path)

        # Payload classification mapping
        self.payload_labels = {0: 'harmful', 1: 'normal'}

        # Track counter
        self.track_counter = defaultdict(int)

        print(f"✅ Model loaded successfully on {device}")

    def classify_payload(self, image_path, conf_threshold=0.25):
        """
        Classify payload harmfulness in an image

        Args:
            image_path (str): Path to the image
            conf_threshold (float): Confidence threshold for classification

        Returns:
            dict: Payload classification results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error loading image: {image_path}")
            return None

        # Get image dimensions
        h, w = image.shape[:2]

        # Run payload classification
        start_time = time.time()
        results = self.payload_model.predict(
            source=image,
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )
        classification_time = time.time() - start_time

        payload_results = {
            'image_path': image_path,
            'image_name': os.path.basename(image_path),
            'image_width': w,
            'image_height': h,
            'classification_time': classification_time,
            'payloads': []
        }

        # Process classification results
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Extract confidence and class
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.payload_labels.get(class_id, 'unknown')

                    # Normalize coordinates
                    x_min_norm = x1 / w
                    y_min_norm = y1 / h
                    x_max_norm = x2 / w
                    y_max_norm = y2 / h

                    # Increment track counter
                    self.track_counter[class_name] += 1
                    track_id = 0

                    # Calculate prob_harmful
                    prob_harmful = confidence if class_id == 0 else (1.0 - confidence)

                    payload_obj = {
                        'track_id': track_id,
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'bbox_norm': [x_min_norm, y_min_norm, x_max_norm, y_max_norm],
                        'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                        'prob_harmful': prob_harmful
                    }

                    payload_results['payloads'].append(payload_obj)

        return payload_results

    def process_single_image(self, image_path, payload_conf=0.25):
        """
        Process a single image for payload classification

        Args:
            image_path (str): Path to the image
            payload_conf (float): Confidence threshold for payload classification

        Returns:
            dict: Complete processing results
        """
        print(f"🔍 Processing image: {os.path.basename(image_path)}")

        # Classify payload
        payload_results = self.classify_payload(image_path, payload_conf)
        if payload_results is None:
            return None

        return payload_results

    def process_image_directory(self, input_dir, output_csv, payload_conf=0.25):
        """
        Process all images in a directory and generate CSV results

        Args:
            input_dir (str): Directory containing images
            output_csv (str): Path to output CSV file
            payload_conf (float): Confidence threshold for payload classification
        """
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_files = []

        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
            image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

        if not image_files:
            print(f"❌ No images found in directory: {input_dir}")
            return

        print(f"📸 Found {len(image_files)} images to process")

        # CSV data storage
        csv_data = []

        # Process each image
        start_time = time.time()
        for i, image_path in enumerate(image_files):
            try:
                results = self.process_single_image(image_path, payload_conf)

                if results:
                    # Convert results to CSV rows
                    frame_name = results['image_name']

                    if results['payloads']:
                        for payload in results['payloads']:
                            csv_row = {
                                'Frame_name': frame_name,
                                'track_id': payload['track_id'],
                                'x_min_norm': round(payload['bbox_norm'][0], 6),
                                'y_min_norm': round(payload['bbox_norm'][1], 6),
                                'x_max_norm': round(payload['bbox_norm'][2], 6),
                                'y_max_norm': round(payload['bbox_norm'][3], 6),
                                'class_label': 0,
                                'direction': 0,  # Set to 0 as requested
                                'confidence_detection': 0,  # Set to 0 since no detection is done
                                'inference_time_detection (ms)': 0,  # Set to 0 since no detection is done
                                'confidence_track': 0,  # Set to 0 since no tracking is done
                                'inference_time_track (ms)': 0,  # Set to 0 since no tracking is done
                                'payload_label': payload['class_name'],
                                'prob_harmful': round(payload['prob_harmful'], 4)
                            }
                            csv_data.append(csv_row)
                    else:
                        # No payloads found
                        csv_row = {
                            'Frame_name': frame_name,
                            'track_id': 0,
                            'x_min_norm': 0.0,
                            'y_min_norm': 0.0,
                            'x_max_norm': 0.0,
                            'y_max_norm': 0.0,
                            'class_label': 'none',
                            'direction': 0,
                            'confidence_detection': 0,
                            'inference_time_detection (ms)': 0,
                            'confidence_track': 0,
                            'inference_time_track (ms)': 0,
                            'payload_label': 'none',
                            'prob_harmful': 0.0
                        }
                        csv_data.append(csv_row)

                # Progress update
                if (i + 1) % 10 == 0:
                    progress = ((i + 1) / len(image_files)) * 100
                    elapsed = time.time() - start_time
                    eta = (elapsed / (i + 1)) * (len(image_files) - i - 1)
                    print(f"⏳ Progress: {progress:.1f}% ({i + 1}/{len(image_files)}) - ETA: {eta:.1f}s")

            except Exception as e:
                print(f"⚠️ Error processing {image_path}: {str(e)}")
                # Add error row
                csv_row = {
                    'Frame_name': os.path.basename(image_path),
                    'track_id': 0,
                    'x_min_norm': 0.0,
                    'y_min_norm': 0.0,
                    'x_max_norm': 0.0,
                    'y_max_norm': 0.0,
                    'class_label': 'error',
                    'direction': 0,
                    'confidence_detection': 0,
                    'inference_time_detection (ms)': 0,
                    'confidence_track': 0,
                    'inference_time_track (ms)': 0,
                    'payload_label': 'error',
                    'prob_harmful': 0.0
                }
                csv_data.append(csv_row)

        # Write CSV file
        self.write_csv(csv_data, output_csv)

        # Print summary
        total_time = time.time() - start_time
        self.print_summary(csv_data, total_time, len(image_files))

    def write_csv(self, csv_data, output_csv):
        """Write CSV data to file"""
        csv_headers = [
            'Frame_name',
            'track_id',
            'x_min_norm',
            'y_min_norm',
            'x_max_norm',
            'y_max_norm',
            'class_label',
            'direction',
            'confidence_detection',
            'inference_time_detection (ms)',
            'confidence_track',
            'inference_time_track (ms)',
            'payload_label',
            'prob_harmful'
        ]

        try:
            with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_headers, delimiter='\t')
                writer.writeheader()
                writer.writerows(csv_data)

            print(f"✅ CSV file written successfully: {output_csv}")

        except Exception as e:
            print(f"❌ Error writing CSV file: {str(e)}")

    def print_summary(self, csv_data, total_time, total_images):
        """Print processing summary"""
        # Calculate statistics
        total_payloads = len([row for row in csv_data if row['class_label'] not in ['none', 'error']])
        harmful_payloads = len([row for row in csv_data if row['payload_label'] == 'harmful'])
        normal_payloads = len([row for row in csv_data if row['payload_label'] == 'normal'])

        # Calculate average prob_harmful
        prob_harmful_values = [row['prob_harmful'] for row in csv_data if row['prob_harmful'] > 0]
        avg_prob_harmful = np.mean(prob_harmful_values) if prob_harmful_values else 0

        print(f"\n📊 Processing Summary:")
        print(f"   📸 Total Images: {total_images}")
        print(f"   ⏱️ Total Processing Time: {total_time:.2f}s")
        print(f"   🚀 Images per Second: {total_images / total_time:.2f}")
        print(f"   📝 CSV Rows: {len(csv_data)}")
        print(f"   🎯 Total Payloads: {total_payloads}")
        print(f"   ☠️ Harmful Payloads: {harmful_payloads}")
        print(f"   ✅ Normal Payloads: {normal_payloads}")
        print(f"   📊 Avg Prob Harmful: {avg_prob_harmful:.3f}")

    def visualize_payloads(self, image_path, output_path=None, payload_conf=0.25):
        """
        Visualize payload detections on an image

        Args:
            image_path (str): Path to input image
            output_path (str): Path to save annotated image (optional)
            payload_conf (float): Confidence threshold for payload classification
        """
        # Process the image
        results = self.process_single_image(image_path, payload_conf)
        if not results:
            return

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return

        # Draw payload detections
        for payload in results['payloads']:
            bbox = payload['bbox']
            class_name = payload['class_name']
            prob_harmful = payload['prob_harmful']

            # Color based on harmfulness
            if class_name == 'harmful':
                color = (0, 0, 255)  # Red for harmful
            elif class_name == 'normal':
                color = (0, 255, 0)  # Green for normal
            else:
                color = (128, 128, 128)  # Gray for unknown

            # Draw bounding box
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # Draw label
            label = f"{class_name} {prob_harmful:.2f}"

            # Add text background
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (bbox[0], bbox[1] - text_height - 10),
                         (bbox[0] + text_width, bbox[1]), color, -1)

            # Add text
            cv2.putText(image, label, (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Save or display
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"✅ Annotated image saved: {output_path}")
        else:
            cv2.imshow('Payload Detections', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def process_single_video_with_fusion(video_path, model_cpu_rgb, model_cpu_ir, model_gpu_rgb, model_gpu_ir, tracking_system):
    """Process a single video with late fusion capability"""
    print(f"🎬 Processing video: {os.path.basename(video_path)}")
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Reset tracking system for new video
    tracking_system.reset()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error opening video: {video_path}")
        return []
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 Video specs: {w}x{h} @ {fps}fps, {total_frames} frames")
    
    csv_data = []
    frame_num = 0
    progress_interval = max(int(fps), 30)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        frame_name = f"{video_name}_{frame_num:06d}"
        
        try:
            if tracking_system.use_late_fusion:
                # Late fusion processing
                cpu_start = time.time()
                
                # Get detections from both RGB and IR models (CPU)
                results_cpu_rgb = model_cpu_rgb.track(
                    source=frame, persist=True, tracker="botsort.yaml",
                    conf=0.25, stream=True, verbose=False, device='cpu'
                )
                results_cpu_ir = model_cpu_ir.track(
                    source=frame, persist=True, tracker="botsort.yaml",
                    conf=0.25, stream=True, verbose=False, device='cpu'
                )
                
                detections_cpu_rgb = list(results_cpu_rgb)
                detections_cpu_ir = list(results_cpu_ir)
                cpu_time = time.time() - cpu_start
                
                # GPU processing
                gpu_start = time.time()
                results_gpu_rgb = model_gpu_rgb.track(
                    source=frame, persist=True, tracker="botsort.yaml",
                    conf=0.25, stream=True, verbose=False, device='cuda'
                )
                results_gpu_ir = model_gpu_ir.track(
                    source=frame, persist=True, tracker="botsort.yaml",
                    conf=0.25, stream=True, verbose=False, device='cuda'
                )
                
                detections_gpu_rgb = list(results_gpu_rgb)
                detections_gpu_ir = list(results_gpu_ir)
                gpu_time = time.time() - gpu_start
                
                # Fuse detections (using GPU results)
                fused_detections = fuse_detections_from_models(
                    detections_gpu_rgb, detections_gpu_ir, iou_thresh=0.5
                )
                
                # Use fused detections for tracking
                current_detections = tracking_system.process_frame(frame, fused_detections, cpu_time, gpu_time)
                
            else:
                # Original single-model processing
                cpu_start = time.time()
                results_cpu = model_cpu_rgb.track(
                    source=frame, persist=True, tracker="botsort.yaml",
                    conf=0.25, stream=True, verbose=False, device='cpu'
                )
                detections_cpu = list(results_cpu)
                cpu_time = time.time() - cpu_start
                
                gpu_start = time.time()
                results_gpu = model_gpu_rgb.track(
                    source=frame, persist=True, tracker="botsort.yaml",
                    conf=0.25, stream=True, verbose=False, device='cuda'
                )
                detections_gpu = list(results_gpu)
                gpu_time = time.time() - gpu_start
                
                current_detections = tracking_system.process_frame(frame, detections_gpu, cpu_time, gpu_time)
            
            # Get current FPS measurements
            fps_data = tracking_system.get_current_fps()
            
            # Track class-specific IDs for this frame
            class_track_ids = defaultdict(int)
            
            # Process each detection for CSV
            for global_track_id, detection_data in current_detections.items():
                bbox = detection_data['bbox']
                class_name = detection_data['class_name']
                class_id = detection_data['class_id']
                detection_confidence = detection_data['confidence']
                
                # Get tracker for additional info
                tracker = tracking_system.trackers.get(global_track_id)
                if not tracker:
                    continue
                
                # Increment class-specific track ID
                class_track_ids[class_name] += 1
                track_id = class_track_ids[class_name]
                
                # Normalize coordinates (0-1 range)
                x_min_norm = bbox[0] / w
                y_min_norm = bbox[1] / h
                x_max_norm = bbox[2] / w
                y_max_norm = bbox[3] / h
                
                # Determine direction (only for drones)
                direction = "unknown"
                if class_name == 'Drone':
                    approach_status = tracker.is_approaching()
                    if approach_status == "APPROACHING":
                        direction = "approaching"
                    elif approach_status == "RECEDING":
                        direction = "receding"
                
                # Get confidence values
                real_detection_confidence = detection_confidence
                real_tracking_confidence = tracker.get_tracking_confidence()
                
                # Create CSV row
                csv_row = {
                    'Frame_name': frame_name,
                    'track_id': track_id,
                    'x_min_norm': round(x_min_norm, 6),
                    'y_min_norm': round(y_min_norm, 6),
                    'x_max_norm': round(x_max_norm, 6),
                    'y_max_norm': round(y_max_norm, 6),
                    'class_label': class_name.lower(),
                    'direction': direction,
                    'confidence_detection': round(real_detection_confidence, 4),
                    'inference_time_detection (ms)': round(gpu_time * 1000, 2),
                    'FPS (CPU)': round(fps_data['cpu_fps'], 2),
                    'FPS (GPU)': round(fps_data['gpu_fps'], 2),
                    'confidence_track': round(real_tracking_confidence, 4),
                    'inference_time_track (ms)': round(cpu_time * 1000 * 0.3, 2),
                    'payload_label': 'unknown',
                    'prob_harmful': 0.0,
                    'fusion_method': 'late_fusion' if tracking_system.use_late_fusion else 'single_model'
                }
                
                csv_data.append(csv_row)
            
            # If no detections in frame, add empty row
            if not current_detections:
                fps_data = tracking_system.get_current_fps()
                csv_row = {
                    'Frame_name': frame_name,
                    'track_id': 0,
                    'x_min_norm': 0.0,
                    'y_min_norm': 0.0,
                    'x_max_norm': 0.0,
                    'y_max_norm': 0.0,
                    'class_label': 'none',
                    'direction': 'unknown',
                    'confidence_detection': 0.0,
                    'inference_time_detection (ms)': round(gpu_time * 1000, 2),
                    'FPS (CPU)': round(fps_data['cpu_fps'], 2),
                    'FPS (GPU)': round(fps_data['gpu_fps'], 2),
                    'confidence_track': 0.0,
                    'inference_time_track (ms)': 0.0,
                    'payload_label': 'none',
                    'prob_harmful': 0.0,
                    'fusion_method': 'late_fusion' if tracking_system.use_late_fusion else 'single_model'
                }
                csv_data.append(csv_row)
                
        except Exception as e:
            print(f"⚠ Error processing frame {frame_num}: {str(e)}")
            fps_data = tracking_system.get_current_fps()
            csv_row = {
                'Frame_name': frame_name,
                'track_id': 0,
                'x_min_norm': 0.0,
                'y_min_norm': 0.0,
                'x_max_norm': 0.0,
                'y_max_norm': 0.0,
                'class_label': 'error',
                'direction': 'unknown',
                'confidence_detection': 0.0,
                'inference_time_detection (ms)': 0.0,
                'FPS (CPU)': round(fps_data['cpu_fps'], 2),
                'FPS (GPU)': round(fps_data['gpu_fps'], 2),
                'confidence_track': 0.0,
                'inference_time_track (ms)': 0.0,
                'payload_label': 'error',
                'prob_harmful': 0.0,
                'fusion_method': 'late_fusion' if tracking_system.use_late_fusion else 'single_model'
            }
            csv_data.append(csv_row)
        
        # Progress reporting
        if frame_num % progress_interval == 0:
            progress = (frame_num / total_frames) * 100
            print(f"⏳ Video Progress: {progress:.1f}% ({frame_num}/{total_frames})")
    
    cap.release()
    return csv_data

def main():
    """Main function to demonstrate the payload detection system"""

    # Configuration
    payload_model_path = '/content/drive/MyDrive/VIP_cup/payload.pt'

    # Input directory containing images
    input_directory = '/content/drive/MyDrive/VIP_cup/Test_Images'

    # Output CSV file
    output_csv = '/content/drive/MyDrive/VIP_cup/payload_detection_results.csv'

    # Payload classification threshold
    payload_conf = 0.25

    # Device to use
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"🚀 Starting Payload Detection System")
    print(f"📁 Input Directory: {input_directory}")
    print(f"💾 Output CSV: {output_csv}")
    print(f"🖥️ Device: {device}")
    print(f"🎯 Payload Confidence: {payload_conf}")

    try:
        # Initialize the detection system
        detector = PayloadDetectionSystem(
            payload_model_path=payload_model_path,
            device=device
        )

        # Process all images in the directory
        detector.process_image_directory(
            input_dir=input_directory,
            output_csv=output_csv,
            payload_conf=payload_conf
        )

        print(f"\n🎉 Payload processing completed successfully!")

    except FileNotFoundError as e:
        print(f"\n❌ File not found error: {str(e)}")
        print("   Please check if the model file and input directory exist")

    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")

    finally:
        # Clean up CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 CUDA cache cleared")


if __name__ == "__main__":
    main()