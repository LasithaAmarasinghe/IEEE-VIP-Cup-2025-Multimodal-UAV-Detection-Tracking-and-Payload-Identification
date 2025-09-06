import torch
import time
import os
import cv2
import numpy as np
import pandas as pd
import csv
import glob
from collections import defaultdict
from ultralytics import YOLO
from torchvision.ops import nms
from pathlib import Path

class VideoPayloadDetector:
    """Enhanced video payload detection system with RGB/IR fusion capabilities"""

    def __init__(self, model_path_rgb, model_path_ir=None, device='cuda'):
        """
        Initialize the video payload detection system

        Args:
            model_path_rgb (str): Path to the RGB payload detection model
            model_path_ir (str): Path to the IR payload detection model (optional)
            device (str): Device to use ('cuda' or 'cpu')
        """
        self.device = device
        self.model_path_rgb = model_path_rgb
        self.model_path_ir = model_path_ir
        self.fusion_enabled = model_path_ir is not None

        # Load RGB payload detection model
        print(f"🤖 Loading RGB payload detection model: {model_path_rgb}")
        self.model_rgb = YOLO(model_path_rgb)

        # Load IR payload detection model if provided
        if self.fusion_enabled:
            print(f"🤖 Loading IR payload detection model: {model_path_ir}")
            self.model_ir = YOLO(model_path_ir)
            print("🔄 Fusion mode enabled - RGB + IR detection")
        else:
            self.model_ir = None
            print("📷 Single modality mode - RGB detection only")

        # Payload classification mapping (0: harmful, 1: normal)
        self.payload_labels = {0: 'harmful', 1: 'normal'}

        # Detection counter for frame processing
        self.frame_count = 0
        self.detection_count = 0

        print(f"✅ Payload detection model(s) loaded successfully on {device}")

    def convert_to_tensor_format(self, detections):
        """Convert YOLO output to tensor format for NMS"""
        boxes = []
        scores = []
        class_ids = []
        
        for det in detections:
            if det.boxes is not None:
                for xyxy, conf, cls in zip(det.boxes.xyxy, det.boxes.conf, det.boxes.cls):
                    boxes.append(xyxy.cpu())
                    scores.append(conf.cpu())
                    class_ids.append(cls.cpu())
        
        if len(boxes) == 0:
            return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,), dtype=torch.int64)
        
        return torch.stack(boxes), torch.tensor(scores), torch.tensor(class_ids)

    def fuse_detections(self, rgb_frame, ir_frame, iou_thresh=0.5, conf_threshold=0.25):
        """
        Fuse RGB and IR detections using NMS

        Args:
            rgb_frame (numpy.ndarray): RGB frame
            ir_frame (numpy.ndarray): IR frame
            iou_thresh (float): IoU threshold for NMS
            conf_threshold (float): Confidence threshold for detection

        Returns:
            list: Fused detection results
        """
        # Run detection on both modalities
        start_time = time.time()
        
        result_rgb = self.model_rgb.predict(
            source=rgb_frame,
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )
        
        result_ir = self.model_ir.predict(
            source=ir_frame,
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )
        
        inference_time = time.time() - start_time

        # Convert to tensor format
        boxes_rgb, scores_rgb, classes_rgb = self.convert_to_tensor_format(result_rgb)
        boxes_ir, scores_ir, classes_ir = self.convert_to_tensor_format(result_ir)

        # Combine detections
        if len(boxes_rgb) == 0 and len(boxes_ir) == 0:
            return [], inference_time
        elif len(boxes_rgb) == 0:
            boxes, scores, classes = boxes_ir, scores_ir, classes_ir
        elif len(boxes_ir) == 0:
            boxes, scores, classes = boxes_rgb, scores_rgb, classes_rgb
        else:
            boxes = torch.cat([boxes_rgb, boxes_ir])
            scores = torch.cat([scores_rgb, scores_ir])
            classes = torch.cat([classes_rgb, classes_ir])

        # Apply NMS to remove duplicate detections
        keep = nms(boxes, scores, iou_threshold=iou_thresh)

        # Extract final detections
        fused_boxes = boxes[keep]
        fused_scores = scores[keep]
        fused_classes = classes[keep]

        # Format results
        detections = []
        h, w = rgb_frame.shape[:2]
        
        for i in range(len(fused_boxes)):
            detection_id = i + 1
            box = fused_boxes[i]
            confidence = float(fused_scores[i])
            class_id = int(fused_classes[i])
            
            # Extract coordinates
            x1, y1, x2, y2 = box.tolist()
            
            # Get payload label
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
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'fusion_mode': 'RGB+IR'
            }
            
            detections.append(detection)

        return detections, inference_time

    def detect_payload_in_frame(self, frame, conf_threshold=0.25):
        """
        Detect payload in a single frame (RGB only)

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
        results = self.model_rgb.predict(
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
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'fusion_mode': 'RGB'
                    }
                    
                    detections.append(detection)

        return detections, inference_time

    def process_video(self, video_path, output_csv, ir_video_path=None, conf_threshold=0.25, iou_thresh=0.5):
        """
        Process video file for payload detection

        Args:
            video_path (str): Path to input RGB video file
            output_csv (str): Path to output CSV file
            ir_video_path (str): Path to input IR video file (optional)
            conf_threshold (float): Confidence threshold for detection
            iou_thresh (float): IoU threshold for NMS in fusion mode
        """
        print(f"🎬 Processing video: {os.path.basename(video_path)}")
        
        # Check if fusion mode is requested
        use_fusion = self.fusion_enabled and ir_video_path is not None
        
        if use_fusion:
            print(f"🔄 Fusion mode: RGB + IR")
            print(f"📷 RGB Video: {os.path.basename(video_path)}")
            print(f"🌡️ IR Video: {os.path.basename(ir_video_path)}")
        else:
            print(f"📷 Single modality mode: RGB only")
        
        # Reset counters
        self.frame_count = 0
        self.detection_count = 0
        
        # Open RGB video
        cap_rgb = cv2.VideoCapture(video_path)
        if not cap_rgb.isOpened():
            print(f"❌ Error opening RGB video: {video_path}")
            return
        
        # Open IR video if fusion mode
        cap_ir = None
        if use_fusion:
            cap_ir = cv2.VideoCapture(ir_video_path)
            if not cap_ir.isOpened():
                print(f"❌ Error opening IR video: {ir_video_path}")
                cap_rgb.release()
                return
        
        # Get video properties
        fps = cap_rgb.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 Video specs: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Prepare CSV data
        csv_data = []
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Progress tracking
        progress_interval = max(int(fps), 30)
        start_time = time.time()
        
        while cap_rgb.isOpened():
            ret_rgb, frame_rgb = cap_rgb.read()
            if not ret_rgb:
                break
            
            # Read IR frame if fusion mode
            frame_ir = None
            if use_fusion:
                ret_ir, frame_ir = cap_ir.read()
                if not ret_ir:
                    break
            
            self.frame_count += 1
            frame_name = f"{video_name}_{self.frame_count:06d}"
            
            try:
                # Detect payload in current frame(s)
                if use_fusion:
                    detections, inference_time = self.fuse_detections(
                        frame_rgb, frame_ir, iou_thresh, conf_threshold
                    )
                else:
                    detections, inference_time = self.detect_payload_in_frame(
                        frame_rgb, conf_threshold
                    )
                
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
                            'inference_time (ms)': round(inference_time * 1000, 2),
                            'fusion_mode': detection['fusion_mode']
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
                        'inference_time (ms)': round(inference_time * 1000, 2),
                        'fusion_mode': 'RGB+IR' if use_fusion else 'RGB'
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
                    'inference_time (ms)': 0.0,
                    'fusion_mode': 'error'
                }
                csv_data.append(csv_row)
            
            # Progress reporting
            if self.frame_count % progress_interval == 0:
                progress = (self.frame_count / total_frames) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / self.frame_count) * (total_frames - self.frame_count)
                print(f"⏳ Progress: {progress:.1f}% ({self.frame_count}/{total_frames}) - ETA: {eta:.1f}s")
        
        cap_rgb.release()
        if cap_ir:
            cap_ir.release()
        
        # Write CSV file
        self.write_csv(csv_data, output_csv)
        
        # Print summary
        total_time = time.time() - start_time
        self.print_summary(csv_data, total_time, video_name)

    def process_image_pairs(self, image_pairs, output_csv, conf_threshold=0.25, iou_thresh=0.5):
        """
        Process paired RGB/IR images for payload detection

        Args:
            image_pairs (list): List of (rgb_path, ir_path) tuples
            output_csv (str): Path to output CSV file
            conf_threshold (float): Confidence threshold for detection
            iou_thresh (float): IoU threshold for NMS
        """
        if not self.fusion_enabled:
            print("❌ Fusion mode not enabled. Please provide IR model path.")
            return
        
        print(f"🖼️ Processing {len(image_pairs)} image pairs")
        
        all_results = []
        
        for i, (rgb_path, ir_path) in enumerate(image_pairs):
            print(f"📷 Processing pair {i+1}/{len(image_pairs)}: {os.path.basename(rgb_path)} + {os.path.basename(ir_path)}")
            
            try:
                # Load images
                rgb_frame = cv2.imread(rgb_path)
                ir_frame = cv2.imread(ir_path)
                
                if rgb_frame is None or ir_frame is None:
                    print(f"⚠️ Error loading images: {rgb_path}, {ir_path}")
                    continue
                
                # Detect payload using fusion
                detections, inference_time = self.fuse_detections(
                    rgb_frame, ir_frame, iou_thresh, conf_threshold
                )
                
                # Add results
                image_id = os.path.splitext(os.path.basename(rgb_path))[0]
                
                if detections:
                    for detection in detections:
                        result = {
                            'image_id': image_id,
                            'detection_id': detection['detection_id'],
                            'x_min_norm': round(detection['x_min_norm'], 6),
                            'y_min_norm': round(detection['y_min_norm'], 6),
                            'x_max_norm': round(detection['x_max_norm'], 6),
                            'y_max_norm': round(detection['y_max_norm'], 6),
                            'payload_label': detection['payload_label'],
                            'confidence': round(detection['confidence'], 4),
                            'prob_harmful': round(detection['prob_harmful'], 4),
                            'inference_time (ms)': round(inference_time * 1000, 2),
                            'fusion_mode': detection['fusion_mode']
                        }
                        all_results.append(result)
                else:
                    # No detections
                    result = {
                        'image_id': image_id,
                        'detection_id': 0,
                        'x_min_norm': 0.0,
                        'y_min_norm': 0.0,
                        'x_max_norm': 0.0,
                        'y_max_norm': 0.0,
                        'payload_label': 'none',
                        'confidence': 0.0,
                        'prob_harmful': 0.0,
                        'inference_time (ms)': round(inference_time * 1000, 2),
                        'fusion_mode': 'RGB+IR'
                    }
                    all_results.append(result)
                    
            except Exception as e:
                print(f"⚠️ Error processing pair {i+1}: {str(e)}")
        
        # Save results to CSV
        if all_results:
            df = pd.DataFrame(all_results)
            df.to_csv(output_csv, index=False)
            print(f"✅ Results saved to: {output_csv}")
            print(f"📝 Total detections: {len(all_results)}")
        else:
            print("❌ No results to save")

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
            'inference_time (ms)',
            'fusion_mode'
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
        
        # Count fusion modes
        fusion_modes = [row['fusion_mode'] for row in csv_data]
        mode_counts = pd.Series(fusion_modes).value_counts()

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
        print(f"   🔄 Detection Modes: {dict(mode_counts)}")

    def visualize_detections(self, video_path, output_path=None, ir_video_path=None, conf_threshold=0.25, iou_thresh=0.5, max_frames=100):
        """
        Visualize payload detections in video (optional, for debugging)

        Args:
            video_path (str): Path to input RGB video
            output_path (str): Path to save annotated video (optional)
            ir_video_path (str): Path to input IR video (optional)
            conf_threshold (float): Confidence threshold for detection
            iou_thresh (float): IoU threshold for NMS
            max_frames (int): Maximum number of frames to process for visualization
        """
        use_fusion = self.fusion_enabled and ir_video_path is not None
        
        cap_rgb = cv2.VideoCapture(video_path)
        if not cap_rgb.isOpened():
            print(f"❌ Error opening RGB video: {video_path}")
            return

        cap_ir = None
        if use_fusion:
            cap_ir = cv2.VideoCapture(ir_video_path)
            if not cap_ir.isOpened():
                print(f"❌ Error opening IR video: {ir_video_path}")
                cap_rgb.release()
                return

        # Get video properties
        fps = int(cap_rgb.get(cv2.CAP_PROP_FPS))
        width = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_num = 0
        while cap_rgb.isOpened() and frame_num < max_frames:
            ret_rgb, frame_rgb = cap_rgb.read()
            if not ret_rgb:
                break

            frame_ir = None
            if use_fusion:
                ret_ir, frame_ir = cap_ir.read()
                if not ret_ir:
                    break

            frame_num += 1
            
            # Detect payload in frame(s)
            if use_fusion:
                detections, _ = self.fuse_detections(frame_rgb, frame_ir, iou_thresh, conf_threshold)
            else:
                detections, _ = self.detect_payload_in_frame(frame_rgb, conf_threshold)

            # Draw detections
            for detection in detections:
                bbox = detection['bbox']
                payload_label = detection['payload_label']
                confidence = detection['confidence']
                prob_harmful = detection['prob_harmful']
                fusion_mode = detection['fusion_mode']

                # Color based on payload type
                if payload_label == 'harmful':
                    color = (0, 0, 255)  # Red for harmful
                elif payload_label == 'normal':
                    color = (0, 255, 0)  # Green for normal
                else:
                    color = (128, 128, 128)  # Gray for unknown

                # Draw bounding box
                cv2.rectangle(frame_rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

                # Draw label
                label = f"{payload_label} {confidence:.2f} (harm: {prob_harmful:.2f}) [{fusion_mode}]"
                
                # Add text background
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame_rgb, (bbox[0], bbox[1] - text_height - 10),
                             (bbox[0] + text_width, bbox[1]), color, -1)

                # Add text
                cv2.putText(frame_rgb, label, (bbox[0], bbox[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Add frame info
            mode_text = f"Frame: {frame_num} | Mode: {'RGB+IR' if use_fusion else 'RGB'}"
            cv2.putText(frame_rgb, mode_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if output_path:
                out.write(frame_rgb)
            else:
                cv2.imshow('Payload Detection', frame_rgb)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap_rgb.release()
        if cap_ir:
            cap_ir.release()
        if output_path:
            out.release()
            print(f"✅ Annotated video saved: {output_path}")
        else:
            cv2.destroyAllWindows()


def main(video_path):
    """Main function to process video for payload detection"""
    
    # Model path - T3_RGB payload detection model
    model_path_rgb = 'yolov8_payload_rgb.pt'
    model_path_ir = 'yolov8_payload_ir.pt'  

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
    print(f"🤖 Model Path (RGB): {model_path_rgb}")
    print(f"🤖 Model Path (IR): {model_path_ir}")
    print(f"💾 Output CSV: {output_csv}")
    print(f"🖥️ Device: {device}")
    print(f"🎯 Confidence Threshold: {conf_threshold}")
    
    # Verify video file exists
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        print("   Please update the video_path variable with the correct path to your video file")
        return
    
    # Verify model file exists
    if not os.path.exists(model_path_rgb):
        print(f"❌ Model file not found: {model_path_rgb}")
        print("   Please check if the model file exists at the specified path")
        return
    if not os.path.exists(model_path_ir):
        print(f"❌ Model file not found: {model_path_ir}")
        print("   Please check if the model file exists at the specified path")
        return

    try:
        # Initialize the payload detector
        detector = VideoPayloadDetector(
            model_path_rgb=model_path_rgb,
            model_path_ir=model_path_ir,
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
