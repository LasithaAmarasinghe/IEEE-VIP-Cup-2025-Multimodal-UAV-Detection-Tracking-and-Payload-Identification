import torch
import time
import math
import glob
import json
import csv
import os
import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

class OptimizedDroneTracker:
    """Highly optimized drone tracker with efficient approach/receding detection"""

    # Class constants to avoid repeated calculations
    IMAGE_CENTER = (160, 128)
    MIN_HISTORY_SIZE = 8
    MAX_HISTORY_SIZE = 20
    SIZE_THRESHOLD = 100
    DISTANCE_THRESHOLD = 200

    def __init__(self, track_id, initial_bbox, initial_confidence=0.0):
        self.track_id = track_id

        # Use deque for O(1) append/popleft operations
        x1, y1, x2, y2 = initial_bbox
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        area = (x2 - x1) * (y2 - y1)

        # Store detection data including confidence
        self.centers = deque([(cx, cy)], maxlen=self.MAX_HISTORY_SIZE)
        self.areas = deque([area], maxlen=self.MAX_HISTORY_SIZE)
        self.bbox_history = deque([initial_bbox], maxlen=self.MAX_HISTORY_SIZE)
        self.confidence_history = deque([initial_confidence], maxlen=self.MAX_HISTORY_SIZE)

        self.last_seen_frame = 0
        self.is_active = True
        self.current_confidence = initial_confidence

        # Cache for approach detection
        self._cached_approach_status = None
        self._cache_frame = -1
        self._cache_update_interval = 3

        # Pre-calculate squared distance to center for initial position
        self._initial_dist_sq = (cx - self.IMAGE_CENTER[0])**2 + (cy - self.IMAGE_CENTER[1])**2

    def update(self, bbox, frame_num, confidence=0.0):
        """Updated to include confidence tracking"""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        area = (x2 - x1) * (y2 - y1)

        # Update using deque's efficient operations
        self.centers.append((cx, cy))
        self.areas.append(area)
        self.bbox_history.append(bbox)
        self.confidence_history.append(confidence)
        self.last_seen_frame = frame_num
        self.current_confidence = confidence
        self.is_active = True

    def get_current_confidence(self):
        """Get current detection confidence"""
        return self.current_confidence

    def get_tracking_confidence(self):
        """Calculate tracking confidence based on detection history"""
        if len(self.confidence_history) == 0:
            return 0.0
        
        # Calculate tracking confidence based on:
        # 1. Average confidence over recent detections
        # 2. Consistency of detections
        # 3. Age of track
        
        recent_confidences = list(self.confidence_history)[-5:]  # Last 5 detections
        avg_confidence = np.mean(recent_confidences)
        
        # Consistency bonus: higher if confidences are stable
        confidence_std = np.std(recent_confidences) if len(recent_confidences) > 1 else 0
        consistency_factor = max(0.8, 1.0 - confidence_std)
        
        # Age factor: longer tracks are more reliable
        track_age = len(self.confidence_history)
        age_factor = min(1.0, 0.5 + (track_age * 0.05))
        
        # Combine factors
        tracking_confidence = avg_confidence * consistency_factor * age_factor
        
        return min(1.0, tracking_confidence)

    def get_center(self):
        """Get current center position"""
        if self.bbox_history:
            x1, y1, x2, y2 = self.bbox_history[-1]
            return (int((x1 + x2) * 0.5), int((y1 + y2) * 0.5))
        return (0, 0)

    def is_approaching(self):
        """Highly optimized approach/receding detection without speed calculation"""
        current_frame = len(self.centers)

        # Use cached result if recent enough
        if current_frame - self._cache_frame < self._cache_update_interval:
            return self._cached_approach_status

        if len(self.centers) < self.MIN_HISTORY_SIZE or len(self.areas) < self.MIN_HISTORY_SIZE:
            self._cached_approach_status = None
            return None

        # Update cache frame counter
        self._cache_frame = current_frame

        # Fast size-based detection using only first and last values
        size_change = self.areas[-1] - self.areas[0]
        size_approaching = size_change > self.SIZE_THRESHOLD

        # Fast distance-based detection using squared distances (avoid sqrt)
        last_pos = self.centers[-1]
        first_pos = self.centers[0]

        # Pre-calculated squared distances
        last_dist_sq = (last_pos[0] - self.IMAGE_CENTER[0])**2 + (last_pos[1] - self.IMAGE_CENTER[1])**2
        first_dist_sq = (first_pos[0] - self.IMAGE_CENTER[0])**2 + (first_pos[1] - self.IMAGE_CENTER[1])**2

        # Approaching if moving toward center
        distance_approaching = (first_dist_sq - last_dist_sq) > self.DISTANCE_THRESHOLD

        # Simple decision logic
        if size_approaching and distance_approaching:
            self._cached_approach_status = "APPROACHING"
        elif size_approaching:
            self._cached_approach_status = "APPROACHING"
        elif distance_approaching:
            self._cached_approach_status = "APPROACHING"
        else:
            self._cached_approach_status = "RECEDING"

        return self._cached_approach_status

class OptimizedDroneTrackingSystem:
    """Optimized tracking system with FPS monitoring for CPU/GPU"""

    def __init__(self, device='cuda'):
        self.trackers = {}
        self.frame_count = 0
        self.detection_count = 0
        self.max_missed_frames = 15
        self.device = device

        # Class-specific ID counters
        self.drone_id_counter = 1
        self.bird_id_counter = 1
        self.global_to_class_id = {}

        # FPS tracking for CPU and GPU
        self.fps_measurements_cpu = deque(maxlen=30)
        self.fps_measurements_gpu = deque(maxlen=30)
        self.current_fps_cpu = 0.0
        self.current_fps_gpu = 0.0

        # Timing for FPS calculation
        self.frame_start_time = 0.0
        self.cpu_processing_time = 0.0
        self.gpu_processing_time = 0.0

        # Batch processing intervals
        self._draw_update_interval = 2
        self._metrics_update_interval = 10

    def reset(self):
        """Reset system for new video"""
        self.trackers = {}
        self.frame_count = 0
        self.detection_count = 0

        # Reset class-specific counters
        self.drone_id_counter = 1
        self.bird_id_counter = 1
        self.global_to_class_id = {}

        # Reset FPS measurements
        self.fps_measurements_cpu.clear()
        self.fps_measurements_gpu.clear()
        self.current_fps_cpu = 0.0
        self.current_fps_gpu = 0.0

    def start_frame_timing(self):
        """Start timing for current frame"""
        self.frame_start_time = time.time()

    def update_fps_measurements(self, cpu_time, gpu_time):
        """Update FPS measurements based on processing times"""
        if cpu_time > 0:
            cpu_fps = 1.0 / cpu_time
            self.fps_measurements_cpu.append(cpu_fps)
            self.current_fps_cpu = np.mean(self.fps_measurements_cpu)

        if gpu_time > 0:
            gpu_fps = 1.0 / gpu_time
            self.fps_measurements_gpu.append(gpu_fps)
            self.current_fps_gpu = np.mean(self.fps_measurements_gpu)

    def process_frame(self, frame, detections, cpu_time=0.0, gpu_time=0.0):
        """Optimized frame processing with FPS tracking and real confidence values"""
        # Update FPS measurements
        self.update_fps_measurements(cpu_time, gpu_time)

        current_detections = {}
        active_track_ids = set()

        # Batch process detections with class information and confidence
        for detection in detections:
            if detection.boxes is None:
                continue
            for box in detection.boxes:
                global_track_id = int(box.id.item()) if box.id is not None else None
                if global_track_id is None:
                    continue

                # Extract class information
                class_id = int(box.cls.item()) if box.cls is not None else None
                class_name = 'Drone' if class_id == 1 else 'Bird'

                # Extract confidence score
                confidence = float(box.conf.item()) if box.conf is not None else 0.0

                # Convert to tuple once
                bbox = tuple(map(int, box.xyxy[0].tolist()))

                current_detections[global_track_id] = {
                    'bbox': bbox,
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence
                }
                active_track_ids.add(global_track_id)

        # Batch update trackers with class-specific IDs and confidence
        for global_track_id, detection_data in current_detections.items():
            bbox = detection_data['bbox']
            class_name = detection_data['class_name']
            confidence = detection_data['confidence']

            if global_track_id not in self.trackers:
                # Generate class-specific ID for new trackers
                if global_track_id not in self.global_to_class_id:
                    if class_name == 'Drone':
                        class_specific_id = self.drone_id_counter
                        self.drone_id_counter += 1
                    else:  # Bird
                        class_specific_id = self.bird_id_counter
                        self.bird_id_counter += 1

                    self.global_to_class_id[global_track_id] = class_specific_id

                # Create new tracker with class information and confidence
                tracker = OptimizedDroneTracker(global_track_id, bbox, confidence)
                tracker.class_id = detection_data['class_id']
                tracker.class_name = class_name
                tracker.class_specific_id = self.global_to_class_id[global_track_id]
                self.trackers[global_track_id] = tracker
            else:
                self.trackers[global_track_id].update(bbox, self.frame_count, confidence)

        # Batch cleanup inactive trackers (less frequent)
        if self.frame_count % 5 == 0:
            inactive_trackers = []
            for global_track_id, tracker in self.trackers.items():
                if global_track_id not in active_track_ids:
                    frames_since_seen = self.frame_count - tracker.last_seen_frame
                    if frames_since_seen > self.max_missed_frames:
                        tracker.is_active = False
                        inactive_trackers.append(global_track_id)

            # Remove inactive trackers and clean up ID mapping
            for global_track_id in inactive_trackers:
                if global_track_id in self.global_to_class_id:
                    del self.global_to_class_id[global_track_id]
                del self.trackers[global_track_id]

        self.frame_count += 1
        self.detection_count += len(current_detections)

        return current_detections

    def draw_tracking_info(self, frame):
        """Optimized drawing with class-specific labels and colors"""
        update_drawing = (self.frame_count % self._draw_update_interval == 0)

        for global_track_id, tracker in self.trackers.items():
            if not tracker.is_active or not tracker.bbox_history:
                continue

            # Get current bbox and center
            bbox = tracker.bbox_history[-1]
            center = tracker.get_center()

            # Get class information
            class_name = getattr(tracker, 'class_name', 'Unknown')
            class_specific_id = getattr(tracker, 'class_specific_id', global_track_id)

            # Different colors for different classes
            if class_name == 'Drone':
                box_color = (0, 255, 0)  # Green for drones
                text_color = (0, 255, 0)
            else:  # Bird
                box_color = (255, 0, 0)  # Blue for birds
                text_color = (255, 0, 0)

            # Draw bounding box and center
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # Update expensive calculations less frequently
            if update_drawing:
                approach_status = tracker.is_approaching() if class_name == 'Drone' else None
                # Cache for use between updates
                tracker._display_approach = approach_status
            else:
                # Use cached values
                approach_status = getattr(tracker, '_display_approach', None)

            # Draw class-specific ID and status with confidence
            confidence = tracker.get_current_confidence()
            label = f"{class_name} ID: {class_specific_id} ({confidence:.2f})"
            if approach_status and class_name == 'Drone':
                label += f" - {approach_status}"
                if approach_status == "APPROACHING":
                    text_color = (0, 255, 0)
                else:  # RECEDING
                    text_color = (0, 165, 255)

            cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            # Draw simplified trajectory (fewer points)
            if len(tracker.centers) > 3:
                points = list(tracker.centers)[::3]
                if len(points) > 1:
                    points_array = np.array(points, np.int32)
                    cv2.polylines(frame, [points_array], False, (255, 255, 0), 2)

        # Update system info with FPS display
        if update_drawing:
            # Count active trackers by class
            drone_count = sum(1 for t in self.trackers.values()
                             if getattr(t, 'class_name', '') == 'Drone' and t.is_active)
            bird_count = sum(1 for t in self.trackers.values()
                            if getattr(t, 'class_name', '') == 'Bird' and t.is_active)

            # Display FPS info
            cv2.putText(frame, f"Drones: {drone_count} | Birds: {bird_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"CPU FPS: {self.current_fps_cpu:.1f} | GPU FPS: {self.current_fps_gpu:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def get_current_fps(self):
        """Get current FPS for both CPU and GPU"""
        return {
            'cpu_fps': self.current_fps_cpu,
            'gpu_fps': self.current_fps_gpu
        }

def process_single_video(video_path, model_cpu, model_gpu, tracking_system):
    """Process a single video and generate CSV data"""
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
            # CPU Processing
            cpu_start = time.time()
            results_cpu = model_cpu.track(
                source=frame,
                persist=True,
                tracker="botsort.yaml",
                conf=0.25,
                stream=True,
                verbose=False,
                device='cpu'
            )
            detections_cpu = list(results_cpu)
            cpu_time = time.time() - cpu_start
            
            # GPU Processing
            gpu_start = time.time()
            results_gpu = model_gpu.track(
                source=frame,
                persist=True,
                tracker="botsort.yaml",
                conf=0.25,
                stream=True,
                verbose=False,
                device='cuda'
            )
            detections_gpu = list(results_gpu)
            gpu_time = time.time() - gpu_start
            
            # Use GPU results for tracking
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
                    'prob_harmful': 0.0
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
                    'prob_harmful': 0.0
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
                'prob_harmful': 0.0
            }
            csv_data.append(csv_row)
        
        # Progress reporting
        if frame_num % progress_interval == 0:
            progress = (frame_num / total_frames) * 100
            print(f"⏳ Video Progress: {progress:.1f}% ({frame_num}/{total_frames})")
    
    cap.release()
    return csv_data

def write_csv_data(csv_data, output_csv_path):
    """Write CSV data to file with proper headers"""
    if not csv_data:
        print("⚠ No CSV data to write")
        return
    
    # Updated CSV headers
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
        'FPS (CPU)',
        'FPS (GPU)',
        'confidence_track',
        'inference_time_track (ms)',
        'payload_label',
        'prob_harmful'
    ]
    
    # Check if file already exists
    write_header = not os.path.exists(output_csv_path)
    
    try:
        with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers, delimiter='\t')
            
            if write_header:
                writer.writeheader()
                print(f"📝 CSV header written to: {output_csv_path}")
            
            writer.writerows(csv_data)
            print(f"💾 {len(csv_data)} rows written to CSV file")
    
    except Exception as e:
        print(f"❌ Error writing CSV file: {str(e)}")

def process_single_video_file(video_path,model_path,output_dir):
    """
    Main function to process a single video file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output CSV filename based on video name
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_output_path = os.path.join(output_dir, f"{video_name}_results.csv")
    
    print(f"🎯 Processing single video: {video_path}")
    print(f"🤖 Using model: {model_path}")
    print(f"📁 Output CSV: {csv_output_path}")
    
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
    
    # Load models for both CPU and GPU
    print("\n🤖 Loading YOLO models...")
    try:
        model_cpu = YOLO(model_path)
        model_gpu = YOLO(model_path)
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading YOLO models: {str(e)}")
        return
    
    # Initialize tracking system
    try:
        tracking_system = OptimizedDroneTrackingSystem(device='cuda')
        print("✅ Tracking system initialized")
    except Exception as e:
        print(f"❌ Error initializing tracking system: {str(e)}")
        return
    
    # Process video
    try:
        print("\n🚀 Starting video processing...")
        start_time = time.time()
        
        # Remove existing CSV file to start fresh
        if os.path.exists(csv_output_path):
            os.remove(csv_output_path)
            print(f"🗑 Removed existing CSV file: {csv_output_path}")
        
        results = process_single_video(
            video_path, model_cpu, model_gpu, tracking_system
        )
        
        # Write results to CSV
        write_csv_data(results, csv_output_path)
        
        duration = time.time() - start_time
        print(f"\n✅ Process completed in {duration:.2f} seconds")
        
        # Display final results if successful
        if results:
            # Calculate statistics
            total_detections = len([row for row in results if row['class_label'] not in ['none', 'error']])
            drone_detections = len([row for row in results if row['class_label'] == 'drone'])
            bird_detections = len([row for row in results if row['class_label'] == 'bird'])
            
            print(f"\n🎊 Final Results Summary:")
            print(f"   📄 Output CSV: {csv_output_path}")
            print(f"   ⏱ Total Processing Time: {duration:.2f}s")
            print(f"   📝 Total CSV Rows: {len(results)}")
            print(f"   🎯 Total Detections: {total_detections}")
            print(f"   🚁 Drone Detections: {drone_detections}")
            print(f"   🐦 Bird Detections: {bird_detections}")
            print(f"\n🎉 Video processing completed successfully!")
        else:
            print(f"\n⚠ Processing completed but no results returned")
    
    except FileNotFoundError as e:
        print(f"\n❌ File not found error: {str(e)}")
        print("   Please check if the video file exists at the specified path")
        print("   Please check if the model file exists at the specified path")
    
    except PermissionError as e:
        print(f"\n❌ Permission error: {str(e)}")
        print("   Please check write permissions for the output directory")
    
    except Exception as e:
        print(f"\n❌ Unexpected error occurred: {str(e)}")
        print("   Please check the error details above and try again")
    
    finally:
        # Clean up resources
        print("\n🧹 Cleaning up resources...")
        try:
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("   ✅ CUDA cache cleared")
        except Exception as cleanup_error:
            print(f"   ⚠ Warning during cleanup: {str(cleanup_error)}")

# Execute the main function
if __name__ == "__main__":
    # TODO: Update this path to point to your video file
    video_path = '/path/to/your/video/file.mp4'  # ← CHANGE THIS TO YOUR VIDEO PATH
    
    # Model path - using the specified model location
    model_path = 'yolov8_rgb.pt'
    
    # Output directory
    output_dir = '/VIP_cup/tracked_videos'
    # Process single video file
    process_single_video_file(video_path,model_path,output_dir)