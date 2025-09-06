#!/usr/bin/env python3
"""
YOLOv8 Payload Detection Training 
"""

import os
import random
import shutil
import zipfile
import numpy as np
import torch
from ultralytics import YOLO

def seed_all(seed=42):
    """Set all random seeds for reproducibility"""
    print(f"🌱 Seeding everything with seed={seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def check_gpu():
    """Check if CUDA is available and print GPU info"""
    if torch.cuda.is_available():
        print(f"✅ CUDA is available!")
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"🔢 CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("❌ CUDA is not available. Training will use CPU")
        return False

def extract_dataset(zip_path, extract_to="dataset"):
    """Extract dataset from zip file"""
    if not os.path.exists(zip_path):
        print(f"❌ Dataset zip file not found at: {zip_path}")
        return False
    
    try:
        print(f"📦 Extracting dataset from {zip_path}...")
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ Dataset extracted to: {extract_to}")
        return True
    except Exception as e:
        print(f"❌ Error extracting dataset: {e}")
        return False

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Split dataset into train/val/test sets"""
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return False
    
    if not os.path.exists(labels_dir):
        print(f"❌ Labels directory not found: {labels_dir}")
        return False
        
    print(f"📊 Splitting dataset from {images_dir}...")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if not image_files:
        print(f"❌ No image files found in {images_dir}")
        return False
        
    random.shuffle(image_files)
    total_images = len(image_files)
    
    # Calculate split sizes
    train_size = int(train_ratio * total_images)
    val_size = int(val_ratio * total_images)
    # test_size is the remainder to ensure all images are used
    
    train_files = image_files[:train_size]
    val_files = image_files[train_size:train_size + val_size]
    test_files = image_files[train_size + val_size:]
    
    print(f"📈 Dataset split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    # Copy files to respective directories
    for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        for file in files:
            try:
                # Copy image
                shutil.copy2(os.path.join(images_dir, file), os.path.join(output_dir, split, 'images', file))
                
                # Copy corresponding label
                label_file = file.rsplit('.', 1)[0] + '.txt'
                label_path = os.path.join(labels_dir, label_file)
                if os.path.exists(label_path):
                    shutil.copy2(label_path, os.path.join(output_dir, split, 'labels', label_file))
                else:
                    print(f"⚠️  Label file not found: {label_file}")
            except Exception as e:
                print(f"❌ Error copying {file}: {e}")
                return False
                
    print("✅ Dataset split completed!")
    return True

def create_data_yaml(dataset_path, output_path="data.yaml"):
    """Create YAML config file for YOLO training"""
    yaml_content = f"""train: {os.path.abspath(os.path.join(dataset_path, 'train', 'images'))}
val: {os.path.abspath(os.path.join(dataset_path, 'val', 'images'))}
test: {os.path.abspath(os.path.join(dataset_path, 'test', 'images'))}

nc: 2
names: ['Harmful', 'Normal']
"""
    
    try:
        with open(output_path, 'w') as f:
            f.write(yaml_content.strip())
        print(f"📝 Created data.yaml at: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Error creating data.yaml: {e}")
        return None

def download_pretrained_model(model_name="yolov8m.pt"):
    """Download pretrained YOLOv8 model if not present"""
    try:
        if not os.path.exists(model_name):
            print(f"📥 Downloading {model_name}...")
            YOLO(model_name)  # Triggers download
            print(f"✅ Downloaded {model_name}")
        else:
            print(f"✅ {model_name} already exists")
        return model_name
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return None

def train_model(data_yaml, model_path, save_dir):
    """Train YOLOv8 model and save checkpoints"""
    print("🚀 Starting training...")
    os.makedirs(save_dir, exist_ok=True)

    run_dir = os.path.join(save_dir, "payload_detection")
    ckpt_path = os.path.join(run_dir, "weights", "last.pt")

    # Resume if checkpoint exists
    resume_training = os.path.exists(ckpt_path)
    
    if resume_training:
        print(f"🔄 Resuming training from: {ckpt_path}")
        model = YOLO(ckpt_path)
    else:
        print(f"🆕 Starting fresh training with: {model_path}")
        model = YOLO(model_path)

    try:
        results = model.train(
            data=data_yaml,
            epochs=100,
            batch=64,
            imgsz=(320, 256),
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.001,
            weight_decay=0.0005,
            warmup_epochs=5,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            amp=True,
            device=0 if torch.cuda.is_available() else 'cpu',
            workers=4,
            project=save_dir,
            name="payload_detection",
            exist_ok=True,
            verbose=True,
            save=True,
            save_period=10,
            resume=resume_training,
            plots=True,
        )
        print("✅ Training completed!")
        return results
    except Exception as e:
        print(f"❌ Training error: {e}")
        return None

def main():
    """Main script logic"""
    print("🔥 YOLOv8 payload Detection - Local Training")
    print("=" * 50)

    seed_all(42)
    gpu_available = check_gpu()

    # Paths
    base_dir = os.path.abspath(os.path.dirname(__file__))
    rgb_dir = os.path.join(base_dir)
    results_base = os.path.join(rgb_dir, "Results")
    dataset_dir = os.path.join(rgb_dir, "dataset")
    extracted_rgb_images = os.path.join(dataset_dir, "images")
    extracted_rgb_labels = os.path.join(dataset_dir, "labels")
    split_output_dir = os.path.join(rgb_dir, "split_dataset")
    data_yaml_path = os.path.join(rgb_dir, "data.yaml")
    model_name = "yolov8m.pt"

    # Validate required directories exist
    if not os.path.exists(extracted_rgb_images):
        print(f"❌ Images directory not found: {extracted_rgb_images}")
        print("Please ensure your dataset is properly extracted.")
        return
        
    if not os.path.exists(extracted_rgb_labels):
        print(f"❌ Labels directory not found: {extracted_rgb_labels}")
        print("Please ensure your dataset is properly extracted.")
        return

    # Step 1: Split dataset
    if not split_dataset(extracted_rgb_images, extracted_rgb_labels, split_output_dir):
        print("❌ Dataset splitting failed!")
        return

    # Step 2: Create data.yaml
    yaml_path = create_data_yaml(split_output_dir, data_yaml_path)
    if not yaml_path:
        print("❌ Failed to create data.yaml!")
        return

    # Step 3: Download model
    model_path = download_pretrained_model(model_name)
    if not model_path:
        print("❌ Failed to download model!")
        return

    # Step 4: Train
    results = train_model(yaml_path, model_path, results_base)
    
    if results:
        print("\n🎉 Training pipeline completed!")
        print(f"📊 Results in: {os.path.join(results_base, 'payload_detection')}")
        print(f"🏆 Best model: {os.path.join(results_base, 'payload_detection', 'weights', 'best.pt')}")
    else:
        print("❌ Training pipeline failed!")

if __name__ == "__main__":
    main()
