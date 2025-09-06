#!/usr/bin/env python3
"""
YOLOv8 Bird vs Drone Detection Training
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
    print(f"📦 Extracting dataset from {zip_path}...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Dataset extracted to: {extract_to}")
    return True

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    """Split dataset into train/val/test sets"""
    print(f"📊 Splitting dataset from {images_dir}...")
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(image_files)
    total_images = len(image_files)
    train_size = int(train_ratio * total_images)
    val_size = int(val_ratio * total_images)
    train_files = image_files[:train_size]
    val_files = image_files[train_size:train_size + val_size]
    test_files = image_files[train_size + val_size:]
    print(f"📈 Dataset split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        for file in files:
            shutil.copy2(os.path.join(images_dir, file), os.path.join(output_dir, split, 'images', file))
            label_file = file.rsplit('.', 1)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            if os.path.exists(label_path):
                shutil.copy2(label_path, os.path.join(output_dir, split, 'labels', label_file))
    print("✅ Dataset split completed!")
    return True

def create_data_yaml(dataset_path, output_path="data.yaml"):
    """Create YAML config file for YOLO training"""
    yaml_content = f"""
train: {os.path.join(dataset_path, 'train', 'images')}
val: {os.path.join(dataset_path, 'val', 'images')}
test: {os.path.join(dataset_path, 'test', 'images')}

nc: 2
names: ['Bird', 'Drone']
"""
    with open(output_path, 'w') as f:
        f.write(yaml_content.strip())
    print(f"📝 Created data.yaml at: {output_path}")
    return output_path

def download_pretrained_model(model_name="yolov8m.pt"):
    """Download pretrained YOLOv8 model if not present"""
    if not os.path.exists(model_name):
        print(f"📥 Downloading {model_name}...")
        YOLO(model_name)  # Triggers download
        print(f"✅ Downloaded {model_name}")
    else:
        print(f"✅ {model_name} already exists")
    return model_name

def train_model(data_yaml, model_path, save_dir):
    """Train YOLOv8 model and save checkpoints"""
    print("🚀 Starting training...")
    os.makedirs(save_dir, exist_ok=True)
    model = YOLO(model_path)
    results = model.train(
        data=data_yaml,
        epochs=20,
	patience=10,
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
        device=0,
        workers=4,
        project=save_dir,
        name="drone_bird_detection",
        exist_ok=True,
        verbose=True,
        save=True,
        save_period=5,  # Save every 10 epochs
	resume=False,
	plots=True,
    )
    print("✅ Training completed!")
    return results

def main():
    """Main script logic"""
    print("🔥 YOLOv8 Bird vs Drone Detection - Local Training")
    print("=" * 50)

    seed_all(42)
    check_gpu()

    # Paths
    base_dir = os.path.abspath(os.path.dirname(__file__))
    rgb_dir = os.path.join(base_dir)
    results_base = os.path.join(rgb_dir, "results", "yoloV8", "epoch_100from_80")
    dataset_zip = os.path.join(base_dir, "..", "releasev1-detection&tracking.zip")
    dataset_dir = os.path.join(rgb_dir, "dataset")
    extracted_rgb_images = os.path.join(dataset_dir, "releasev1-detection&tracking", "RGB", "images")
    extracted_rgb_labels = os.path.join(dataset_dir, "releasev1-detection&tracking", "RGB", "labels")
    split_output_dir = os.path.join(rgb_dir, "split_dataset")
    data_yaml_path = os.path.join(rgb_dir, "data.yaml")
    # Start training from 30-epoch best.pt instead of fresh yolov8m.pt
	#model_path = os.path.join(rgb_dir, "results", "yoloV8", "epoch_30", "YOLO_v8_30_epoches_best.pt")
    model_path = os.path.join(
    	rgb_dir, "results", "yoloV8", "epoch_80from_60", "drone_bird_detection", "weights", "best.pt"
    )


    # Step 1: Extract dataset
    if not os.path.exists(dataset_dir):
        if not extract_dataset(dataset_zip, dataset_dir):
            return

    # Step 2: Split dataset
    if not split_dataset(extracted_rgb_images, extracted_rgb_labels, split_output_dir):
        return

    # Step 3: Create data.yaml
    yaml_path = create_data_yaml(split_output_dir, data_yaml_path)

    # Step 4: Download model
    #model_path = download_pretrained_model(model_name)

    # Step 5: Train
    results = train_model(yaml_path, model_path, results_base)

    print("\n🎉 Training pipeline completed!")
    print(f"📊 Results in: {os.path.join(results_base, 'drone_bird_detection')}")
    print(f"🏆 Best model: {os.path.join(results_base, 'drone_bird_detection', 'weights', 'best.pt')}")

if __name__ == "__main__":
    main()
