# IEEE Signal Processing Society - Video & Image Processing Cup 2025 - Drone Detection-Tracking-and-Payload-Identification Challenge

## 📌 Project Overview  
This repository contains **Team Cogniview’s** solution for the **IEEE SPS VIP CUP 2025**.  

Our system is designed to **detect, track, and classify drones and their payloads in real time** using **multimodal fusion of RGB and infrared (IR) imagery**. By leveraging advanced deep learning models, we address the limitations of single-modality approaches and enhance robustness under **low light, fog, occlusion, motion blur,** and other challenging environmental conditions.  

---

<p align="center">
  <img src="Results/video.gif" alt="Demo GIF" />
</p>

  
## 🎯 Challenge Description  

The **IEEE SPS VIP Cup 2025** focuses on **infrared–visual fusion for drone surveillance**. With drones becoming increasingly common in civilian, industrial, and military contexts, the competition highlights the urgent need for systems capable of **real-time UAV detection, tracking, and payload identification** under adverse conditions.  

Traditional RGB-only systems fail in poor visibility, while IR-only systems lack spatial detail. **Fusing RGB and IR modalities** provides complementary strengths, enabling **robust drone recognition even in extreme environments**.  

### Core Tasks  

1. **RGB + IR Drone Detection**  
   - Detect drones and differentiate them from birds in real time.  
   - Handle adverse conditions: **low light, fog, forest cover, hilly terrain**, and distortions like **motion blur, Gaussian noise, uneven illumination, and camera instability**.  
   - Evaluated on **precision, recall, F1-score, mAP**, robustness under distortions, and **real-time inference speed**.  

2. **RGB + IR Drone Tracking**  
   - Track drone trajectories across video frames with persistent IDs, even under occlusion.  
   - Determine whether drones are **approaching or receding** from the camera.  
   - Evaluated on **trajectory IoU (accuracy), consistency (missed frames), directional accuracy, and FPS latency**.  

3. **Payload Detection & Identification**  
   - Classify payloads as **harmful** (e.g., explosives, contraband, surveillance devices) or **normal**.  
   - Address variability in **size, shape, and thermal signatures**.  
   - Evaluated on **accuracy, precision, recall, F1-score, and mAP**, ensuring high robustness under adverse environments.  

### 📊 Evaluation & Ranking  

- **Detection Score (S_det):** Combines accuracy, F1, precision, recall, robustness, and speed.  
- **Tracking Score (S_track):** Based on IoU, continuity, directional accuracy, and speed.  
- **Payload Score (S_payload):** Based on classification metrics and mAP.  
- **Overall Ranking (S_overall):** Average of all three tasks. 

---

## 👨‍💻 Team Members – *Team Cogniview*  
- **Lahiru Cooray** *(Team Leader)*  
- **Lasitha Amarasinghe**  
- **Shaveen Herath**  
- **Mihiraja Kuruppu**  
- **Shemal Perera**  
- **Kavishka Abeywardana**  
- **Ravija Dulnath**  
- **Dinuka Madhushan**  
- **Dilsha Mihiranga**  
- **Chandeepa Peiris**  
- **Dr. Wageesha Manamperi** *(Supervisor, University of Moratuwa)*  
- **Muditha Fernando** *(Postgraduate Mentor)*  

---

## 📂 Dataset  
The challenge dataset includes:  
- **45,000 IR-RGB image pairs** for drone detection training  
- **25,000 IR-RGB image pairs** for payload identification training  
- Image resolution: **320 × 256 pixels**  
- Data captured under various **environmental conditions** (day, night, fog, occlusion) and **distortions** (motion blur, noise)  

---

## ⚙️ Methodology  

Our solution integrates multiple deep learning and computer vision components to ensure **high detection accuracy**, **robust tracking**, and **efficient real-time performance**. The pipeline is modular, supporting independent evaluation of RGB, IR, and fused modalities, as required by the competition.  

### 1. Multimodal Fusion  
We implemented **two fusion strategies** to combine RGB and IR modalities:  

**(a) Early Fusion – Y-shape Dynamic Transformer (YDTR)**  
- Implemented a **Y-shape Dynamic Transformer (YDTR)** to fuse features from RGB and IR images.  
- The **RGB branch** captures fine-grained **textural and spatial details**, while the **IR branch** extracts **thermal signatures** that remain robust under poor visibility.  
- Features from both branches are merged in a **Dynamic Transformer Module (DTRM)**, which models **long-range dependencies** and aligns semantic features across modalities.  
- The fused representation is then passed into a YOLO-based detector for classification of drones vs. birds.  
- This approach significantly improves detection and classification performance under **low light, fog, occlusion, motion blur, and camera instability**.  

**(b) Late Fusion – Decision-Level Fusion**  
- Object detection is performed **independently** on RGB-only and IR-only pipelines using YOLOv8.  
- Each modality produces bounding boxes, class labels, and confidence scores.  
- The results are then merged using **Non-Maximum Suppression (NMS)** to remove redundant detections and consolidate final predictions.  
- This method allows the system to **fall back on the stronger modality** in cases where one modality is heavily degraded (e.g., RGB in dense fog or IR in thermal clutter).  

### 2. Object Detection  
- Adopted **YOLOv8** and **YOLOv10** as backbone detectors due to their balance of **accuracy and inference speed**.  
- Three parallel models are trained as required by the competition:  
  - **RGB-only detector**  
  - **IR-only detector**  
  - **RGB-IR fusion detector (YDTR + YOLO)**  
- Standard preprocessing: resizing to **320×256**, normalization, and heavy augmentations (**flips, jitter, blur, noise**) for robustness.  
- Evaluation metrics: **Precision, Recall, F1-score, mAP@50, mAP@50–95**.  

### 3. Multi-Object Tracking  
- Integrated **BoT-SORT** (ByteTrack with appearance embeddings) for robust ID association across frames.  
- Enhancements include:  
  - **Kalman filtering** for trajectory prediction.  
  - **Deque-based buffers** to store object history for smoother velocity and area change estimation.  
  - **Adaptive windowing** (5–20 frames) to reduce noise and jitter in real-time tracking.  
- Achieved **>90% tracking persistence**, even under occlusion and camera instability.  

### 4. Motion Behavior Analysis  
- Developed a **trajectory analysis algorithm** to determine whether a drone is **approaching** or **receding** from the field of view.  
- Classification based on:  
  - **Change in bounding box size (area growth/shrinkage)**  
  - **Velocity vectors** relative to the camera.  
- Real-time visualization overlays: bounding boxes, IDs, velocity trails, and approach/recede labels.  

### 5. Payload Classification  
- Built a **RGB-IR early-fusion CNN classifier** with stacked channel inputs.  
- Architecture:  
  - 3 × convolutional layers with **ReLU + BatchNorm**  
  - 2 × fully connected layers  
  - Softmax for final classification  
- Task: Distinguish **harmful payloads** (explosives, contraband, surveillance devices) from **normal payloads**.  
- Achieved **>99% F1-score** and **0.9947 mAP@50–95** on the validation dataset.  

### 6. Performance Optimization  
- Real-time efficiency optimized through:  
  - **Lazy evaluation & caching** to minimize redundant computations.  
  - **Batch normalization & lightweight CNN backbones** for reduced inference cost.  
  - **Mixed precision inference** (FP16) on GPU.  
- Achieved **25–30 FPS** on GPU with minimal latency, making the system deployable in real-world scenarios.  

Overall, our methodology ensures **robust multimodal fusion**, **high-precision detection**, **persistent tracking**, and **accurate payload identification**, while meeting the **real-time performance requirements** of the IEEE SPS VIP Cup 2025.  

---

## 📊 Results  
- **Drone Detection (RGB-IR Fusion)** – F1-Score: **0.9846**  
- **Payload Classification** – F1-Score: **>99%**, mAP@50-95: **0.9947**  
- **Tracking Performance** – **90%+** tracking persistence under occlusion and environmental distortions  
- **Real-Time Capability** – **25–30 FPS** with minimal resource usage  

---

### 🚁 Drone Detection – RGB Only  
<div style="text-align:center; width:100%;">
  <div style="display:inline-block; width:32%; margin:0 1%;">
    <img src="Results/drone_detection_rgb/confusion_matrix.png" alt="RGB Confusion Matrix" style="width:100%;"/>
    <div>Confusion Matrix</div>
  </div>
  <div style="display:inline-block; width:32%; margin:0 1%;">
    <img src="Results/drone_detection_rgb/val_batch0_pred.jpg" alt="Detection Example 1" style="width:100%;"/>
    <div>Detection Example 1</div>
  </div>
  <div style="display:inline-block; width:32%; margin:0 1%;">
    <img src="Results/drone_detection_rgb/val_batch1_pred.jpg" alt="Detection Example 2" style="width:100%;"/>
    <div>Detection Example 2</div>
  </div>
</div>
  

---

### 🚁 Drone Detection – IR Only  
<p align="center">
  <img src="Results/drone_detect_ir/confusion_matrix.png" alt="IR Confusion Matrix" style="width:33%; max-width:325px;"/>
  <img src="Results/drone_detect_ir/val_batch0_pred.jpg" alt="IR Detection Sample 1" style="width:33%; max-width:300px;"/>
  <img src="Results/drone_detect_ir/val_batch1_pred.jpg" alt="IR Detection Sample 2" style="width:33%; max-width:300px;"/>
</p>  
<p align="center">
  Confusion Matrix | Detection Example 1 | Detection Example 2
</p>  

---

### 🚁 Drone Detection – RGB + IR Fusion  
<p align="center">
  <img src="Results/drone_detect_fuse/confusion_matrix.png" alt="Fusion Confusion Matrix" style="width:33%; max-width:325px;"/>
  <img src="Results/drone_detect_fuse/val_batch0_pred.jpg" alt="Fusion Detection Sample 1" style="width:33%; max-width:300px;"/>
  <img src="Results/drone_detect_fuse/val_batch1_pred.jpg" alt="Fusion Detection Sample 2" style="width:33%; max-width:300px;"/>
</p>  
<p align="center">
  Confusion Matrix | Detection Example 1 | Detection Example 2
</p>  

---

### 📦 Payload Classification – RGB Only  
<p align="center">
  <img src="Results/payload_rgb/confusion_matrix.png" alt="RGB Payload Confusion Matrix" width="450"/>
  <img src="Results/payload_rgb/val_batch2_labels.jpg" alt="RGB Payload Sample" width="350"/>
</p>  
<p align="center">
  Confusion Matrix | Classification Example
</p>  

---

### 📦 Payload Classification – IR Only  
<p align="center">
  <img src="Results/payload_ir/confusion_matrix.png" alt="IR Payload Confusion Matrix" width="450"/>
  <img src="Results/payload_ir/val_batch2_labels.jpg" alt="IR Payload Sample" width="350"/>
</p>  
<p align="center">
  Confusion Matrix | Classification Example
</p>  

---

### 📦 Payload Classification – RGB + IR Fusion  
<p align="center">
  <img src="Results/payload_fused/confusion_matrix.png" alt="Fusion Payload Confusion Matrix" width="450"/>
  <img src="Results/payload_fused/val_batch1_pred.jpg" alt="Fusion Payload Sample" width="350"/>
</p>  
<p align="center">
  Confusion Matrix | Classification Example
</p>  

---

## 🚀 Applications  
This system demonstrates strong potential for:  
- **Security & Defense** – Perimeter and airspace monitoring  
- **Critical Infrastructure Protection** – Detection of unauthorized UAV activity  
- **Event Surveillance** – Crowd safety and aerial monitoring  
- **Wildlife Protection** – Differentiating UAVs from birds to reduce false alarms  

---
