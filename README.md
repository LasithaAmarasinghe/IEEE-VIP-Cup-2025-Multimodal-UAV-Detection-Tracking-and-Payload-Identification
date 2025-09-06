# IEEE Signal Processing Society - Video & Image Processing Cup 2025 - Drone Detection-Tracking-and-Payload-Identification Challenge

## 📌 Project Overview  
This repository contains **Team Cogniview’s** solution for the **IEEE SPS VIP CUP 2025**.  

Our system is designed to **detect, track, and classify drones and their payloads in real time** using **multimodal fusion of RGB and infrared (IR) imagery**. By leveraging advanced deep learning models, we address the limitations of single-modality approaches and enhance robustness under **low light, fog, occlusion, motion blur,** and other challenging environmental conditions.  

---
  
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
Our solution integrates multiple components to ensure **high detection accuracy, robust tracking, and efficient real-time performance**:  

- **Multimodal Fusion** – Implemented **Y-shape Dynamic Transformer (YDTR)** to combine RGB and IR features for improved detection robustness.  
- **Object Detection** – Used **YOLOv8/YOLOv10** for high-precision drone and bird detection.  
- **Multi-Object Tracking** – Integrated **BoT-SORT** for persistent ID tracking and motion analysis.  
- **Motion Behavior Analysis** – Designed algorithm to determine whether a drone is **approaching or receding** from the camera.  
- **Payload Classification** – Built RGB-IR early-fusion CNN classifier to identify **harmful vs. benign payloads** with **>99% F1-score**.  
- **Performance Optimization** – Achieved **25–30 FPS** on GPU with low latency and high scalability for real-world deployment.  

---

## 📊 Results  
- **Drone Detection (RGB-IR Fusion)** – F1-Score: **0.9846**  
- **Payload Classification** – F1-Score: **>99%**, mAP@50-95: **0.9947**  
- **Tracking Performance** – **90%+** tracking persistence under occlusion and environmental distortions  
- **Real-Time Capability** – **25–30 FPS** with minimal resource usage  

---

## 🚀 Applications  
This system demonstrates strong potential for:  
- **Security & Defense** – Perimeter and airspace monitoring  
- **Critical Infrastructure Protection** – Detection of unauthorized UAV activity  
- **Event Surveillance** – Crowd safety and aerial monitoring  
- **Wildlife Protection** – Differentiating UAVs from birds to reduce false alarms  

---
