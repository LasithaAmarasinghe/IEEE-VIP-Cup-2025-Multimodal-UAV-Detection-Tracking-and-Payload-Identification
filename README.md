# IEEE Signal Processing Society - Video & Image Processing Cup 2025 - Drone Detection-Tracking-and-Payload-Identification Challenge

## 📌 Project Overview  
This repository contains **Team Cogniview’s** solution for the **IEEE SPS VIP CUP 2025**.  

Our system is designed to **detect, track, and classify drones and their payloads in real time** using **multimodal fusion of RGB and infrared (IR) imagery**.  
By leveraging advanced deep learning models, we address the limitations of single-modality approaches and enhance robustness under **low light, fog, occlusion, motion blur,** and other challenging environmental conditions.  

---

## 🎯 Challenge Description  
The competition involves three core tasks:  

1. **RGB + IR Drone Detection** – Distinguish drones from birds using both visual (RGB) and thermal (IR) imagery.  
2. **RGB + IR Drone Tracking** – Track drone trajectories and determine **approaching vs. receding** motion patterns.  
3. **Payload Detection** – Classify payloads as **harmful or normal** using RGB-IR fused inputs.  

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
