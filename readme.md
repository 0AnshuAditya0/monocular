# MONOCULAR // Spatial Risk Engine

![ss](sample12.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-black.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-black.svg)](https://opencv.org/)

**MONOCULAR** is a real-time visual-spatial intelligence system that fuses monocular depth estimation (MDE) with semantic object detection to perform 3D-aware collision risk assessment from a single RGB camera stream. 

By bypassing the need for physical LiDAR or stereo-camera rigs, MONOCULAR acts as a software-defined depth engine, projecting standard 2D video feeds into a simulated 3D spatial coordinate system.



---

##  System Visualization

The pipeline outputs a synchronous, three-panel spatial telemetry stream:

* **Left (Semantic Detection):** YOLOv8n tracking with dynamic class hazard overlays.
* **Center (Latent Depth Map):** Frame-by-frame relative depth map visualized via the `INFERNO` colormap.
* **Right (Bird's Eye View - BEV):** A top-down 2D spatial grid mapping object proximity and lateral orientation.

> [!NOTE]
> *Insert your 3-panel screenshot here showing the pedestrian scene and the reconstructed BEV grid.*

---

##  Core Technical Features

* **Pseudo-LiDAR Projection:** Translates 2D bounding boxes and relative depth maps into top-down $(X, Z)$ coordinate space, plotting target objects relative to the camera center.
* **Scene-Adaptive Risk Thresholding:** Evaluates hazard levels dynamically per frame using percentile-based cuts ($Top\ 30\% = \text{DANGER}$, $Mid\ 30\% = \text{WARNING}$) to adapt automatically to changing environments without hard-coded distance limitations.
* **Decoupled Multi-Queue Inference:** Processes lightweight YOLOv8n detections at full frame rate while decoupling the heavier MDE backbone (Depth Anything V2) to execute on a $1:3$ frame ratio, utilizing spatial temporal propagation to maintain system throughput.
* **Center-Weighted Median Sampling:** Extracts depth indicators exclusively from the central 50% of the bounding box coordinates, eliminating edge noise and background leakage artifacts.

---

## System Architecture
```mermaid
graph TD
    subgraph Input_Processing [Input Stream]
        A[RGB Video Frame] --> B{Pre-processor}
        B -->|Resized Tensor| C1[Semantic Stream: YOLOv8n]
        B -->|Normalized Image| C2[Geometric Stream: Depth Anything V2]
    end

    subgraph Frequency_Control [Inference Scheduling]
        C1 -->|Every Frame| D1[Bounding Boxes]
        C2 -->|Every 3rd Frame| D2[Relative Depth Map]
    end

    subgraph Fusion_Layer [Spatial Fusion Engine]
        D1 & D2 --> E[Median ROI Sampling]
        E --> F[Coordinate Transformation]
        F --> G[Percentile-based Risk Logic]
    end

    subgraph Output_Telemetry [Visualization Panels]
        G --> H1[Panel 1: Risk Overlays]
        G --> H2[Panel 2: Depth Heatmap]
        G --> H3[Panel 3: BEV Grid Map]
    end

    style Input_Processing fill:#f9f9f9,stroke:#333,stroke-width:2px
    style Fusion_Layer fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style Output_Telemetry fill:#fff3e0,stroke:#e65100,stroke-width:2px
---

## 📊 Performance & Specifications

| Dimension | Metric / Technology |
| :--- | :--- |
| **Depth Engine** | Depth Anything V2 Small (ViT-S backbone, 62M image pre-training) |
| **Semantic Detector** | YOLOv8n (Ultralytics) |
| **Inference Latency** | ~15-20 FPS on standard cloud-allocated T4 GPUs |
| **Alert Tiers** | `DANGER` (Red), `WARNING` (Orange), `SAFE` (Green) |
| **Output Render** | OpenCV custom BEV grid & rendering engine |

---

## 📂 Project Directory Structure

```text
monocular/
├── models/
│   ├── depth_anything_v2_small.onnx       # Quantized ONNX weights
│   └── depth_anything_v2_small.onnx.data  # Calibration metadata
├── depth.py                               # Main CLI inference & fusion pipeline
├── app.py                                 # Gradio web portal (In Development)
├── requirements.txt                       # Engine dependencies
└── README.md                              # System documentation


🚀 Installation & Local Execution
1. Environment Setup
Bash
# Clone the repository
git clone [https://github.com/0AnshuAditya0/monocular](https://github.com/0AnshuAditya0/monocular)
cd monocular

# Initialize virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt