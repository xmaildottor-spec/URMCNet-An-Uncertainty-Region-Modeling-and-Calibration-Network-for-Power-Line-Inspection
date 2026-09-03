# URMCNet: An Uncertainty Region Modeling and Calibration Network for Power Line Inspection - CODE

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-VITLD%20%7C%20TTPLA-green)](https://github.com/xmaildottor-spec/An-Uncertainty-Region-Modeling-and-Calibration-Network)

> **Official implementation of URMCNet.**
> This repository contains the code for *An Uncertainty Region Modeling and Calibration Network for Power Line Inspection*, including the updated training pipeline and a newly added Stereo 3D Reconstruction & UAV Path Planning module.

---

## 📖 Abstract

Accurate power line detection is critical for unmanned aerial vehicle (UAV)-based inspection systems. However, detection accuracy is often compromised by visually complex backgrounds (similar textures, environmental interference), leading to substantial false positives and false negatives. 

To address this, we propose **URMCNet**, featuring:
* **Uncertainty Region Modeling and Calibration Strategy:** Explicitly models and calibrates regions prone to errors.
* **Primary Feature Calibration Mechanism:** Incorporates False-Positive Region Suppression and False-Negative Region Compensation modules to adaptively calibrate uncertain regions.
* **High-Frequency Aware Fusion Decoder:** Effectively restores fine-grained details to guarantee the continuity of prediction results.

Experimental results on the **VITLD** and **TTPLA** datasets demonstrate that URMCNet outperforms state-of-the-art methods.

---

## 🏗️ Architecture

![Process Model](https://github.com/xmaildottor-spec/URMCNet-An-Uncertainty-Region-Modeling-and-Calibration-Network-for-Power-Line-Inspection-code/blob/main/main.png)

---

## 🎯 Motivation & Application

> **Note to Practitioners:**
> UAV-based inspection is crucial for smart grids but fails in complex environments where power lines are obscured by trees or buildings. 

Our approach enhances reliability by:
1. Focusing on areas where prediction errors are most likely (Uncertainty Modeling).
2. Restoring fine details to ensure continuous detection lines.

This technique is applicable not only to power lines but also to environmental monitoring, infrastructure inspection, and agricultural analysis.

---

## 📂 Datasets

We evaluate our method on the following datasets:

| Dataset | Link | Description |
| :--- | :--- | :--- |
| **VITLD** | [Download Link](https://bit.ly/3FBYjBY) | The Visible-Infrared Transmission Line Detection dataset. |
| **TTPLA** | [Download Link](https://drive.usercontent.google.com/download?id=1Yz59yXCiPKS0_X4K3x9mW22NLnxjvrr0&export=download&authuser=0) | Transmission Tower and Power Line Aerial images. |

## 📂 Processed VITL Datasets

For seamless reproduction of our framework, we provide two pre-processed versions of the VITL dataset. In both versions, the raw infrared data has been globally normalized and converted from CSV to standard image formats to prevent stitching seams and radiometric discontinuities.

| Dataset | Link | Description |
| :--- | :--- | :--- |
| **VITL_Version_1** | [Download Link](https://drive.google.com/file/d/1kqmcpOdlLE_-1iVfQ1icy6lLyR07ONRd/view?usp=sharing) | The **cropped** version of the VITL dataset (256×256 patch format). |
| **VITL_Version_2** | [Download Link](https://drive.google.com/file/d/1d_-_MB0ywp0thZgr8Y_uy2mLg0KCGqS2/view?usp=sharing) | The **stitched** version of the VITL dataset (512×512 high-resolution format). |

### ⚠️ VITLD Dataset Details

1. **Data Grouping:** In the original VITLD dataset, **every four images correspond to one sliced sample**. During testing, ensure images are processed in groups of four in their original order.
2. **Infrared Images:** IR images in this repo are for **visualization purposes only** and are not used in the network training/testing phases.
3. **Preprocessing:** We recommend applying **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to input images to highlight texture details.
4. **Metrics:** We report the macro metrics evaluated on the original test set.

### ⚠️ TTPLA Dataset Details

#### 1. Access and Download
| Item | Link / Details |
| :--- | :--- |
| **Official Repository** | [R3ab/ttpla_dataset](https://github.com/R3ab/ttpla_dataset) |
| **Download Link** | [Google Drive (Direct Download)](https://drive.google.com/uc?export=download&confirm=no_antivirus&id=1Yz59yXCiPKS0_X4K3x9mW22NLnxjvrr0) |
| **Annotation Format** | Original images with **pixel-level annotations** in **COCO format**. |
| **Data Split** | We follow the official split (Train/Val/Test) as reported by the original authors. |

#### 2. Preprocessing & Resolution Scaling
* **Resolution:** The original image resolution is 3840 × 2160. Consistent with previous research, we downsample both images and masks to **512 × 512** for training and testing.
* **Data Cleaning:** We observed minor labeling errors in the original dataset (e.g., transmission towers or rooftops mislabeled as power lines, or missed annotations). To ensure data quality, we manually filtered and **removed these incorrectly labeled images** from our pipeline.

Examples of incorrect labels (incorrect labels are not limited to the following two images).
<p align="center">
  <img src="https://github.com/xmaildottor-spec/URMCNet-An-Uncertainty-Region-Modeling-and-Calibration-Network-for-Power-Line-Inspection-code/raw/main/TTPLA_F.png" alt="Data Cleaning" width="50%">
</p>

## 💾 Trained Weights

We provide the pre-trained weights for **URMCNet-B** and **URMCNet-L** evaluated on the **VITLD** and **TTPLA** datasets. All checkpoints are hosted on Google Drive.

> **💡 Note on VITLD Checkpoints:** 
> While maintaining the core methodology of our proposed model, we have further optimized the codebase. This optimization significantly reduces the model's parameters and computational complexity (FLOPs) while achieving improved performance metrics. Additionally, we provide weight files trained under different random seeds for this dataset to ensure reproducibility.

| Dataset | Model Variant | Checkpoint (Google Drive) | Description |
| :---: | :--- | :---: | :--- |
| **VITLD** | 🚀 **URMCNet-B** | [Download 🔗](https://drive.google.com/drive/folders/1Kx1kkWAXOmDFcXuTuexRQ6BsXNXC7fGX?usp=sharing) | Multi-seed weights; Optimized code with fewer params & better metrics |
| **VITLD** | 🌟 **URMCNet-L** | [Download 🔗](https://drive.google.com/drive/folders/1U8YIUtGuM0ztluRq3lWn9QKJwHL5C2GF?usp=sharing) | Multi-seed weights; Optimized code with fewer params & better metrics |
| **TTPLA** | 🚀 **URMCNet-B** | [Download 🔗](https://drive.google.com/drive/folders/1otFweAm1iQv_rUDRxT3b1i4hl-5o6SDZ?usp=sharing) | Pre-trained weights on the TTPLA dataset; Optimized code with fewer params & better metrics |
| **TTPLA** | 🌟 **URMCNet-L** | [Download 🔗](https://drive.google.com/drive/folders/17j516Jqk-v6SC7mLZQP-8QvJ9PDhkdae?usp=sharing) | Pre-trained weights on the TTPLA dataset; Optimized code with fewer params & better metrics |

## 📚 Corrigendum
Due to the inadvertent submission of a draft version during the editing process, the following corrections are made:
![Network Architecture](https://github.com/xmaildottor-spec/URMCNet-An-Uncertainty-Region-Modeling-and-Calibration-Network-for-Power-Line-Inspection-code/blob/main/IMG.png)

## 🙏 Acknowledgement

We thank [Multimodal-FFM-TLD](https://github.com/hyeyeon08/Multimodal-FFM-TLD) for providing relevant data and code.

## 🚁 Extension: Stereo 3D Reconstruction & UAV Path Planning

Beyond segmentation, we provide a demo for **stereo-camera 3D reconstruction** to assist in UAV obstacle avoidance.

### Modules Overview

#### 1. `3d_demo`: Stereo 3D Reconstruction
Generates 3D points from segmentation masks.
* **Input:** `left_mask.png`, `right_mask.png`
* **Output:** `pts_cam.txt`, `pts_body.txt`, `waypoints.json`
* **Core Functions:** Skeleton extraction, Disparity computation, Stereo triangulation, Coordinate transformation.

#### 2. `demo_0`: Stereo Correspondence Visualization
* Visualizes line-to-line matches between left and right masks.

#### 3. `demo_1`: Match Verification
* **Yellow Points:** Matched points.
* **Red Points:** Unmatched points.
* *Use case:* Debugging segmentation gaps or occlusion failures.

#### 4. `UAV_waypoints`: Path Generation
Fits a B-spline curve to the 3D cloud and computes a safe flight path.
* **Parameters:**
    * `base_min`: Safety distance (10–30m).
    * `alpha`: Curvature weight (higher curvature = larger offset).
    * `Offset`: Default +Y (right). Set negative for left.

#### 5. MAVLink / ROS Integration
Converts `uav_waypoints.json` for flight controllers.
* **Supports:** PX4 Mission items, ROS2 `nav_msgs/Path`, Nav2 FollowPath.

#### 6. `UAV_simu`: 3D Simulation
An Open3D-based visual simulation for real-time UAV movement debugging.

---

## 📚 References

1.  Cheng Y, Chen Z, Liu D. *PL-UNeXt: per-stage edge detail and line feature guided segmentation for power line detection*. ICIP 2023.
2.  Abdelfattah R, Wang X, Wang S. *Plgan: Generative adversarial networks for power-line segmentation in aerial images*. IEEE TIP 2023.
3.  Choi H, Koo G, Kim B J, et al. *Real-time power line detection network using visible light and infrared images*. IVCNZ 2019.
4.  Zhang S, Zhang X, Ren W, et al. *Bringing RGB and IR Together: Hierarchical Multi-Modal Enhancement for Robust Transmission Line Detection*. arXiv 2025.
5.  Choi H, Yun J P, Kim B J, et al. Attention-Based Multimodal Image Feature Fusion Module for Transmission Line Detection. IEEE Transactions on Industrial Informatics 2022.
