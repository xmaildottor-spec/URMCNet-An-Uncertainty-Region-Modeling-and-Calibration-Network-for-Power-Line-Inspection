# URMCNet: An Uncertainty Region Modeling and Calibration Network for Power Line Inspection

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-VITLD%20%7C%20TTPLA-green)](https://github.com/xmaildottor-spec/An-Uncertainty-Region-Modeling-and-Calibration-Network)

> **Official implementation of URMCNet.**
> This repository contains the code for *An Uncertainty Region Modeling and Calibration Network for Power Line Inspection*, including the updated training pipeline and a newly added Stereo 3D Reconstruction & UAV Path Planning module.

## 📖 Abstract

Accurate power line detection is critical for unmanned aerial vehicle (UAV)-based inspection systems. However, detection accuracy is often compromised by visually complex backgrounds (similar textures, environmental interference), leading to substantial false positives and false negatives. 

To address this, we propose **URMCNet**, featuring:
* **Uncertainty Region Modeling and Calibration Strategy:** Explicitly models and calibrates regions prone to errors.
* **Primary Feature Calibration Mechanism:** Incorporates False-Positive Region Suppression and False-Negative Region Compensation modules to adaptively calibrate uncertain regions.
* **High-Frequency Aware Fusion Decoder:** Effectively restores fine-grained details to guarantee the continuity of prediction results.

Experimental results on **VITLD** and **TTPLA** datasets demonstrate that URMCNet outperforms state-of-the-art methods.

## 🏗️ Architecture
![Process Model](https://github.com/xmaildottor-spec/An-Uncertainty-Region-Modeling-and-Calibration-Network/blob/main/main.png)


## 🎯 Motivation & Application

**Note to Practitioners:**
UAV-based inspection is crucial for smart grids but fails in complex environments where power lines are obscured by trees or buildings. Our approach enhances reliability by:
1.  Focusing on areas where prediction errors are most likely (Uncertainty Modeling).
2.  Restoring fine details to ensure continuous detection lines.

This technique is applicable not only to power lines but also to environmental monitoring, infrastructure inspection, and agricultural analysis.

## 📂 Visualization

Visualization results for the models on the TTPLA dataset are available via the Baidu Netdisk links below.

| Model Variant | Access Link | Extraction Code | Description |
| :--- | :--- | :---: | :--- |
| 🚀 **Model-B** | [Download 🔗](https://pan.baidu.com/s/1Ie431w52Sk3Cc-1tobDCJg?pwd=sdh6) | `sdh6` | TTPLA Prediction Results |
| 🌟 **Model-L** | [Download 🔗](https://pan.baidu.com/s/1yYp9tqTd6nPnhGLxaLq8MA?pwd=z7hu) | `z7hu` | TTPLA Prediction (Star version) |

> [!TIP]
> **Note:** After clicking the link, please enter the 4-digit **Extraction Code** to access the files.


## 📂 Datasets

We evaluate our method on the following datasets:

| Dataset | Link | Description |
| :--- | :--- | :--- |
| **VITLD** | [Download Link](https://bit.ly/3FBYjBY) | The Visible-Infrared Transmission Line Detection dataset. |
| **TTPLA** | [Download Link](https://drive.usercontent.google.com/download?id=1Yz59yXCiPKS0_X4K3x9mW22NLnxjvrr0&export=download&authuser=0) | Transmission Tower and Power Line Aerial images. |

### VITL Dataset Details
The organized VITL dataset is available here:
* **Link:** [Baidu Drive](https://pan.baidu.com/s/1gmfbENIXuLGKtdlqYTqrdA)
* **Password:** `r8zw`

**File Structure:**
* `seed0`, `seed1`: Original datasets.
* `seed2`: Variant where IR images are converted to PNG for visualization.
* `seed2_merge`: Merged (stitched) image patches from the original dataset.
* `fn_GT` / `fp_GT`: Ground-truth labels for False Negative/Positive samples.

> **⚠️ Important Notice:**
> 1.  **Data Grouping:** In the original VITLD dataset, **every four images correspond to one sliced sample**. During testing, ensure images are processed in groups of four in their original order.
> 2.  **Infrared Images:** IR images in this repo are for **visualization purposes only** and are not used in the network training/testing phases.
> 3.  **Preprocessing:** We recommend applying **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to input images to highlight texture details.
> 4.  **metrics:** Although binary classification performance is conventionally evaluated using foreground-specific metrics, we adopt class-macro-averaged metrics (encompassing both foreground and background) for fair and consistent comparisons with prior literature [1][3][4][5]. Additionally, our reported metrics are averaged over multiple independent trials to demonstrate model stability.


## 💾 Trained Weights

We provide the pre-trained weights for **Model-B** and **Model-L** trained on the power equipment dataset. All checkpoints are hosted on Baidu Netdisk.

| Model Variant | Checkpoint (Baidu Netdisk) | Extraction Code | Description |
| :--- | :--- | :---: | :--- |
| 🚀 **Model-B** | [Download 🔗](https://pan.baidu.com/s/1cLHUeRUykt1wqWcyeRE3TQ?pwd=g2ap) | `g2ap` | **Base version**: Best performance for high-accuracy requirements. |
| 🌟 **Model-L** | [Download 🔗](https://pan.baidu.com/s/1Uif9aNf7yGEWpG3RZ5WjDg?pwd=kuak) | `kuak` | **Large version**: Balanced performance and efficiency. |

---

### 🛠️ How to Use

1. **Download:** Get the `.pth` (or `.onnx`) files from the links above.
2. **Directory:** Place the downloaded weights into the `./checkpoints` folder of this project:
   ```bash
   mkdir checkpoints
   # Move the downloaded files here

## 🚀 Training 
Following the original publication, we refined the training pipeline to achieve State-of-the-Art (SOTA) results. Through meticulous hyperparameter tuning, the configuration settings below enable the model to achieve performance metrics that surpass those reported in the original paper.
### Optimized Hyperparameters
* **Optimizer:** AdamW (Betas: 0.9, 0.999)
* **Learning Rate:** Initial LR `3e-4` with Polynomial Decay scheduler (`power=0.9`).
* **Regularization:** Weight decay `0.005`.

## 📚 Corrigendum
Due to the inadvertent submission of a draft version during the editing process, the following corrections are made:
![Network Architecture](https://github.com/xmaildottor-spec/An-Uncertainty-Region-Modeling-and-Calibration-Network/blob/main/IMG.png)

## 🙏 Acknowledgement

We thank [Multimodal-FFM-TLD](https://github.com/hyeyeon08/Multimodal-FFM-TLD) for providing relevant training/testing data and associated code.

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
