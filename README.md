# URMCNET： An Uncertainty Region Modeling and Calibration Network for Power Line Inspection

## Abstract：
Accurate power line detection is critical for unmanned aerial vehicle (UAV)-based inspection systems. However, the detection accuracy is severely compromised by visually complex backgrounds characterized by similar textures and environmental interference, which often lead to substantial false positives and false negatives in detection results. To address these limitations, we propose a novel Uncertainty Region Modeling and Calibration Network (URMCNet), which incorporates an Uncertainty Region Modeling and Calibration Strategy that explicitly models and calibrates regions prone to false positives and false negatives, so as to enhance the accuracy of power line detection. The core component of URMCNet is the Primary Feature Calibration Mechanism, which incorporates the False-Positive Region Suppression Module and the False-Negative Region Compensation Module. These two modules adaptively calibrate uncertain regions, thereby making the feature representation more accurate and comprehensive. Furthermore, we introduce a High-Frequency Aware Fusion Decoder, which effectively restores the fine-grained details, thereby guaranteeing the continuity of the prediction results. The comprehensive experimental results demonstrate that URMCNet significantly outperforms state-of-the-art methods on several key performance metrics on the publicly available VITLD and TTPLA datasets.

## Note to Practitioners：
UAV-based inspection has become a crucial method for maintaining power grid safety and developing smart grids, especially in remote or hard-to-reach areas. However, existing detection systems often perform poorly in visually complex environments, where power lines may be obscured by trees, buildings, or other obstacles, leading to significant errors that compromise the reliability of inspections and increase maintenance costs. To address these challenges, we propose a new approach that enhances power line detection accuracy by focusing on areas where prediction errors are most likely to occur. Our method employs an adaptive calibration process that specifically targets regions susceptible to false positives and false negatives. This approach enhances detection accuracy and reliability in complex environments, thereby improving the efficiency of UAV-based inspections. Another key contribution of this research is the ability to restore fine details in the detection results, which is crucial for ensuring the continuous detection of power lines in complex or noisy environments. In addition to power line inspection, the techniques proposed can be applied to other fields that also require object detection in complex visual conditions, such as environmental monitoring, infrastructure inspection, and agricultural analysis.

## Overall process model inference:
![image](https://github.com/xmaildottor-spec/An-Uncertainty-Region-Modeling-and-Calibration-Network/blob/main/main.png)
## Dataset：
 1)  VITLD:  The dataset is available at https://bit.ly/3FBYjBY.
 3)  TTPLA:  The dataset is available at https://drive.usercontent.google.com/download?id=1Yz59yXCiPKS0_X4K3x9mW22NLnxjvrr0&export=download&authuser=0

## Notice:
In the original VITLD dataset, every four images correspond to one sliced sample. During testing, please ensure that the images are processed in groups of four and in their original order.

## Acknowledgement:
Thanks 【[Multimodal-FFM-TLD](https://github.com/hyeyeon08/Multimodal-FFM-TLD)】 for providing the relevant training and testing data as well as the associated code.

Due to a mix-up between the draft file and the final version, the errata are as follows:
![image](https://github.com/xmaildottor-spec/An-Uncertainty-Region-Modeling-and-Calibration-Network/blob/main/IMG.png)
































 
