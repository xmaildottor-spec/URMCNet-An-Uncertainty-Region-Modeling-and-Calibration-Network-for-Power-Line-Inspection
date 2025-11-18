import cv2
import numpy as np
import matplotlib.pyplot as plt
from pl_3d_localization import (
    skeletonize_mask, group_pixels_by_connected_components,
    per_row_representative, align_rows_and_compute_disparity
)


maskL = cv2.imread("left_mask.png", 0)
maskR = cv2.imread("right_mask.png", 0)

if maskL is None or maskR is None:
    print("Error: 请确保 left_mask.png 和 right_mask.png 在当前目录！")
    exit()

# Skeleton
skL = skeletonize_mask(maskL)
skR = skeletonize_mask(maskR)

# groups
groupsL = group_pixels_by_connected_components(skL)
groupsR = group_pixels_by_connected_components(skR)

if len(groupsL) == 0 or len(groupsR) == 0:
    print("No connected wires found!")
    exit()

# we take the largest group in each mask
gL = max(groupsL, key=lambda g: len(g))
gR = max(groupsR, key=lambda g: len(g))

# per-row mean points
repL, rowsL = per_row_representative(gL)
repR, rowsR = per_row_representative(gR)

# row matching
matched_rows, disparities, xL_list, xR_list, y_list = align_rows_and_compute_disparity(
    repL, repR, rowsL, rowsR, y_tolerance=1
)

# ================================
# 可视化左右图 + 匹配点连线
# ================================
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# 左图
ax[0].imshow(maskL, cmap='gray')
ax[0].set_title("Left Mask + Matched Points")
ax[0].scatter(xL_list, y_list, s=10, c='yellow')

# 右图
ax[1].imshow(maskR, cmap='gray')
ax[1].set_title("Right Mask + Matched Points")
ax[1].scatter(xR_list, y_list, s=10, c='yellow')

# 为了更清楚展示对应关系，我们将左右图放入同一个坐标系
# 在新图中拼接左右图，然后绘制连线
H, W = maskL.shape
canvas = np.zeros((H, W*2), dtype=np.uint8)
canvas[:, :W] = maskL
canvas[:, W:] = maskR

plt.figure(figsize=(14, 6))
plt.imshow(canvas, cmap='gray')
plt.title("Left–Right Match Visualization (Connected Lines)")

# 在拼接图上画点和连线
for xl, xr, yl in zip(xL_list, xR_list, y_list):
    plt.scatter(xl, yl, c='red', s=8)       # left point
    plt.scatter(xr + W, yl, c='cyan', s=8)  # right point
    plt.plot([xl, xr + W], [yl, yl], 'y-', linewidth=0.5)  # connecting line

plt.show()
