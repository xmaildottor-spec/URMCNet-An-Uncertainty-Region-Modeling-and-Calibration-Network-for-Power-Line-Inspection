import cv2
import numpy as np
import matplotlib.pyplot as plt

from pl_3d_localization import (
    skeletonize_mask,
    group_pixels_by_connected_components,
    per_row_representative,
    align_rows_and_compute_disparity
)


maskL = cv2.imread("left_mask.png", 0)
maskR = cv2.imread("right_mask.png", 0)

if maskL is None or maskR is None:
    print("Error: 请确保 left_mask.png 和 right_mask.png 在当前目录！")
    exit()

# Skeleton
skL = skeletonize_mask(maskL)
skR = skeletonize_mask(maskR)

# 连通域
groupsL = group_pixels_by_connected_components(skL)
groupsR = group_pixels_by_connected_components(skR)

if len(groupsL) == 0 or len(groupsR) == 0:
    print("Error")
    exit()

# 选择最长线段
gL = max(groupsL, key=lambda g: len(g))
gR = max(groupsR, key=lambda g: len(g))

# Per-row 均值点
repL, rowsL = per_row_representative(gL)  # dict, list
repR, rowsR = per_row_representative(gR)

rowsL = np.array(rowsL)
rowsR = np.array(rowsR)

# 匹配
matched_rows, disparities, xL_list, xR_list, y_list = align_rows_and_compute_disparity(
    repL, repR, rowsL, rowsR, y_tolerance=1
)

matched_rows = np.array(matched_rows)  # Nx2 形式 (y_left, y_right)

# ================================
# 计算未匹配行
# ================================
matched_L = matched_rows[:, 0] if len(matched_rows) > 0 else []
matched_R = matched_rows[:, 1] if len(matched_rows) > 0 else []

unmatched_L = np.setdiff1d(rowsL, matched_L)
unmatched_R = np.setdiff1d(rowsR, matched_R)

# 未匹配点坐标
unmatched_xL = np.array([repL[y] for y in unmatched_L]) if len(unmatched_L) > 0 else np.array([])
unmatched_yL = unmatched_L

unmatched_xR = np.array([repR[y] for y in unmatched_R]) if len(unmatched_R) > 0 else np.array([])
unmatched_yR = unmatched_R

# ================================
# 可视化 - 左右图
# ================================
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# 左图
ax[0].imshow(maskL, cmap='gray')
ax[0].set_title("Left Mask: Matched (yellow) & Unmatched (red)")
ax[0].scatter(xL_list, y_list, s=10, c='yellow', label="matched")
if len(unmatched_xL) > 0:
    ax[0].scatter(unmatched_xL, unmatched_yL, s=10, c='red', label="unmatched")
ax[0].legend()

# 右图
ax[1].imshow(maskR, cmap='gray')
ax[1].set_title("Right Mask: Matched (yellow) & Unmatched (red)")
ax[1].scatter(xR_list, y_list, s=10, c='yellow', label="matched")
if len(unmatched_xR) > 0:
    ax[1].scatter(unmatched_xR, unmatched_yR, s=10, c='red', label="unmatched")
ax[1].legend()

plt.show()

# ================================
# 拼接图 + 连线
# ================================
H, W = maskL.shape
canvas = np.zeros((H, W*2), dtype=np.uint8)
canvas[:, :W] = maskL
canvas[:, W:] = maskR

plt.figure(figsize=(16, 6))
plt.imshow(canvas, cmap='gray')
plt.title("Matched & Unmatched Points (Connected Lines for Matched)")

# 匹配成功：连线
for xl, xr, yl in zip(xL_list, xR_list, y_list):
    plt.scatter(xl, yl, c='yellow', s=10)
    plt.scatter(xr + W, yl, c='yellow', s=10)
    plt.plot([xl, xr + W], [yl, yl], 'yellow', linewidth=0.6)

# 左右未匹配点（红色）
plt.scatter(unmatched_xL, unmatched_yL, c='red', s=10)
plt.scatter(unmatched_xR + W, unmatched_yR, c='red', s=10)

plt.show()
