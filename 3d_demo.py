# ==============================================================
#  Power Line 3D Localization Demo
# ==============================================================

import cv2
import numpy as np
import json
from skimage.morphology import skeletonize
from scipy.ndimage import binary_fill_holes
from scipy.interpolate import UnivariateSpline
from visualize_point import visualize_pointcloud_matplotlib
from fit_powerline_spline import fit_powerline_spline, visualize_3d_curve

# --------------------------------------------------------------
#  Skeleton & segmentation
# --------------------------------------------------------------
def skeletonize_mask(mask):
    mask_bin = (mask > 0).astype(np.uint8)
    filled = binary_fill_holes(mask_bin).astype(np.uint8)
    sk = skeletonize(filled > 0)
    return sk.astype(np.uint8)


def group_pixels_by_cc(skel):
    num, labels = cv2.connectedComponents(skel.astype(np.uint8), connectivity=8)
    groups = []
    for i in range(1, num):
        ys, xs = np.where(labels == i)
        coords = np.stack([xs, ys], axis=1)
        coords = coords[np.argsort(coords[:, 0])]
        groups.append(coords)
    return groups


def per_row_representative(points):
    row_dict = {}
    for x, y in points:
        row_dict.setdefault(int(y), []).append(float(x))
    rep = {y: np.mean(xs) for y, xs in row_dict.items()}
    rows = sorted(rep.keys())
    return rep, rows


def align_and_disparity(repL, repR, rowsL, rowsR, y_tol=1):
    disparities = []
    xL_list = []
    xR_list = []
    y_list = []
    for y in rowsL:
        candidates = [yr for yr in rowsR if abs(yr - y) <= y_tol]
        if len(candidates) == 0:
            continue
        yr = min(candidates, key=lambda v: abs(v - y))
        xL = repL[y]
        xR = repR[yr]
        disparities.append(xL - xR)
        xL_list.append(xL)
        xR_list.append(xR)
        y_list.append(float(y))
    return np.array(disparities), np.array(xL_list), np.array(xR_list), np.array(y_list)


# --------------------------------------------------------------
#  Triangulation
# --------------------------------------------------------------
def disparity_to_3d(xL, yL, d, K, baseline):
    fx = K[0, 0]
    cx = K[0, 2]
    cy = K[1, 2]
    eps = 1e-6
    Z = (fx * baseline) / (d + eps)
    X = (xL - cx) * Z / fx
    Y = (yL - cy) * Z / fx
    pts = np.stack([X, Y, Z], axis=1)
    return pts


def smooth_points(pts, s=1.5):
    if pts.shape[0] < 5:
        return pts
    t = np.arange(len(pts))
    xs = UnivariateSpline(t, pts[:, 0], s=s)(t)
    ys = UnivariateSpline(t, pts[:, 1], s=s)(t)
    zs = UnivariateSpline(t, pts[:, 2], s=s)(t)
    return np.stack([xs, ys, zs], axis=1)


# --------------------------------------------------------------
#  Waypoint generation
# --------------------------------------------------------------
def generate_waypoints(points_b, side_offset=3.0, step_x=4.0):
    if len(points_b) == 0:
        return []
    pts = points_b[np.argsort(points_b[:, 0])]
    waypoints = []
    last_x = pts[0, 0] - 2 * step_x

    for p in pts:
        if p[0] - last_x >= step_x:
            wp = p.copy()
            wp[1] += side_offset
            waypoints.append(wp.tolist())
            last_x = p[0]
    return waypoints


# --------------------------------------------------------------
#  Main Pipeline
# --------------------------------------------------------------
def simple_pipeline_from_masks(maskL, maskR, K, baseline, R_cb=None, T_cb=None):
    if R_cb is None:
        R_cb = np.eye(3)
    if T_cb is None:
        T_cb = np.zeros(3)

    skL = skeletonize_mask(maskL)
    skR = skeletonize_mask(maskR)

    groupsL = group_pixels_by_cc(skL)
    groupsR = group_pixels_by_cc(skR)

    if len(groupsL) == 0 or len(groupsR) == 0:
        return {"pts_cam": np.zeros((0, 3)),
                "pts_body": np.zeros((0, 3)),
                "waypoints": []}

    # 选最长连通域
    gL = max(groupsL, key=lambda g: len(g))
    gR = max(groupsR, key=lambda g: len(g))

    repL, rowsL = per_row_representative(gL)
    repR, rowsR = per_row_representative(gR)

    d, xL, xR, y = align_and_disparity(repL, repR, rowsL, rowsR)

    valid = np.abs(d) > 0.3
    d = d[valid]
    xL = xL[valid]
    y = y[valid]

    if len(d) == 0:
        return {"pts_cam": np.zeros((0, 3)),
                "pts_body": np.zeros((0, 3)),
                "waypoints": []}

    pts_cam = disparity_to_3d(xL, y, d, K, baseline)
    pts_cam = smooth_points(pts_cam)

    pts_body = (R_cb @ pts_cam.T).T + T_cb.reshape(1, 3)

    waypoints = generate_waypoints(pts_body)

    return {
        "pts_cam": pts_cam,
        "pts_body": pts_body,
        "waypoints": waypoints
    }


# --------------------------------------------------------------
#  Entry point
# --------------------------------------------------------------
if __name__ == "__main__":
    print("Loading left_mask.png and right_mask.png ...")

    maskL = cv2.imread("left_mask.png", 0)
    maskR = cv2.imread("right_mask.png", 0)

    if maskL is None or maskR is None:
        exit()

    H, W = maskL.shape

    # 相机内参与基线（你可以换成自己的）
    fx = 700.0
    fy = 700.0
    cx = W / 2.0
    cy = H / 2.0
    baseline = 0.25

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])

    print("Running 3D localization pipeline ...")

    res = simple_pipeline_from_masks(maskL, maskR, K, baseline)

    pts_cam = res["pts_cam"]
    pts_body = res["pts_body"]
    waypoints = res["waypoints"]

    print("3D 点数量:", pts_cam.shape[0])
    print("waypoints 数量:", len(waypoints))

    # 保存结果
    np.savetxt("pts_cam.txt", pts_cam)
    np.savetxt("pts_body.txt", pts_body)
    with open("waypoints.json", "w") as f:
        json.dump(waypoints, f, indent=2)

    print("pts_cam.txt, pts_body.txt, waypoints.json")

    visualize_pointcloud_matplotlib(pts_cam)   #点云图

    pts_cam = res["pts_cam"]

    curve_points = fit_powerline_spline(pts_cam) 
    visualize_3d_curve(pts_cam, curve_points) #3D曲线拟合图
