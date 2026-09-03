import os
import json
import numpy as np
import cv2
import math
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pl_3d_localization import simple_pipeline_from_masks


# ---------------------------
# Helper: create synthetic masks if real ones not present
# ---------------------------
def make_synthetic_masks(H=240, W=320, shift=8):
    maskL = np.zeros((H,W), dtype=np.uint8)
    maskR = np.zeros_like(maskL)
    for y in range(30, H-30):
        x = int(0.5*(y-30) + 60)
        if 0 <= x < W: maskL[y,x] = 255
        xr = x - shift
        if 0 <= xr < W: maskR[y,xr] = 255
    return maskL, maskR

# ---------------------------
# Fit B-spline to points (returns sampled points, derivatives)
# ---------------------------
def fit_spline(points, smoothness=1.0, n_samples=500):
    """
    points: Nx3 array
    returns:
        curve_pts: Mx3 sampled points along spline
        ders1: Mx3 first derivative (dx/du,dy/du,dz/du)
        ders2: Mx3 second derivative
        u_fine: 1D parameter
    """
    if len(points) < 4:
        raise ValueError("点数太少，无法拟合样条")
    # sort by X (assume X is along-line axis); if noisy, sort by arc-length approx
    pts = points[np.argsort(points[:,0])]
    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    # use splprep: parameterize by u in [0,1]
    tck, u = splprep([x, y, z], s=smoothness, k=3)
    u_fine = np.linspace(0, 1, n_samples)
    x_f, y_f, z_f = splev(u_fine, tck)
    # derivatives
    dx1, dy1, dz1 = splev(u_fine, tck, der=1)
    dx2, dy2, dz2 = splev(u_fine, tck, der=2)
    curve_pts = np.vstack([x_f, y_f, z_f]).T
    ders1 = np.vstack([dx1, dy1, dz1]).T
    ders2 = np.vstack([dx2, dy2, dz2]).T
    return curve_pts, ders1, ders2, u_fine

# ---------------------------
# Compute curvature kappa from derivatives
# kappa = || r' x r'' || / ||r'||^3
# ---------------------------
def compute_curvature(d1, d2):
    # d1, d2 are Mx3
    cross = np.cross(d1, d2)
    num = np.linalg.norm(cross, axis=1)
    denom = (np.linalg.norm(d1, axis=1) ** 3) + 1e-12
    kappa = num / denom
    return kappa  # shape M

# ---------------------------
# Recommend safe side offset based on curvature and base_min
# Strategy:
#   recommended = max(base_min, base_min + alpha*(kappa_norm))
#   where kappa_norm = (kappa - kappa_min) / (kappa_max - kappa_min)
# ---------------------------
def recommend_side_offset(kappa, base_min=10.0, alpha=10.0):
    # kappa: array
    kmin, kmax = np.min(kappa), np.max(kappa)
    if kmax - kmin < 1e-12:
        return base_min
    k_norm = (kappa - kmin) / (kmax - kmin)
    # take some percentile (e.g., 90th) to be conservative
    perc90 = np.percentile(k_norm, 90)
    recommended = base_min + alpha * perc90
    return float(max(base_min, recommended))

# ---------------------------
# Generate offset path (right-side) and yaw per waypoint
# Coordinate conv: assume points_b in body frame (X forward, Y right, Z down)
# For each sampled point:
#   tangent = d1 (approx direction)
#   up = [0,0,-1] (since Z down), so global up vector is negative Z in this frame -> choose up = [0,0,-1]
#   right = normalize(cross(tangent, up))
#   offset_point = point + right * offset
# Yaw: heading angle around Z (body): atan2(tangent_y, tangent_x)
# ---------------------------
def generate_offset_waypoints(curve_pts, d1, offset_m=10.0, spacing_m=4.0):
    """
    curve_pts: Mx3, d1: Mx3 (first derivatives)
    returns:
        waypoints: list of dict {x,y,z,yaw}
        offset_curve: sampled offset curve points (Kx3)
    """
    # compute approximate arc-length along curve to sample by spacing_m
    # approximate distances between consecutive curve_pts
    seg = np.linalg.norm(np.diff(curve_pts, axis=0), axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seg)])
    total_len = cumlen[-1]
    if total_len <= 0:
        return [], curve_pts

    # desired sample positions along arc length
    s_vals = np.arange(0.0, total_len, spacing_m)
    if s_vals[-1] < total_len:
        s_vals = np.append(s_vals, total_len)

    # interpolate curve_pts to get points at s_vals
    # use linear interpolation on cumlen
    xs = np.interp(s_vals, cumlen, curve_pts[:,0])
    ys = np.interp(s_vals, cumlen, curve_pts[:,1])
    zs = np.interp(s_vals, cumlen, curve_pts[:,2])
    # also interpolate derivatives d1
    d1x = np.interp(s_vals, cumlen, d1[:,0])
    d1y = np.interp(s_vals, cumlen, d1[:,1])
    d1z = np.interp(s_vals, cumlen, d1[:,2])

    sampled_pts = np.vstack([xs, ys, zs]).T
    sampled_d1 = np.vstack([d1x, d1y, d1z]).T

    # compute offset points and yaw
    waypoints = []
    offset_curve = []
    # define up vector (body frame): since Z is down, up_vec = [0,0,-1]
    up_vec = np.array([0.0, 0.0, -1.0])

    for p, t in zip(sampled_pts, sampled_d1):
        # normalize tangent
        t_norm = np.linalg.norm(t)
        if t_norm < 1e-8:
            t_unit = np.array([1.0, 0.0, 0.0])
        else:
            t_unit = t / t_norm
        # right vector = normalize(cross(tangent, up))
        r = np.cross(t_unit, up_vec)
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-8:
            # fallback: use global right
            r_unit = np.array([0.0, 1.0, 0.0])
        else:
            r_unit = r / r_norm
        offset_pt = p + r_unit * offset_m
        # yaw: heading in body xy-plane (atan2(y, x) of tangent projection)
        yaw = math.atan2(t_unit[1], t_unit[0])
        waypoints.append({"x": float(offset_pt[0]), "y": float(offset_pt[1]), "z": float(offset_pt[2]), "yaw": float(yaw)})
        offset_curve.append(offset_pt)

    return waypoints, np.array(offset_curve)

# ---------------------------
# Visualization util
# ---------------------------
def visualize_curve_and_path(points3d, curve_pts, offset_curve, waypoints, show=True):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    # original 3D points
    if len(points3d) > 0:
        ax.scatter(points3d[:,0], points3d[:,1], points3d[:,2], s=6, c='blue', label='raw 3D pts')
    # fitted curve
    ax.plot(curve_pts[:,0], curve_pts[:,1], curve_pts[:,2], c='red', linewidth=2, label='fitted spline')
    # offset path
    if offset_curve is not None and len(offset_curve)>0:
        ax.plot(offset_curve[:,0], offset_curve[:,1], offset_curve[:,2], c='green', linewidth=2, label='offset UAV path')
    # waypoints (as markers)
    if waypoints:
        wpts = np.array([[wp['x'], wp['y'], wp['z']] for wp in waypoints])
        ax.scatter(wpts[:,0], wpts[:,1], wpts[:,2], s=30, c='orange', marker='^', label='waypoints')

    ax.set_xlabel('X (Forward)')
    ax.set_ylabel('Y (Right)')
    ax.set_zlabel('Z (Down)')
    ax.set_title('Powerline spline & UAV offset path')
    ax.legend()
    if show:
        plt.show()

# ---------------------------
# Main routine
# ---------------------------
def main():
    # load masks if exist
    if os.path.exists("left_mask.png") and os.path.exists("right_mask.png"):
        maskL = cv2.imread("left_mask.png", 0)
        maskR = cv2.imread("right_mask.png", 0)
    else:
        print("未找到 left_mask.png 或 right_mask.png，使用合成示例生成 mask。")
        maskL, maskR = make_synthetic_masks(H=240, W=320, shift=8)
        cv2.imwrite("synth_left_mask.png", maskL)
        cv2.imwrite("synth_right_mask.png", maskR)
        print("synth_left_mask.png, synth_right_mask.png")

    H, W = maskL.shape
    # example intrinsics (请替换为真实内参)
    fx = 700.0; fy = 700.0; cx = W/2.0; cy = H/2.0
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    baseline = 0.25  # meters

    # run pipeline ( pts_cam and pts_body)
    res = simple_pipeline_from_masks(maskL, maskR, K, baseline)
    pts_cam = res["pts_cam"]       # in camera coords
    pts_body = res["pts_body"]     # in body coords (we assumed identity R/T)

    if pts_body.shape[0] < 5:
        print("提取到的 3D 点过少，无法拟合。点数:", pts_body.shape[0])
        return

    # fit spline
    smoothness = 1.0
    curve_pts, d1, d2, u_fine = fit_spline(pts_body, smoothness=smoothness, n_samples=600)

    # compute curvature
    kappa = compute_curvature(d1, d2)
    # recommended side offset (meters)
    base_min = 10.0  # minimal safe distance (m) 可按工程改
    alpha = 15.0     # curvature放大因子
    recommended_offset = recommend_side_offset(kappa, base_min=base_min, alpha=alpha)
    print("基于曲率的推荐侧偏 (meters): ", recommended_offset)
    # user can choose desired_offset; prioritize recommended
    desired_offset = recommended_offset

    # generate offset waypoints (spacing along arc length)
    spacing = 4.0  # meters between waypoints
    waypoints, offset_curve = generate_offset_waypoints(curve_pts, d1, offset_m=desired_offset, spacing_m=spacing)
    print("waypoint:", len(waypoints))

    # save waypoints
    with open("uav_waypoints.json", "w", encoding="utf-8") as f:
        json.dump(waypoints, f, indent=2, ensure_ascii=False)
    np.save("fitted_curve.npy", curve_pts)
    np.save("offset_curve.npy", offset_curve)

    print("保存： uav_waypoints.json, fitted_curve.npy, offset_curve.npy")

    # visualization
    visualize_curve_and_path(pts_body, curve_pts, offset_curve, waypoints, show=True)

if __name__ == "__main__":
    main()
