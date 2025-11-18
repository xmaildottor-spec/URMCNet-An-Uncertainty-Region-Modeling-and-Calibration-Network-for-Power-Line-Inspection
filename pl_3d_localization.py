# pl_3d_localization.py - Power Line 3D Localization utilities
import numpy as np
import cv2
from skimage.morphology import skeletonize
from scipy.ndimage import binary_fill_holes
from scipy.interpolate import UnivariateSpline
import math

def undistort_image(img, K, dist):
    """
    Undistort image using camera intrinsics and distortion coefficients.
    img: HxW or HxWx3 image
    K: 3x3 intrinsic matrix
    dist: distortion vector (k1,k2,p1,p2[,k3])
    """
    h,w = img.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 0)
    und = cv2.undistort(img, K, dist, None, newK)
    return und, newK

def skeletonize_mask(mask):
    """
    Given a binary mask (0/1 or 0/255), fill holes, thin to single-pixel skeleton.
    Returns skeleton as boolean array.
    """
    # ensure boolean
    binm = (mask > 0).astype(np.uint8)
    # fill small holes to make continuous lines
    filled = binary_fill_holes(binm).astype(np.uint8)
    # skeletonize expects boolean with True for foreground
    sk = skeletonize(filled > 0)
    return sk.astype(np.uint8)

def vertical_sample_intersections(skel, n_samples=50):
    """
    Place n_samples vertical lines (x positions uniformly across width),
    count intersections and return the x positions of sampled lines and the list of y coords where intersections occur.
    skel: HxW skeleton (0/1)
    Returns: xs (list), intersections (dict mapping x->list of y indices where skel==1)
    """
    h,w = skel.shape
    xs = np.linspace(0, w-1, n_samples, dtype=int)
    intersections = {}
    for x in xs:
        ys = np.where(skel[:, x] > 0)[0]
        intersections[int(x)] = ys.tolist()
    return xs.tolist(), intersections

def group_pixels_by_connected_components(skel):
    """
    Label connected components on skeleton and return list of pixel coordinate arrays for each component.
    Each component likely corresponds to a continuous wire segment (though crossing can split).
    """
    num, lab = cv2.connectedComponents(skel.astype(np.uint8), connectivity=8)
    groups = []
    for i in range(1, num):
        ys, xs = np.where(lab == i)
        coords = np.stack([xs, ys], axis=1)  # Nx2 (x,y)
        # sort by x (left->right)
        coords = coords[np.argsort(coords[:,0])]
        groups.append(coords)
    return groups

def pixel_tracking_from_seed(skel, seed_x, seed_y, max_step=2):
    """
    Simple BFS pixel-tracking from a seed pixel on a skeleton to collect connected pixels.
    Returns ordered list of (x,y) along the track by greedy neighbor selection (follow direction).
    """
    h,w = skel.shape
    visited = set()
    path = []
    stack = [(seed_x, seed_y)]
    while stack:
        x,y = stack.pop(0)
        if (x,y) in visited: 
            continue
        visited.add((x,y))
        path.append((x,y))
        # explore 8-neighbors
        neigh = []
        for dx in range(-1,2):
            for dy in range(-1,2):
                nx,ny = x+dx, y+dy
                if nx<0 or ny<0 or nx>=w or ny>=h: continue
                if skel[ny,nx] > 0 and (nx,ny) not in visited:
                    neigh.append((nx,ny))
        # push neighbors prioritizing same y then x (to follow approximate horizontal flow)
        neigh = sorted(neigh, key=lambda p: (abs(p[1]-y), p[0])) 
        stack[0:0] = neigh  # prepend to BFS
    return path

def per_row_representative(points):
    """
    points: Nx2 array of (x,y) integer pixel coordinates for a single wire
    returns dict mapping y->mean_x (float), and a sorted list of rows (ys)
    """
    if len(points)==0:
        return {}, []
    ys = points[:,1].astype(int)
    xs = points[:,0].astype(float)
    row_dict = {}
    for x,y in zip(xs,ys):
        row_dict.setdefault(int(y), []).append(float(x))
    rep = {y: float(np.mean(row_dict[y])) for y in row_dict}
    rows = sorted(rep.keys())
    return rep, rows

def align_rows_and_compute_disparity(repL, repR, rowsL, rowsR, y_tolerance=1):
    """
    Align rows by y coordinate (allowing small tolerance) and compute disparities xL - xR for matched rows.
    repL, repR: dict y->x
    rowsL, rowsR: sorted lists
    returns matched_rows, disparities, xL_list, xR_list, y_list
    """
    matched_rows = []
    disparities = []
    xL_list = []
    xR_list = []
    y_list = []
    setR = set(rowsR)
    for y in rowsL:
        # find nearest y in rowsR within tolerance
        candidates = [yr for yr in rowsR if abs(yr - y) <= y_tolerance]
        if not candidates:
            continue
        # choose nearest
        yr = min(candidates, key=lambda v: abs(v-y))
        xL = repL[y]
        xR = repR[yr]
        d = xL - xR
        matched_rows.append((y, yr))
        disparities.append(d)
        xL_list.append(xL)
        xR_list.append(xR)
        y_list.append(y)
    return matched_rows, np.array(disparities), np.array(xL_list), np.array(xR_list), np.array(y_list)

def disparity_to_points(xL, yL, disparities, K, baseline):
    """
    Triangulate per-row 3D points in camera coordinate (left camera).
    xL,yL: arrays of image coordinates (float)
    disparities: xL - xR
    K: intrinsic matrix (3x3) in pixels
    baseline: in same physical units as desired depth (meters)
    returns Nx3 array of 3D points (X,Y,Z) in left camera coordinates
    """
    fx = K[0,0]
    cx = K[0,2]
    cy = K[1,2]
    eps = 1e-6
    Z = (fx * baseline) / (disparities + eps)  # avoid division by zero; caller should filter small disparities
    X = (xL - cx) * Z / fx
    Y = (yL - cy) * Z / fx
    pts = np.stack([X, Y, Z], axis=1)
    return pts

def filter_and_smooth_points(pts, z_min=0.5, z_max=500.0, spline_s=1.0):
    """
    Filter by depth range and smooth trajectory along Z using a spline on the index.
    pts: Nx3 (X,Y,Z)
    returns filtered & smoothed pts
    """
    if len(pts)==0:
        return pts
    mask = (pts[:,2] > z_min) & (pts[:,2] < z_max)
    pts = pts[mask]
    if len(pts) < 4:
        return pts
    t = np.arange(len(pts))
    # smooth each coord with UnivariateSpline
    xs = UnivariateSpline(t, pts[:,0], s=spline_s)(t)
    ys = UnivariateSpline(t, pts[:,1], s=spline_s)(t)
    zs = UnivariateSpline(t, pts[:,2], s=spline_s)(t)
    out = np.stack([xs, ys, zs], axis=1)
    return out

def camera_to_body(pts, R_cb, T_cb):
    """
    Transform points from camera coords to body coords: p_b = R_cb * p_c + T_cb
    R_cb: 3x3 rotation (camera->body)
    T_cb: 3 vector translation (meters)
    pts: Nx3
    """
    return (R_cb @ pts.T).T + T_cb.reshape((1,3))

def body_to_world(pts_b, R_bw, T_bw):
    """
    p_w = R_bw * p_b + T_bw
    """
    return (R_bw @ pts_b.T).T + T_bw.reshape((1,3))

def generate_waypoints_from_wire(points_b, desired_side_offset=10.0, desired_along_offset=4.0):
    """
    Given smoothed wire 3D points in body coordinates, propose waypoints for UAV to follow.
    Strategy: take a sequence of sample points along the wire, offset to the right by desired_side_offset (body y-axis),
    and place waypoints spaced by desired_along_offset along body x-axis projection.
    points_b: Nx3 (X forward, Y right, Z down convention assumed)
    returns list of waypoints in body coords
    """
    if len(points_b)==0:
        return []
    # project along X axis (forward). We'll pick points at increasing X.
    pts = points_b[np.argsort(points_b[:,0])]
    # sample by along-axis spacing
    waypoints = []
    last_x = pts[0,0] - desired_along_offset*2.0
    for p in pts:
        if p[0] - last_x >= desired_along_offset:
            wp = p.copy()
            # offset to the right (+y) by side offset
            wp[1] += desired_side_offset
            waypoints.append(wp.tolist())
            last_x = p[0]
    return waypoints

def simple_pipeline_from_masks(maskL, maskR, K, baseline, distL=None, distR=None, R_cb=None, T_cb=None):
    """
    High-level pipeline that takes binary masks and returns 3D points and suggested waypoints.
    - maskL, maskR: binary masks (HxW)
    - K: intrinsic matrix (assumed same for left/right after undistort)
    - baseline: meters (distance from left to right camera along x in body frame)
    - R_cb, T_cb: camera->body extrinsics (if None, assume identity/zero)
    returns dict with keys: pts_cam (Nx3), pts_body (Nx3), waypoints (list)
    """
    if R_cb is None:
        R_cb = np.eye(3)
    if T_cb is None:
        T_cb = np.zeros(3)
    # skeletonize masks
    skL = skeletonize_mask(maskL)
    skR = skeletonize_mask(maskR)
    # group components and pick the largest components as candidate wires
    groupsL = group_pixels_by_connected_components(skL)
    groupsR = group_pixels_by_connected_components(skR)
    # for robustness, pick the longest group in each as primary (simplification)
    if len(groupsL)==0 or len(groupsR)==0:
        return {"pts_cam": np.zeros((0,3)), "pts_body": np.zeros((0,3)), "waypoints": []}
    gL = max(groupsL, key=lambda g: len(g))
    gR = max(groupsR, key=lambda g: len(g))
    repL, rowsL = per_row_representative(gL)
    repR, rowsR = per_row_representative(gR)
    matched_rows, disparities, xL_list, xR_list, y_list = align_rows_and_compute_disparity(repL, repR, rowsL, rowsR)
    # filter disparities small or nonpositive
    disparities = np.array(disparities)
    good_mask = np.abs(disparities) > 0.5  # pixel threshold
    if good_mask.sum() == 0:
        return {"pts_cam": np.zeros((0,3)), "pts_body": np.zeros((0,3)), "waypoints": []}
    xL_good = np.array(xL_list)[good_mask]
    y_good = np.array(y_list)[good_mask].astype(float)
    d_good = disparities[good_mask].astype(float)
    pts_cam = disparity_to_points(xL_good, y_good, d_good, K, baseline)
    pts_cam_s = filter_and_smooth_points(pts_cam)
    pts_body = camera_to_body(pts_cam_s, R_cb, T_cb)
    waypoints = generate_waypoints_from_wire(pts_body)
    return {"pts_cam": pts_cam_s, "pts_body": pts_body, "waypoints": waypoints}

# --- Helper wrappers re-exported for external use ---
def skeletonize_mask(mask):
    return (skeletonize((mask>0).astype(int))>0).astype(np.uint8)

def group_pixels_by_connected_components(skel):
    num, lab = cv2.connectedComponents(skel.astype(np.uint8), connectivity=8)
    groups = []
    for i in range(1, num):
        ys, xs = np.where(lab == i)
        coords = np.stack([xs, ys], axis=1)  # Nx2 (x,y)
        coords = coords[np.argsort(coords[:,0])]
        groups.append(coords)
    return groups

def per_row_representative(points):
    if len(points)==0:
        return {}, []
    ys = points[:,1].astype(int)
    xs = points[:,0].astype(float)
    row_dict = {}
    for x,y in zip(xs,ys):
        row_dict.setdefault(int(y), []).append(float(x))
    rep = {y: float(np.mean(row_dict[y])) for y in row_dict}
    rows = sorted(rep.keys())
    return rep, rows

def align_rows_and_compute_disparity(repL, repR, rowsL, rowsR, y_tolerance=1):
    matched_rows = []
    disparities = []
    xL_list = []
    xR_list = []
    y_list = []
    for y in rowsL:
        candidates = [yr for yr in rowsR if abs(yr - y) <= y_tolerance]
        if not candidates:
            continue
        yr = min(candidates, key=lambda v: abs(v-y))
        xL = repL[y]
        xR = repR[yr]
        d = xL - xR
        matched_rows.append((y, yr))
        disparities.append(d)
        xL_list.append(xL)
        xR_list.append(xR)
        y_list.append(y)
    return matched_rows, np.array(disparities), np.array(xL_list), np.array(xR_list), np.array(y_list)

def disparity_to_points(xL, yL, disparities, K, baseline):
    fx = K[0,0]
    cx = K[0,2]
    cy = K[1,2]
    eps = 1e-6
    Z = (fx * baseline) / (disparities + eps)
    X = (xL - cx) * Z / fx
    Y = (yL - cy) * Z / fx
    pts = np.stack([X, Y, Z], axis=1)
    return pts

def camera_to_body(pts, R_cb, T_cb):
    return (R_cb @ pts.T).T + T_cb.reshape((1,3))

def generate_waypoints_from_wire(points_b, desired_side_offset=10.0, desired_along_offset=4.0):
    if len(points_b)==0:
        return []
    pts = points_b[np.argsort(points_b[:,0])]
    waypoints = []
    last_x = pts[0,0] - desired_along_offset*2.0
    for p in pts:
        if p[0] - last_x >= desired_along_offset:
            wp = p.copy()
            wp[1] += desired_side_offset
            waypoints.append(wp.tolist())
            last_x = p[0]
    return waypoints

