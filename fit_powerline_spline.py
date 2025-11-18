import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev

def fit_powerline_spline(points, smoothness=3.0):

    if len(points) < 5:
        print("error")
        return points


    pts = points[np.argsort(points[:, 0])]

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

 
    tck, u = splprep([x, y, z], s=smoothness)

    u_fine = np.linspace(0, 1, 300)
    x_f, y_f, z_f = splev(u_fine, tck)

    curve_points = np.vstack([x_f, y_f, z_f]).T
    return curve_points


def visualize_3d_curve(points, curve_points):

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c='blue', s=10, label="Original 3D Points")

 
    ax.plot(curve_points[:, 0], curve_points[:, 1], curve_points[:, 2],
            'r-', linewidth=3, label="Fitted Powerline Spline")

    ax.set_xlabel("X (Forward)")
    ax.set_ylabel("Y (Right)")
    ax.set_zlabel("Z (Down)")

    ax.set_title("3D Power Line Curve Fitting (Spline)")
    ax.legend()
    plt.show()


