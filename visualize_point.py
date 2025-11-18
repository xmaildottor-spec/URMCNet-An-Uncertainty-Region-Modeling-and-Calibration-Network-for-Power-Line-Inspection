import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_pointcloud_matplotlib(points):
    """
    Matplotlib 3D visualization
    points: Nx3 array (X,Y,Z)
    """
    if len(points) == 0:
        print("No points to visualize!")
        return
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]

    ax.scatter(X, Y, Z, s=3, c=Z, cmap='viridis')
    ax.set_xlabel("X (Forward)")
    ax.set_ylabel("Y (Right)")
    ax.set_zlabel("Z (Down)")

    ax.set_title("Power Line 3D Point Cloud (Camera/Body Frame)")
    plt.show()

