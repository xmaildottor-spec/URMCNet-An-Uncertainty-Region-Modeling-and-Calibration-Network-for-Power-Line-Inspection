import numpy as np
import open3d as o3d
from scipy.interpolate import splprep, splev
import time
import os

def create_open3d_animation(powerline_file, uav_path_file):
    
    if not os.path.exists(powerline_file) or not os.path.exists(uav_path_file):
        print("error")
        return

    powerline_pts = np.load(powerline_file)
    uav_waypoints = np.load(uav_path_file)
    
    if len(uav_waypoints) < 3:
        k_spline = 1 
    elif len(uav_waypoints) == 3:
        k_spline = 2 
    else:
        k_spline = 3 
    

    try:
        tck, u = splprep(uav_waypoints.T, s=0, k=k_spline)
    except TypeError as e:
        print(f"error")
        raise e
        
    num_frames = 100 
    u_fine = np.linspace(0, 1, num_frames)
    smooth_path_points = np.array(splev(u_fine, tck)).T
    
  
    powerline_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(powerline_pts),
        lines=o3d.utility.Vector2iVector(
            [[i, i + 1] for i in range(len(powerline_pts) - 1)]
        )
    )
    powerline_line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(powerline_pts) - 1)]) 

  
    path_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(smooth_path_points),
        lines=o3d.utility.Vector2iVector(
            [[i, i + 1] for i in range(len(smooth_path_points) - 1)]
        )
    )
    path_line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(smooth_path_points) - 1)])

 
    uav_radius = 0.5 
    uav_model = o3d.geometry.TriangleMesh.create_sphere(radius=uav_radius)
    uav_model.paint_uniform_color([0, 0, 1])
    uav_model.translate(smooth_path_points[0], relative=False)



    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='UAV Flight Animation (Open3D)', width=1024, height=768)

    vis.add_geometry(powerline_line_set)
    vis.add_geometry(path_line_set)
    vis.add_geometry(uav_model)

    view_control = vis.get_view_control()
    scene_center = powerline_line_set.get_axis_aligned_bounding_box().get_center()
    view_control.set_lookat(scene_center)
    view_control.set_front([-0.7, -0.7, 0.7]) 
    view_control.set_up([0, 0, 1])           
    view_control.set_zoom(0.8)

    frame_delay_seconds = 0.05
    frame_counter = 0

    print("[Q] quit")
    
  
    while vis.poll_events():
        
     
        if not vis.poll_events(): 
             break
        
        current_point = smooth_path_points[frame_counter % num_frames]
        
      
        current_uav_center = uav_model.get_center()
        translation_vector = current_point - current_uav_center
        uav_model.translate(translation_vector, relative=True)
        
        vis.update_geometry(uav_model)
        vis.update_renderer()

        frame_counter += 1
        if frame_counter >= num_frames:
            frame_counter = 0
        
        time.sleep(frame_delay_seconds)

    vis.destroy_window()
    print("quit")

if __name__ == "__main__":
    create_open3d_animation("fitted_curve.npy", "offset_curve.npy")