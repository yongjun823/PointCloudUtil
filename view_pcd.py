import open3d as o3d
import numpy as np

def append_coordinate(points, colors=None):
    if colors is None:
        colors = np.zeros_like(points)
    
    x_arr = [[x, 0, 0] for x in  np.linspace(0, 0.1, num=100)]
    x_color = [[1, 0, 0]] * len(x_arr)

    y_arr = [[0, y, 0] for y in  np.linspace(0, 0.1, num=100)]
    y_color = [[0, 1, 0]] * len(y_arr)
    
    z_arr = [[0, 0, z] for z in  np.linspace(0, 0.1, num=100)]
    z_color = [[0, 0, 1]] * len(z_arr)
    
    new_points = np.concatenate((points, x_arr, y_arr, z_arr))
    colors = np.concatenate((colors,
                             x_color, y_color, z_color))
    
    return new_points, colors


def main():
    pcd = o3d.io.read_point_cloud("./partial/partial.pcd")
    points = np.asarray(pcd.points)
    
    points, colors = append_coordinate(points)
    
    # Initiailize open3d PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
    
    # save point cloud
    # pcd & xyz format
    o3d.io.write_point_cloud("test.pcd", pcd)
    o3d.io.write_point_cloud("test.xyz", pcd)
    
    
if __name__ == "__main__":
    main()
    