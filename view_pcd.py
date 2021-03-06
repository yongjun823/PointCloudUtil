import open3d as o3d
import numpy as np
from utils import rotate_points, append_coordinate, view_points

def main():
    # read point cloud
    pcd = o3d.io.read_point_cloud("./partial/partial.pcd")
    points = np.asarray(pcd.points)
    
    # add color coordinate & point cloud
    points, colors = append_coordinate(points)
    
    # View Point cloud
    view_points(points, colors)
    
if __name__ == "__main__":
    main()
    