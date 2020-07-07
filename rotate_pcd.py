import open3d as o3d
import numpy as np
from utils import rotate_points, append_coordinate, view_points

def main():
    pcd = o3d.io.read_point_cloud("./data/test7_4.pcd")
    points = np.asarray(pcd.points)
    
    # rotate point cloud with degree
    # points, x, y, z
    points = rotate_points(points, 30, 30, 30)
    
    points, colors = append_coordinate(points)
    
    view_points(points, colors)
    
    
if __name__ == "__main__":
    main()
    