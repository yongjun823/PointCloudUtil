from concurrent.futures import ProcessPoolExecutor
import numpy as np
import open3d as o3d
from tqdm import tqdm
import random

# full model size: 2048
# partial model size: 512

# add coordinate point & color
def append_coordinate(points, colors=None, ff=-0.1):
    if colors is None:
        colors = np.zeros_like(points)
    
    x_arr = [[x, 0, 0] for x in  np.linspace(0, ff, num=100)]
    x_color = [[1, 0, 0]] * len(x_arr)

    y_arr = [[0, y, 0] for y in  np.linspace(0, 0.1, num=100)]
    y_color = [[0, 1, 0]] * len(y_arr)
    
    z_arr = [[0, 0, z] for z in  np.linspace(0, 0.1, num=100)]
    z_color = [[0, 0, 1]] * len(z_arr)
    
    new_points = np.concatenate((points, x_arr, y_arr, z_arr))
    colors = np.concatenate((colors,
                             x_color, y_color, z_color))
    
    return new_points, colors

# get distance from a
def straight_dist(c, a, b):
    ab = b - a
    ac = c - a

    area = np.linalg.norm(np.cross(ab, ac))
    d = area / np.linalg.norm(ab)

    return d

# calculate partial point cloud
def process_func(data):
    idx, p = data[0][0], data[0][1]
    points = data[1]
    camera_point = data[2]
    
    # point distance
    d = 0.005
    
    # Calculate distance from one point to all points
    distances = [(straight_dist(x, camera_point, p), x, idx) for idx, x in enumerate(points)]
    distances = filter(lambda x: x[0] < d, distances)

    c_distance = [(np.linalg.norm(camera_point - p), p, idx) for _, p, idx in distances]
    c_distance = sorted(c_distance, key=lambda x: x[0])
    
    min_idx = c_distance[0][2]
    
    return min_idx


def view_partial(points, camera_point):
    idx_arr = []
    pbar = tqdm(total=len(points))
    
    with ProcessPoolExecutor() as executor:
        for min_idx in executor.map(process_func, zip(
            enumerate(points),
            [points] * len(points),
            [camera_point] * len(points)
        )):
            pbar.update(1)
            idx_arr.append(min_idx)
    
    partial_idx = list(set(idx_arr))
    
    if len(partial_idx) < 512:
        partial_idx += [0] * (512 - len(partial_idx))
    else:
        random.shuffle(partial_idx)
        partial_idx = partial_idx[:512]
    
    return partial_idx
    
def main():
    camera_point = np.array([[0, 0, 0.2]])    
    
    # read random selected point cloud
    pcd = o3d.io.read_point_cloud("./data/test7_4.pcd")
    points = np.asarray(pcd.points)
    
    center = np.array([np.mean(points[:, 0]),
                       np.mean(points[:, 1]),
                       np.mean(points[:, 2])])
    
    center[2] += 0.4
    
    partial_idx = view_partial(points, center)
    partial_points = points[partial_idx]
    
    pcd = o3d.geometry.PointCloud()
    
    result_points = np.concatenate((
        partial_points, # partial
        points, # full
        camera_point # camera 
    ))
    
    colors = np.concatenate((
        [[1., 0., 0.]] * len(partial_points), # partial
        np.zeros_like(points), # full
        [[0, 1., 0.]] # camera 
    ))
    
    points, colors = append_coordinate(result_points, colors)
    
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.visualization.draw_geometries([pcd])
    
    # save partial point cloud
    partial_pcd = o3d.geometry.PointCloud()
    partial_pcd.points = o3d.utility.Vector3dVector(partial_points)
    o3d.io.write_point_cloud("partial/partial.pcd", partial_pcd)
    o3d.io.write_point_cloud("partial/partial.xyz", partial_pcd)

    
if __name__ == "__main__":
    main()
    