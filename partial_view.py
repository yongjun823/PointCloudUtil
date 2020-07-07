from concurrent.futures import ProcessPoolExecutor
import numpy as np
import open3d as o3d
from tqdm import tqdm
import random
from utils import append_coordinate, view_points, save_points, rotate_points

# full model size: 2048
# partial model size: 512

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
    # x y z
    camera_point = np.array([[0, 0.2, 0.2]])    
    
    # read random selected point cloud
    pcd = o3d.io.read_point_cloud("./data/test7_4.pcd")
    points = np.asarray(pcd.points)
    
    # rotate point cloud
    points = rotate_points(points, 30, 30, 30)
    
    # get viewable partial point
    partial_idx = view_partial(points, camera_point)
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
    
    view_points(points, colors)
    
    # save partial point cloud .pcd or .xyz
    save_points('partial/partial.pcd', partial_points)
    
    
if __name__ == "__main__":
    main()
    