import numpy as np
import os
import open3d as o3d

# sample in point cloud
# return point index
def sample_points(points, point_num):
    idx = np.random.choice(range(len(points)), point_num, replace=False)
    
    return idx

# view point cloud & color
# with coordinate
def view_points(points, colors=None):
    if colors is None:
        colors = np.zeros_like(points)
    
    points, colors = append_coordinate(points, colors)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.visualization.draw_geometries([pcd])
    
# save point cloud .pcd format
def save_points(name, points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if not colors is None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.io.write_point_cloud(name, pcd)    


def rotate(s, theta=0, axis='x'):
    """
    Counter Clock wise rotation of a vector s, along the axis by angle theta
    s:= array/list of scalars. Contains the vector coordinates [x,y,z]
    theta:= scalar, <degree> rotation angle for counterclockwise rotation
    axis:= str, rotation axis <x,y,z>
    """
    theta = np.radians(theta) # degree -> radians
    r = 0
    if axis.lower() == 'x':
        r = [s[0],
             s[1]*np.cos(theta) - s[2]*np.sin(theta),
             s[1]*np.sin(theta) + s[2]*np.cos(theta)]
    elif axis.lower() == 'y':
        r = [s[0]*np.cos(theta) + s[2]*np.sin(theta),
             s[1],
             -s[0]*np.sin(theta) + s[2]*np.cos(theta)]
    elif axis.lower() == 'z':
        r = [s[0] * np.cos(theta) - s[1]*np.sin(theta),
             s[0] * np.sin(theta) + s[1]*np.cos(theta),
             s[2]]
    else:
        print("Error! Invalid axis rotation")
        
    return r

# rotate point x, y, z degree
def rotate_points(points, x=0, y=0, z=0):
    point_arr = []
    
    for point in points:
        point = rotate(point, x, 'x')
        point = rotate(point, y, 'y')
        point = rotate(point, z, 'z')
        
        point_arr.append(point)
    
    points = np.array(point_arr)
    
    return points

def batch_rotation(batch, x, y, z):
    temp_batch = []
    
    for points in batch:
        temp_points = rotate_points(points, x, y, z)
        temp_batch.append(temp_points)
    
    temp_batch = np.array(temp_batch)
    
    return temp_batch
            
# normalize point cloud
def normalize_vector(arr):
    value = (min(arr), max(arr))
    f = lambda x: (x - value[0]) / (value[1] - value[0])
    normalized_arr = np.array([f(x) for x in arr])
    
    return normalized_arr

# get seta x,y,z from rotation matrix
def get_seta_rotation(rotation):
    seta_x = np.rad2deg(np.arctan2(rotation[2, 1], rotation[2, 2]))
    seta_y = np.rad2deg(np.arctan2(-rotation[2, 0], 
                                   np.sqrt(np.power(rotation[2, 1], 2) + np.power(rotation[2, 2], 2))))
    seta_z = np.rad2deg(np.arctan2(rotation[1, 0], rotation[0, 0]))
    
    return seta_x, seta_y, seta_z

# add coordinate point & color
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

# rotation function of x y z
def rotation_x(seta):
    r = np.identity(4)
    rad = np.deg2rad(seta)
    sin, cos = np.sin(rad), np.cos(rad)
    
    r[1, 1] = cos
    r[1, 2] = sin
    r[2, 1] = -sin
    r[2, 2] = cos
    
    return r

def rotation_y(seta):
    r = np.identity(4)
    rad = np.deg2rad(seta)
    sin, cos = np.sin(rad), np.cos(rad)
    
    r[0, 0] = cos
    r[0, 2] = -sin
    r[2, 0] = sin
    r[2, 2] = cos
    
    return r

def rotation_z(seta):
    r = np.identity(4)
    rad = np.deg2rad(seta)
    sin, cos = np.sin(rad), np.cos(rad)
    
    r[0, 0] = cos
    r[0, 1] = -sin
    r[1, 0] = sin
    r[1, 1] = cos
    
    return r

# make transform matrix 
# translation & rotation
def get_transform(tx, ty, tz,
                  rx, ry, rz):
    rotation = np.matmul(np.matmul(rotation_x(rx), rotation_y(ry)), rotation_z(rz))

    for idx, t in enumerate([tx, ty, tz]):
        rotation[idx, 3] = t
    
    return rotation
