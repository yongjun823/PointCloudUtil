# PointCloud utility code

Provides Python code for the formulas used primarily in 3D vision. <br/>
All code is based on numpy and open3d. <br/>
This code rotates the point cloud or calculates a partial point.<br/>

Point Cloud & Image data collection program (Intel realsense d415) <br/>

## Feature
* point cloud viewer (with xyz coordinate)
* rotation by degree (x, y, z)
* get x, y, z degree from rotation matrix
* get transformation matrix (rotation + translation)
* get partial point (viewable point at camera view point)
* sample point
* save point
* intel realsense data collect - save image & point cloud (same resolution)
