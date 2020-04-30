# Monocular Visual Odometry
This package is a simple implementation of a complete Monocular Visual Odonetry pipeline with the most essential features, viz. initialization of 3D features, feature tracking between two frames, pose estimation using 2D<->3D correspondences and triangulation of new landmarks.
## Videos
<a href="https://www.youtube.com/embed/t6wC1vPhBfQ" target="_blank"><img src="http://img.youtube.com/vi/t6wC1vPhBfQ/0.jpg" 
alt="offroad" width="320" height="240" border="10" /></a>
<a href="https://www.youtube.com/embed/3ZPp9PxQwT0" target="_blank"><img src="http://img.youtube.com/vi/3ZPp9PxQwT0/0.jpg" 
alt="kitti" width="320" height="240" border="10" /></a>

## Datasets
A bag file from the kitti dataset can be found [here](https://drive.google.com/a/tamu.edu/file/d/1qLUkfHIDTzxIVfaOFqIsrVKcxPlSz3_f/view?usp=sharing)

## Launch Instructions

For kitti dataset

```
  roslaunch odometry odometry_kitti_node.launch
```

For Basler dataset
```
  roslaunch odometry odometry_pylon_node.launch
```
