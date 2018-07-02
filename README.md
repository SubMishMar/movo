# Monocular Visual Odometry using Epipolar Geometry

![Alt text](docs/screenshot.png?raw=true "Screenshot during VO operation")

# Usage instruction
Download:
```git clone https://github.com/SubMishMar/movo.git```

build and execute:
```cd movo
mkdir build
cd build
cmake ..
make -j7
./movo ../datasets/kitti/image_0 ../config/kitti_stereo_calib.yaml
```


