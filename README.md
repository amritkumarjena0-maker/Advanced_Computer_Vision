
# Advanced Computer Vision

## Description
This project contains eight advanced computer vision programs implemented in Python. It covers 3D reconstruction, stereo vision, deep learning, differentiable rendering, camera calibration, autonomous driving, physics-informed motion estimation, and multi-sensor fusion.

## Features
1. 3D Reconstruction using Structure from Motion
2. Stereo Vision for Depth and Distance Estimation
3. Image Anomaly Detection using an Autoencoder
4. Differentiable Rendering for 3D Scene Optimization
5. Camera Calibration and Nonlinear Bundle Adjustment
6. Autonomous Driving Perception Pipeline
7. Physics-Informed Motion Estimation
8. Multi-Sensor Fusion using RGB, LiDAR, Depth, and IMU

## Technologies
- Python
- OpenCV
- NumPy
- SciPy
- Matplotlib
- TensorFlow
- PyTorch
- Ultralytics YOLO

## Requirements
- Python programming environment
- Required input images and videos for the relevant programs
- Compatible model weights for AI-based detection

## How to Run
Run each program from the project folder.

### 1. Structure from Motion
`python structure_from_motion.py`

### 2. Stereo Vision
`python stereo_depth.py`

### 3. Image Anomaly Detection
`python image_anomaly_autoencoder.py`

### 4. Differentiable Rendering
`python differentiable_rendering.py`

### 5. Camera Calibration and Bundle Adjustment
`python camera_calibration_bundle_adjustment.py`

### 6. Autonomous Driving Perception
`python autonomous_driving_perception.py`

### 7. Physics-Informed Motion Estimation
`python physics_informed_motion.py`

### 8. Multi-Sensor Fusion
`python multi_sensor_fusion.py`

## Algorithm
1. **Structure from Motion:** Detect features, match images, estimate camera pose, and triangulate 3D points.
2. **Stereo Vision:** Calculate disparity and estimate scene depth.
3. **Anomaly Detection:** Train an autoencoder and identify unusual reconstruction errors.
4. **Differentiable Rendering:** Optimize scene parameters using gradient descent.
5. **Bundle Adjustment:** Minimize reprojection error using nonlinear least squares.
6. **Autonomous Driving:** Detect objects, segment instances, and track objects across video frames.
7. **Physics-Informed Vision:** Track image motion and fit a physical motion model.
8. **Sensor Fusion:** Combine normalized features from multiple sensor modalities.

## Flowchart
```text
START
  |
  v
Load Images, Video, or Sensor Data
  |
  v
Extract Features and Measurements
  |
  v
Apply Computer Vision Algorithms
  |
  v
Estimate 3D Geometry, Depth, Anomalies,
Motion, or Fused Features
  |
  v
Display or Save Results
  |
  v
END
```

## Note
- Use the correct input filenames specified in each Python program.
- `image1.jpg` and `image2.jpg` must show the same scene from different viewpoints.
- `left.jpg` and `right.jpg` must be a stereo image pair.
- `driving_video.mp4` is used for autonomous driving perception.
- `motion_video.mp4` is used for motion tracking.
- Some programs use simplified demonstrations and require further calibration or training for real-world accuracy.

## Author
Advanced Computer Vision Project

Amrit Kumar Jena
