# Motion-tracking-SLAM-autonomous-systems
End-to-end motion estimation and SLAM pipeline

# Probabilistic Motion Tracking & SLAM for Autonomous Systems 

This project implements foundational techniques for autonomous navigation and vehicle tracking, including object localization, Kalman filtering, and SLAM (Simultaneous Localization and Mapping). The goal is to understand how self-driving vehicles model motion, estimate position, and construct maps of unknown environments using probabilistic and mathematical techniques.

## 🧠 Core Concepts Covered

- **Optical Flow & Motion Representation**  
  Track objects across time using motion models.

- **Bayesian Filters & Robot Localization**  
  Estimate a robot's position over time under uncertainty.

- **2D Histogram Filters (Mini Project)**  
  Implemented sense and move functions with probabilistic occupancy mapping.

- **1D Kalman Filter**  
  Designed a 1D tracker with Gaussian updates and predictions.

- **State Representation & Matrix Transformations**  
  Use linear algebra to represent and evolve robot state vectors.

- **Multivariate Kalman Filters**  
  Matrix operations for multidimensional state estimation.

- **SLAM (Simultaneous Localization and Mapping)**  
  Integrate motion models and landmark detection to build maps while localizing.

## Implemented

- **2D Histogram Filter**  
  Probabilistic map update using sensing and movement models.

- **Kalman Filter (1D & 2D)**  
  Recursive estimation using Gaussian distributions.

- **SLAM from Scratch**  
  Track moving agents and simultaneously build a landmark map using sensor data.


## 📁 Structure

```bash
motion-tracking-slam-autonomous-systems/
│
├── 01_motion/              # Optical flow and motion intro
├── 02_robot_localization/        # Bayesian filtering & 2D histogram filter
├── 03_kalman_filter/             # 1D Kalman Filter implementation
├── 04_state_representation/      # State vectors and motion modeling
├── 05_multidim_kalman/           # Matrices and multidimensional filtering
├── 06_slam/                      # SLAM implementation
├── 07_landmark_tracking/         # SLAM + landmark detection
├── utils/                        # Reusable plotting, matrix ops, and helpers
└── README.md
```