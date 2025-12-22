# Dual-View ArUco Tracking & Trajectory Analysis

This repository contains a computer vision pipeline designed to track the 3D movement of a machine (e.g., 3D printer, robotic arm) using ArUco markers and verify its accuracy against a G-code reference file.

## 📂 Project Structure

The project consists of two main Python scripts:

1.  **`tracker.py`** (Code #1): Processes video feeds to extract 3D coordinates.
2.  **`visualizer.py`** (Code #2): Analyzes the extracted data, compares it to G-code, and calculates error metrics.

---

## 🛠️ Prerequisites

You will need Python 3.8+ and the following dependencies:

```bash
pip install numpy opencv-contrib-python matplotlib pandas
