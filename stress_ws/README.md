# ROS 2 Humble Workspace

This repository contains a ROS 2 workspace designed for real-time physiological signal processing, stress inference, and data collection using EmotiBit sensors and the WESAD dataset.

## Prerequisites

* **ROS 2 Version:** [Humble](https://docs.ros.org/en/humble/Installation.html) 
* **Dependencies:** `colcon`, `rosbag2`, `std_msgs`, Python 3.10+
* **Hardware:** EmotiBit (live inference)

---

## Workspace Setup

First, initialize the environment and build the core workspace.

```bash
# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Create and build the workspace
mkdir -p ~/stress_ws/src
cd ~/stress_ws
colcon build
source install/setup.bash
```

### Create the Inference Package
Create the Python-based package:
```bash
cd ~/stress_ws/src
ros2 pkg create --build-type ament_python stress_inference_pkg
# Note: Ensure all scripts are in stress_inference_pkg/stress_inference_pkg/

colcon build --packages-select stress_inference_pkg
# Use this command, everytime a new script is added or edited in this workspace
```

---

## Data Collection & Visualization

### 1. Real-time Visualization
Download and setup emotibit using the following link: [Emotibit](https://github.com/EmotiBit/EmotiBit_Docs/blob/master/Getting_Started.md)

To launch the EmotiBit oscilloscope for EDA and PPG (Green) signals:
```bash
run_emotibit
```

### 2. Recording Physiological Data
Record incoming BVP and EDA signals into a ROS 2 bag file:
```bash
cd ~/stress_ws
source install/setup.bash
ros2 bag record /bvp_raw /eda_raw
```

---

## Data Processing Pipeline

### 1. Convert Rosbag to CSV

Navigate to conversion utility to extract `.db3` data into readable CSVs, here [ros2bag_convert](https://github.com/fishros/ros2bag_convert) is used:
```bash
cd ros2bag_convert
ros2bag-convert <your_bag_file>.db3
```
*Output: `eda_raw.csv` and `bvp_raw.csv` inside the bag folder.*

### 2. Compress Data (Pickle)
Combine CSV files into a compressed `.pkl` format for faster processing:
```bash
# Edit pickle_creator.py with your file paths first
python3 ~/stress_ws/src/stress_inference_pkg/stress_inference_pkg/pickle_creator.py
```
---

## Running Inference & Analysis

Ensure the workspace root and source the setup file before running these nodes.

```bash
cd stress_ws
source install/setup.bash 
# run it in every terminal
```

| Task | Command |
| :--- | :--- |
| **Mock Recorded Data Publisher** | `ros2 run stress_inference_pkg mock_publisher` |
| **Mock WESAD Dataset Publisher** | `ros2 run stress_inference_pkg mock_wesad_publisher` |
| **Real-time Inference (Static)** | `ros2 run stress_inference_pkg stress_inference_realtime` |
| **Real-time Inference (Adaptive)** | `ros2 run stress_inference_pkg stress_inference_realtime_al` |
| **EmotiBit Publisher** | `ros2 run stress_inference_pkg emobit_publisher` |

###  Feedback

To manually trigger feedback:
```bash
ros2 topic pub /user_feedback std_msgs/msg/String "data: 'stressed'" --once
```

---

### GUI Interface

For an interactive user interface:
```bash
ros2 run stress_inference_pkg gui
```

## Notes

* **Adding New Nodes:** When adding new scripts, remember to register them in `setup.py` under the `console_scripts` entry point.
* **Source Path:** All scripts are located in:  
    `stress_ws/src/stress_inference_pkg/`
