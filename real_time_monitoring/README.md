# Real-Time Data Generation & Hardware Streaming Subsystems

This section of the framework provides the information of foundational data pipelines required to feed the real-time stress inference nodes. It contains independent ROS 2 publisher nodes designed to fetch, align, and stream physiological signals (EDA and BVP) into the system's operational topics.

---

### 1. `wesad_publisher.py` (Benchmark Validation Dataset Streamer)

This node acts as a mock real-time data generator utilizing the public, standard WESAD (Wearable Stress and Affect Detection) dataset. Its primary purpose is to simulate a live participant session using historical, lab-validated benchmark profiles.

The WESAD data format logs signals at vastly different sampling frequencies (BVP at 64 Hz and EDA at 4 Hz). This node enforces a strict fractional ratio update counter ($64 / 4 = 16$). The core timer ticks exactly 64 times a second, publishing a raw BVP sample on every cycle while releasing an EDA sample precisely once every 16 ticks. To prioritize the evaluation of the inference state machine, this node automatically skips non-relevant states (such as amusement or transitional cognitive tasks. It only streams valid `Baseline` (1) or `Stress` (2) boundary.

### 2. `emotibit_publisher.py` (Hardware LSL Bridge)

This node acts as the direct hardware interface to an active EmotiBit wearable sensor platform. It establishes low-latency communication with the physical sensor array by leveraging Lab Streaming Layer (LSL) network protocols.

To ensure that continuous hardware polling never blocks the ROS execution engine, this file spawns two independent background worker threads dedicated to isolating the incoming data rates of each distinct biological signal stream. The node targets the `EDA` and `PPGGreen` infrared frequency band for cardiovascular tracking, as the green wavelength provides a cleaner signal-to-noise ratio and more stable heart rate tracking over movement artifacts than alternative bands.

### 3. `mock_realtime_publisher.py` (Pre-Recorded Real-Time Simulator)

This node bridges the gap between active hardware testing and reproducible simulation. It allows researchers to re-play live data previously captured from an EmotiBit sensor session without needing a physical sensor connected to a participant.

Unlike the WESAD dataset configuration, EmotiBit hardware profile streams signals at a different internal sampling rate balance (BVP at 25 Hz and EDA at 15 Hz). This file maps that exact timing behavior into its processing loops to mirror native sensor properties. The script re-loads a pre-compiled data structure (`combined_data.pkl`) containing flattened sensor logs. A precise ROS 2 execution timer ticks to read data arrays sequentially, ensuring that the historical timeline matches the exact physiological pacing of the original recording session.

---

## Execution

All the files are located in 
`stress-management-using-active-learning/stress_ws/src/stress_inference_pkg` folder

### 1. Launch the Dataset Simulator 

```bash
ros2 run stress_inference_pkg mock_wesad_publisher

```

### 2. Launch the Live Hardware Bridge 

```bash
ros2 run stress_inference_pkg emotibit_publisher

```

### 3. Launch the Recorded Sensor Data

```bash
ros2 run stress_inference_pkg mock_publisher 

```
