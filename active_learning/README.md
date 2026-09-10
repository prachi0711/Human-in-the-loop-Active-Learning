# Stress Inference & Active Learning Framework 

This folder talks about a real-time, human-in-the-loop framework designed to detect psychological stress and deliver non-intrusive, conversational interventions via an intelligent agent. 

By streaming raw physiological signals, Electrodermal Activity (EDA) and Blood Volume Pulse (BVP). The system processes data streams, runs machine learning inferences, manages prediction uncertainty, and personalizes classification boundaries on the fly without introducing latency into the real-time workspace.

---

### 1. `stress_inference_realtime.py` (Static Inference Node)

This file acts as the standard real-time monitoring engine for the project. Its primary purpose is to continuously listen to the incoming biological data streams, verify that enough signal history has been collected, and pass that data through a fixed, pre-trained machine learning model to estimate the user's psychological state.

Beyond simple classification, this node runs the foundational conversational state machine. It monitors model confidence over rolling time intervals to prevent the robot from interrupting the user on a single, isolated spike in stress data. If the system experiences a prolonged period of high uncertainty, or if a high-confidence stress state is sustained, it manages the timing of the agent's intervention while enforcing strict post-conversation cooldown periods to prevent user fatigue.

### 2. `stress_inference_realtime_al.py` (Active Learning Inference Node)

This file serves as the advanced, interactive core of the framework. It retains the exact same raw data handling, signal processing, and timing constraints as the static node, but adds an active collaboration layer between the human and the machine learning pipeline.

Instead of operating with fixed baseline boundaries, this node constantly communicates with a personalization manager to adjust model predictions based on live user feedback. When the system detects a sustained period of high confusion, this file instructs the agent to explicitly ask the user for confirmation of their state. It catches these interactive responses in real time and immediately logs the pairing of the biological metrics with the user's true state, completely bypassing passive monitoring to actively update the system's intelligence.

### 3. `active_learning_manager.py` (Personalization Engine)

This file is the dedicated mathematical background coordinator for the active learning pipeline. It operates completely independently of the live ROS 2 timer loops to ensure that intensive data operations and model calculations never cause the sensor tracking to freeze, lag, or drop input frames.

Whenever a user responds to an interaction query, this engine caches the biological parameters alongside the user's self-reported state to a persistent local file, ensuring memory is preserved across system restarts. It then launches an isolated background thread to retrain a custom, personalized model on this data. Finally, it uses a bounded mathematical blending equation to gracefully combine the generic rules of the base model with the custom patterns of the personal model, capping the personal model's influence to prevent the agent from overfitting or drifting due to isolated, noisy responses.

---

## Execution

### 1. Build the Workspace

```bash
# Navigate to the workspace directory
cd stress_ws

# Compile the specific package
colcon build --packages-select stress_inference_pkg

# Source the environment
source install/setup.bash

```

### 2. Launch the Static Infrastructure Node

Run this node to perform real-time tracking using only the generalized base model:

```bash
ros2 run stress_inference_pkg stress_inference_realtime

```

### 3. Launch the Adaptive Personalization Node

Run this node to enable the system to actively learn individual by user labeling, and dynamically adjust boundaries:

```bash
ros2 run stress_inference_pkg stress_inference_realtime_al

```