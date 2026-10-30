# Conversational Dialogue Manager 

This module talks about the `DialogueManager` class, which serves as the interactive front end of the stress mitigation framework. It is responsible for translating physiological system states (such as high-confidence stress or prolonged uncertainty) into supportive, empathetic, and collaborative human-agent dialogue.

The module eliminates rigid, repetitive scripting by structuring conversational workflows through a hybrid approach: a deterministic state machine manages the timing of interventions, while a Large Language Model (LLM) handles the semantic classification and response generation. This ensures interactions remain dynamically helpful and respectful of the user's focus.

---

## The LLM Integration (Qwen2.5:3b)

The dialogue manager utilizes a local instance of the `qwen2.5:3b` model (via Ollama) to process user interactions in two distinct phases: Semantic Intent Classification and Response Generation.

---

## Prerequisites & Setup (Ollama + Qwen)

This dialogue manager runs entirely locally to ensure data privacy and zero-latency interactions. It requires Ollama and the Qwen 2.5 (3B) language model to function.

Before running the ROS 2 node, you must install and serve the model:

1. **Install Ollama:** Download and install Ollama for your operating system from [ollama.com](https://ollama.com/).
2. **Download the Model:** Open your terminal and pull the specific Qwen model used in this framework:
   ```bash
   ollama pull qwen2.5:3b
   ```
3. **Verify the Server is Running**: Ensure the Ollama background service is active. (On Linux, it typically runs automatically as a system service, or you can start it manually with ollama serve).

### 1. Semantic Intent Classification
When a user replies to the system, the LLM acts as a strict text classifier, mapping the user's free-text input into one of eight distinct operational categories. To ensure reliability in a healthcare context, the LLM is constrained by strict prompt rules:

* **`STRESS` / `CALM`:** Explicit declarations of internal feelings in the present tense (e.g., *"I feel anxious"* or *"I am fine"*).
* **`ACCEPTANCE` / `REJECTION`:** Overriding rules that prioritize handling explicit agreements or refusals when a breathing exercise has just been offered (e.g., *"yes, please"* or *"no thanks"*).
* **`PAUSE` / `RESUME`:** Commands to manage the monitoring state.
* **`AMBIGUOUS`:** Expressions of uncertainty (e.g., *"I don't know"* or *"maybe"*).
* **`UNKNOWN`:** The default safety bucket. The LLM is explicitly instructed to categorize physical symptoms (*"my head hurts"*), past tense statements (*"I was stressed earlier"*), and external complaints (*"traffic was bad"*) as `UNKNOWN` to prevent the system from offering breathing exercises for physical ailments or unrelated venting.

### 2. Response Generation
Once the semantic intent is classified, the LLM generates a tailored response based on a "Task Goal." The LLM is constrained to act as a minimalist companion, ensuring it replies in 1-2 direct sentences, avoids inventing data, and never offers breathing instructions unless explicitly instructed by the state machine.

---

## State Machine Logic 

While the LLM handles the conversation, the state machine tracks physiological triggers to determine *when* to initiate dialogue. 

The table below outlines how the system shifts between operational states based on model classifications, computed uncertainty, and LLM-evaluated user replies:

| Current State | Input Condition | Intervention Trigger | Evaluated User Intent | Agent Action |
| --- | --- | --- | --- | --- |
| `buffering` | Queues fill up to 60s windows | None | None | None |
| `monitoring` | System Initialization; waiting for the first prediction | None | None | None |
| `calm` | Pred_label = 0; entropy <= 0.45 | LLM generates warm encouragement | None | None |
| `uncertain`| entropy > 0.45 for 180s | LLM generates gentle check-in | `CALM` / `UNKNOWN` | Acknowledges statement; returns to monitoring |
| `uncertain` | entropy > 0.45 for 180s | LLM generates gentle check-in | `STRESS` | Empathetic response; offers breathing exercise |
| `stress` | Pred_label = 1; entropy <= 0.45 | States observation; offers breathing exercise | `ACCEPTANCE` | Triggers UI breathing animation |
| `stress` | Pred_label = 1; entropy <= 0.45 | States observation; offers breathing exercise | `REJECTION` / `AMBIGUOUS` | Validates choice; logs response; returns to monitoring |

---

## Component Overview

The code is located within the main workspace directory:

`~/Human-in-the-loop-Active-Learning/stress_ws/src/stress_inference_pkg/`

`dialogue_manager.py`: Contains the complete core runtime logic for calling the Ollama API, managing isolated user state dictionaries (allowing for multi-turn conversational history), tracking intervention timestamps, enforcing semantic parsing rules, and keeping the LLM synced with live runtime conditions.
