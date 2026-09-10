## **Dataset: WESAD (Wearable Stress and Affect Detection)** 
This dataset is a publicly available dataset, which can be downloaded from here [WESAD Dataset](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html).

### **Key Features**  
- **Modalities**: ECG, EDA (GSR), BVP, ACC, RESP, EMG (sampled at 4–64 Hz).  
- **Subjects**: 15 usable subjects (2 discarded) with lab-induced stress.  
- **Study Environment**: Laboratory.
- **Ground Truth Labels:** 1 = Baseline (neutral state); 2 = Stress (induced stress); 3 = Amusement (positive affect).

For this Project, EDA and BVP signals are used which are collected from E4 wearable sensor.

| Signal | Description | Sampling Rate | Sensor | Relevance to Stress |  
|--------|-------------|--------------|--------|---------------------|  
| **EDA** (Electrodermal Activity) | Measures skin conductance (sweat gland activity). | **4 Hz** | Empatica E4 wristband | Direct indicator of sympathetic nervous system arousal (higher EDA = stress). |  
| **BVP** (Blood Volume Pulse) | Tracks blood flow changes (used to derive HR/HRV). | **64 Hz** | Empatica E4 wristband | Heart rate variability (HRV) decreases under stress. |  


### **Stress Level Segmentation**  
To deal with stress data, Label 2 is extracted from the dataset and is then further used for preprocesssing. Label 1 is extracted and used as a non-stress. 

### **Validation Approach**  
- **Leave-One-Subject-Out (LOSO)**: Model trained on 14 subjects, tested on the held-out subject.  

### **Ground Truth**  
- **Self-reports**: Validated via STAI (anxiety) and SSSQ (stress) questionnaires.

# Important Note

This repository does not contain the WESAD dataset files. The dataset must be downloaded separately from the official source.
Extract the contents to:

`Dataset/WESAD`

For full details, refer [WESAD dataset paper](https://doi.org/10.1145/3242969.3242985)

## **Datasets: Natural Language Processing (RQ4)**

In addition to the physiological WESAD dataset, this project utilizes two text-based datasets to evaluate the LLM-based intent recognition capabilities of the Dialogue Manager (RQ4).

### **1. Custom Dialogue Dataset (`custom_llm_dataset.csv`)**
This is a domain-specific dataset created specifically for this thesis. It simulates the ambiguous, conversational ways a user might reply to a query in the real world (e.g., *"I guess I'm okay"*, *"Just walking up the stairs"*, *"Leave me alone"*).

* **Purpose:** To evaluate the system's ability to map fuzzy human feedback into actionable system commands.
* **Size:** 234 phrases.
* **Classes (8-Class Intent Model):**

| Category | Intent Label | Example Phrase |
| :--- | :--- | :--- |
| **State Feedback** | `STRESS` | *"My heart is racing from stress."* |
| | `CALM` | *"I'm in a great mood."* |
| | `AMBIGUOUS` | *"I don't know how I feel."* |
| **System Commands** | `PAUSE` | *"Stop monitoring me."* |
| | `RESUME` | *"Turn monitoring back on."* |
| **Interventions** | `ACCEPTANCE` | *"Yes, let's do the breathing exercise."* |
| | `REJECTION` | *"No thanks, I'm busy."* |
| **Fallback** | `UNKNOWN` | *"The weather is hot."* |

---

### **2. SAD (Stress Annotated Dataset) Benchmark (`SAD_llm_dataset.csv`)**
To prove that the LLM performs well not just on custom data, but on standardized NLP benchmarks, a subset of the public [SAD Dataset](https://github.com/PervasiveWellbeingTech/Stress-Annotated-Dataset-SAD) is used. 

* **Purpose:** To evaluate the LLM's general ability to recognize stressful contexts and events in everyday text compared to traditional keyword-matching.
* **Size:** A subset of 234 sentences for rapid benchmark testing.
* **Classes (Binary Classification):**
  * `1 = Stressor` (e.g., *"An exam at school that everyone is saying you can't pass.."*)
  * `0 = Non-Stressor` (e.g., *"Deciding what to make for dinner."*)


