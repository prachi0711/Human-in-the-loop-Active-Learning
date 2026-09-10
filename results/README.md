# Results derived from the System

## Research Question 1: Real-Time Streaming Architecture Validation

### Files 

* **`rq1_analysis_preprocess_wesad.py` (Offline Baseline)**
    Establishes the computational baseline. It runs the Random Forest model on pre-processed, clean WESAD features, logging the absolute minimum inference latency.
* **`rq1_analysis_raw.py` (Real-Time ROS 2 Node)**
    The core real-time inference engine. It simulates live deployment by ingesting raw EDA and BVP streams. It performs on-the-fly signal cleaning (via `neurokit2`), feature extraction, and model inference. It also includes the State Machine that manages uncertainty thresholds, user dialogue triggers, and intervention cooldowns.
* **`rq1_result.py` (Statistical Validation)**
    The evaluation script. It ingests the latency logs from both the offline baseline and the real-time node (including live Emotibit hardware logs). It executes Shapiro-Wilk, and BootStrapped Two One-Sided Tests to prove computational equivalence.

### How to Run the RQ1 Pipeline

1.  **Generate the Baseline:** Run `rq1_analysis_preprocess_wesad.py` to generate `wesad_ref.csv`.
2.  **Run the Real-Time Node:** Execute `rq1_analysis_raw.py` within ROS 2 workspace. Ensure the raw data topics (`/eda_raw`, `/bvp_raw`) are actively publishing. This will generate `wesad_realtime_log.csv` or `emotibit_realtime_log.csv` .
3.  **Evaluate:** Run `rq1_result.py` to output the Normality and BootStrapped equivalence p-values directly to the console.

## Research Questions 2 & 3: Active Learning and Redundancy Reduction

### Files 

* **`rq2_3_analysis_combine.py`**
    The core simulation engine. It iterates through 15 subjects from the WESAD dataset, streaming their raw data in simulated real-time. It simultaneously evaluates the Static baseline model against the Adaptive AL model across a sweep of blend weights ($w = 0.1$ to $1.0$). It tracks multi-class classification metrics and enforces redundancy tracking to count queries. Outputs `weight_tuning_results.csv`.
* **`rq2_3_result_combine.py`**
    It contains the batch simulation results which can used to compute global averages across all subjects. It generates comparison tables for classification metrics and query burden, and outputs the accuracy graph across all weights to visualize the impact of AL weight tuning.
* **`rq2_3_result_individual.py`**
    Provides an evaluation of a single user to demonstrate temporal learning. It calculates strict per-class metrics (Calm vs. Stress) and generates a combined visualization featuring an accuracy-over-time curve and confusion matrices.

### How to Run the RQ2/RQ3 Pipeline

1.  **Run the Batch Simulation:** Execute `rq2_3_analysis_combine.py`. This will take some time as it processes 15 subjects across 10 weight configurations.
2.  **Generate Global Trends:** Run `rq2_3_result_combine.py` to parse the output CSV, generate the accuracy plot, and print the comparative metric tables.
3.  **Generate Individual Visualizations:** Ensure that the individual subject logs are generated, then run `rq2_3_result_individual.py` to output the comprehensive breakdown and visualization for an individual subject.


## Research Question 4: Intent-Aware Dialogue Evaluation

### Files 

* **`rq4_result_custom.py` (Domain-Specific LLM Evaluation)**
    Evaluates the encapsulated ROS 2 `DialogueManager` against a standard keyword classifier. It processes a custom dataset of domain-specific interactions (`custom_llm_dataset.csv`), mapping user replies to an 8-class intent structure. It outputs accuracy, macro F1, and runs McNemar's Test for statistical significance.
* **`rq4_result_dataset.py` (General Benchmark Evaluation)**
    Provides a standardized baseline by testing the LLM (`qwen2.5:3b`) against a subset of the public [SAD](https://github.com/PervasiveWellbeingTech/Stress-Annotated-Dataset-SAD) dataset. It enforces a strict binary classification (Stressor vs. Non-Stressor) via explicit system prompting to validate the LLM's baseline NLP capabilities.

### How to Run the RQ4 Pipeline

1. **Prerequisites:** Ensure Ollama is running locally and the `qwen2.5:3b` model is pulled and active.
2. **Custom Evaluation:** Run `rq4_result_custom.py`. This interfaces directly with the ROS 2 architecture and will output statistical reports.
3. **Benchmark Evaluation:** Run `rq4_result_dataset.py` to evaluate against the public dataset.


## Research Question 5: Explainable AI (XAI) and User Acceptance

### Files

* **`rq5_xai_analysis.ipynb` (XAI Visualizations)**
This Jupyter Notebook is used for generating Explainable AI visualizations. It utilizes SHAP (SHapley Additive exPlanations) and Random Forest feature importance to interpret the model's decision-making process. It outputs Global Feature Importance charts, SHAP Summary Plots (directional impact of features globally), and SHAP Waterfall Plots (local/individual impact for a single 60-second processing window).

### How to Run the RQ5 Pipeline

1. **Open the Notebook:** Open `rq5_xai_analysis.ipynb` using Google Colab, Jupyter Notebook or VS Code.
2. **Execute the Analysis:** Run the cells sequentially to ingest the physiological data, initialize the `TreeExplainer` model, and generate the SHAP and feature importance plots.

### User Study: Explanation Acceptance and Trust

To determine if translating these model triggers into natural language benefits the end-user, a pilot survey was conducted with $N = 11$ participants.

**Evaluation of Natural Language Explanations**
Participants evaluated the system's explanations across four criteria on a 5-point Likert scale (1 = Strongly Disagree, 5 = Strongly Agree).

| Criterion | Question | 
| --- | --- | 
| **QXAI_01** | *This explanation shows me how reliable the system is.* | 
| **QXAI_02** | *This explanation helps me know when I should trust and not trust the system.* | 
| **QXAI_03** | *This explanation of how the system works has sufficient detail.* | 
| **QXAI_04** | *This explanation shows me how accurate the system is.* |


**Direct System Preference**
Participants were presented with a direct comparative choice regarding how the system should handle uncertainty queries:

* **Option A** (System queries without explanations).
* **Option B** (System queries accompanied by natural language explanations).

