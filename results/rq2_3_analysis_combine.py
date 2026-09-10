# Analysis of the active learning model on batch data
import os
import pickle
import joblib
import warnings
import numpy as np
import pandas as pd
import neurokit2 as nk

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix
)
from scipy.stats import entropy
from al_batch_manager import ActiveLearningManager
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    from neurokit2.misc import NeuroKitWarning
    warnings.simplefilter("ignore", category=NeuroKitWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

WESAD_BASE_DIR = "/home/backup_hp/Dataset/WESAD/"
SCALER_PATH = "/home/backup_hp/stress_ws/src/stress_inference_pkg/stress_inference_pkg/rf_scaler.save"
MODEL_PATH = "/home/backup_hp/stress_ws/src/stress_inference_pkg/stress_inference_pkg/rf_uq_model_60.pkl"

FEATURE_COLS = [
    "EDA_Tonic_SD", "SCR_Peaks_Amplitude_Mean", "SCR_Peaks_N", "eda_kurtosis",
    "eda_max", "eda_mean", "eda_min", "eda_range", "eda_skewness", "eda_std",
    "ibi_max", "ibi_mean", "ibi_median", "ibi_min", "ibi_range", "ibi_std"
]

def get_rates(tn, fp, fn, tp):
    """Helper to calculate FPR, TPR, TNR, FNR safely."""
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return fpr, tpr, tnr, fnr

def simulate_subject(sub_id, max_weight):
    sub_path = os.path.join(WESAD_BASE_DIR, sub_id, f"{sub_id}.pkl")
    if not os.path.exists(sub_path):
        return None

    # Load WESAD data
    with open(sub_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    bvp = data["signal"]["wrist"]["BVP"].flatten()
    eda = data["signal"]["wrist"]["EDA"].flatten()
    labels = data["label"]

    scaler = joblib.load(SCALER_PATH)
    rf_model = joblib.load(MODEL_PATH)

    al_mgr = ActiveLearningManager(sub_id, FEATURE_COLS, "al_data_tmp", fixed_weight=max_weight, is_simulation=True)

    win_size, slide = 64 * 60, 64 * 30
    static_preds, adaptive_preds, ground_truth = [], [], []
    
    # Query performance trackers
    static_queries, static_redundant = 0, 0
    adaptive_queries, adaptive_redundant = 0, 0
    
    current_static_state, static_state_query_count = None, 0
    current_adaptive_state, adaptive_state_query_count = None, 0

    PROFILING_BUDGET = 2

    for i in range(0, len(bvp) - win_size, slide):
        lbl = labels[int((i + win_size / 2) * (len(labels) / len(bvp)))]
        
        # Ignore non-study labels
        if lbl not in [1, 2]:
            current_static_state, static_state_query_count = None, 0
            current_adaptive_state, adaptive_state_query_count = None, 0
            continue

        gt = 1 if lbl == 1 else 0

        # Feature Extraction
        try:
            window_eda = eda[int(i / 16) : int((i + win_size) / 16)]
            window_bvp = bvp[i : i + win_size]

            eda_signals, _ = nk.eda_process(window_eda, sampling_rate=4)
            eda_nk = nk.eda_intervalrelated(eda_signals, sampling_rate=4).iloc[0]
            s_eda = pd.Series(window_eda)

            f_dict = {
                "eda_mean": np.mean(window_eda), "eda_std": np.std(window_eda),
                "eda_min": np.min(window_eda), "eda_max": np.max(window_eda),
                "eda_range": np.max(window_eda) - np.min(window_eda),
                "eda_skewness": s_eda.skew(), "eda_kurtosis": s_eda.kurtosis(),
                "EDA_Tonic_SD": np.std(eda_signals["EDA_Tonic"]),
                "SCR_Peaks_Amplitude_Mean": eda_nk.get("SCR_Peaks_Amplitude_Mean", 0.0),
                "SCR_Peaks_N": eda_nk.get("SCR_Peaks_N", 0.0)
            }

            peaks = nk.ppg_findpeaks(window_bvp, sampling_rate=64)["PPG_Peaks"]
            ibi = np.diff(peaks) / 64
            if len(ibi) > 2:
                f_dict.update({
                    "ibi_mean": np.mean(ibi), "ibi_std": np.std(ibi), "ibi_median": np.median(ibi),
                    "ibi_min": np.min(ibi), "ibi_max": np.max(ibi), "ibi_range": np.max(ibi) - np.min(ibi)
                })

            feats = [f_dict.get(c, 0.0) for c in FEATURE_COLS]
        except Exception:
            continue

        # Inference
        feats_scaled = scaler.transform(pd.DataFrame([feats], columns=FEATURE_COLS))
        base_probs = rf_model.predict_proba(feats_scaled)[0]
        blended_probs, w = al_mgr.blend_with_features(base_probs, feats)

        static_pred = 1 if int(np.argmax(base_probs)) == 0 else 0
        adaptive_pred = 1 if int(np.argmax(blended_probs)) == 0 else 0

        static_preds.append(static_pred)
        adaptive_preds.append(adaptive_pred)
        ground_truth.append(gt)

        # Static Redundancy Tracking
        if float(entropy(base_probs)) > 0.45:
            static_queries += 1
            if current_static_state == gt:
                static_state_query_count += 1
                if static_state_query_count > PROFILING_BUDGET:
                    static_redundant += 1
            else:
                current_static_state, static_state_query_count = gt, 1

        # Adaptive Redundancy Tracking
        if float(entropy(blended_probs)) > 0.45:
            adaptive_queries += 1
            if current_adaptive_state == gt:
                adaptive_state_query_count += 1
                if adaptive_state_query_count > PROFILING_BUDGET:
                    adaptive_redundant += 1
            else:
                current_adaptive_state, adaptive_state_query_count = gt, 1
            
            # Feed label to AL Manager
            al_mgr.add_label(feats, "calm" if gt == 1 else "stress")

    if os.path.exists(al_mgr.csv_path):
        os.remove(al_mgr.csv_path)

    if not ground_truth:
        return None

    # Metrics Calculation
    static_tn, static_fp, static_fn, static_tp = confusion_matrix(ground_truth, static_preds, labels=[0, 1]).ravel()
    adapt_tn, adapt_fp, adapt_fn, adapt_tp = confusion_matrix(ground_truth, adaptive_preds, labels=[0, 1]).ravel()

    s_fpr, s_tpr, s_tnr, s_fnr = get_rates(static_tn, static_fp, static_fn, static_tp)
    a_fpr, a_tpr, a_tnr, a_fnr = get_rates(adapt_tn, adapt_fp, adapt_fn, adapt_tp)

    return {
        "Static_Acc": accuracy_score(ground_truth, static_preds),
        "Adaptive_Acc": accuracy_score(ground_truth, adaptive_preds),
        
        "Static_BalAcc": balanced_accuracy_score(ground_truth, static_preds),
        "Adaptive_BalAcc": balanced_accuracy_score(ground_truth, adaptive_preds),
        
        "Static_Precision": precision_score(ground_truth, static_preds, pos_label=1, zero_division=0),
        "Adaptive_Precision": precision_score(ground_truth, adaptive_preds, pos_label=1, zero_division=0),
        
        "Static_Recall": recall_score(ground_truth, static_preds, pos_label=1, zero_division=0),
        "Adaptive_Recall": recall_score(ground_truth, adaptive_preds, pos_label=1, zero_division=0),
        
        "Static_F1": f1_score(ground_truth, static_preds, pos_label=1, zero_division=0),
        "Adaptive_F1": f1_score(ground_truth, adaptive_preds, pos_label=1, zero_division=0),

        "Static_FPR": s_fpr, "Adaptive_FPR": a_fpr,
        "Static_TPR": s_tpr, "Adaptive_TPR": a_tpr,
        "Static_TNR": s_tnr, "Adaptive_TNR": a_tnr,
        "Static_FNR": s_fnr, "Adaptive_FNR": a_fnr,

        "Static_Queries": static_queries,
        "Static_Redundant": static_redundant,
        "Adaptive_Queries": adaptive_queries,
        "Adaptive_Redundant": adaptive_redundant
    }


# batch runner
if __name__ == "__main__":
    subjects = [f"S{i}" for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]]
    weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    results = []

    for w in weights:
        print(f"Evaluating Max Blend Weight Ceiling: {w}")
        for sub in subjects:
            metrics = simulate_subject(sub, w)
            if metrics:
                row_data = {"Subject": sub, "Max_Weight": w}
                row_data.update(metrics)
                results.append(row_data)

    df_res = pd.DataFrame(results)
    df_res.to_csv("weight_tuning_results.csv", index=False)
    print("Sweep complete, Saved to weight_tuning_results.csv")