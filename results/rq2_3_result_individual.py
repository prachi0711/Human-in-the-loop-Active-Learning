import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)

def run_analysis():
    # check the location of the files for the individual subject analysis
    try:
        static = pd.read_csv('/stress_ws/al_data/subject_metrics_static.csv') # this gets generated from stress-management-using-active-learning/stress_ws/src/stress_inference_pkg/stress_inference_realtime.py
        adaptive = pd.read_csv('/stress_ws/al_data/subject_metrics_adaptive.csv') # this gets generated from stress-management-using-active-learning/stress_ws/src/stress_inference_pkg/stress_inference_realtime_al.py
    except:
        print("CSV files not found. Run simulation first."); return

    # RQ2: Redundancy 
    red_s = static['redundant'].sum()
    red_a = adaptive['redundant'].sum()
    
    # RQ3: MULTI-CLASS PERFORMANCE PROFILE (SUBJECT S3)
    y_true = static['gt'].values
    y_pred_s = static['pred'].values
    y_pred_a = adaptive['pred'].values

    # Base Accuracy metrics
    acc_s = accuracy_score(y_true, y_pred_s)
    acc_a = accuracy_score(y_true, y_pred_a)
    bal_acc_s = balanced_accuracy_score(y_true, y_pred_s)
    bal_acc_a = balanced_accuracy_score(y_true, y_pred_a)

    # CLASS 0 (CALM) METRICS
    prec_s_c0 = precision_score(y_true, y_pred_s, pos_label=0, zero_division=0)
    prec_a_c0 = precision_score(y_true, y_pred_a, pos_label=0, zero_division=0)
    rec_s_c0 = recall_score(y_true, y_pred_s, pos_label=0, zero_division=0)
    rec_a_c0 = recall_score(y_true, y_pred_a, pos_label=0, zero_division=0)
    f1_s_c0 = f1_score(y_true, y_pred_s, pos_label=0, zero_division=0)
    f1_a_c0 = f1_score(y_true, y_pred_a, pos_label=0, zero_division=0)

    # CLASS 1 (STRESS) METRICS
    prec_s_c1 = precision_score(y_true, y_pred_s, pos_label=1, zero_division=0)
    prec_a_c1 = precision_score(y_true, y_pred_a, pos_label=1, zero_division=0)
    rec_s_c1 = recall_score(y_true, y_pred_s, pos_label=1, zero_division=0)
    rec_a_c1 = recall_score(y_true, y_pred_a, pos_label=1, zero_division=0)
    f1_s_c1 = f1_score(y_true, y_pred_s, pos_label=1, zero_division=0)
    f1_a_c1 = f1_score(y_true, y_pred_a, pos_label=1, zero_division=0)

    # Print comprehensive double-class performance report to terminal
    print("RQ3: INDIVIDUAL SUBJECT PERFORMANCE PROFILE")
    print(f"{'Global Metrics':<25} | {'Static Baseline':<17} | {'Adaptive AL':<15}")
    print(f"{'Overall Accuracy':<25} | {acc_s*100:<17.2f}% | {acc_a*100:<15.2f}%")
    print(f"{'Balanced Accuracy':<25} | {bal_acc_s*100:<17.2f}% | {bal_acc_a*100:<15.2f}%")
    
    print(f"{'Class 0 (Calm) Metrics':<25} | {'Static Baseline':<17} | {'Adaptive AL':<15}")
    print(f"{'Precision (Calm)':<25} | {prec_s_c0*100:<17.2f}% | {prec_a_c0*100:<15.2f}%")
    print(f"{'Recall/TPR (Calm)':<25} | {rec_s_c0*100:<17.2f}% | {rec_a_c0*100:<15.2f}%")
    print(f"{'F1-Score (Calm)':<25} | {f1_s_c0*100:<17.2f}% | {f1_a_c0*100:<15.2f}%")
    
    print(f"{'Class 1 (Stress) Metrics':<25} | {'Static Baseline':<17} | {'Adaptive AL':<15}")
    print(f"{'Precision (Stress)':<25} | {prec_s_c1*100:<17.2f}% | {prec_a_c1*100:<15.2f}%")
    print(f"{'Recall/TPR (Stress)':<25} | {rec_s_c1*100:<17.2f}% | {rec_a_c1*100:<15.2f}%")
    print(f"{'F1-Score (Stress)':<25} | {f1_s_c1*100:<17.2f}% | {f1_a_c1*100:<15.2f}%")
    
    print(f"RQ2 Result: Static Redundancy={red_s}, Adaptive={red_a}")

    # VISUALIZATIONS (ACCURACY CURVE + CONFUSION MATRIX)
    sns.set_theme(style="white")
    fig = plt.figure(figsize=(16, 5))
    
    ax1 = plt.subplot(1, 2, 1)
    def get_acc_curve(df):
        return [accuracy_score(df['gt'].iloc[:i+1], df['pred'].iloc[:i+1]) for i in range(10, len(df))]
    ax1.plot(get_acc_curve(static), label="Static (Baseline)", color='gray', linestyle='--', linewidth=2)
    ax1.plot(get_acc_curve(adaptive), label="Adaptive (AL)", color='green', linewidth=2.5)
    ax1.set_title("Accuracy Performance Trend over Windows (S3)", fontweight='bold', pad=10)
    ax1.set_ylabel("Mean Accuracy")
    ax1.set_xlabel("Sliding Windows")
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    cm_static = confusion_matrix(y_true, y_pred_s, labels=[0, 1])
    cm_adaptive = confusion_matrix(y_true, y_pred_a, labels=[0, 1])
    labels = ['Calm', 'Stress']

    ax2 = plt.subplot(1, 4, 3)
    sns.heatmap(cm_static, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax2,
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 12, "weight": "bold"})
    ax2.set_title("Static Baseline CM", fontweight='bold')
    ax2.set_ylabel("True Labels")
    ax2.set_xlabel("Predicted Labels")

    ax3 = plt.subplot(1, 4, 4)
    sns.heatmap(cm_adaptive, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax3,
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 12, "weight": "bold"})
    ax3.set_title("Adaptive AL CM", fontweight='bold')
    ax3.set_xlabel("Predicted Labels")
    ax3.set_yticklabels([])

    plt.tight_layout()
    plt.savefig("/stress_ws/al_data/comprehensive_report.png", dpi=300)
    plt.show()

if __name__ == "__main__": 
    run_analysis()
