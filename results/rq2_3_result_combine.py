import pandas as pd
import matplotlib.pyplot as plt

# check the location of the file; it gets generated from stress-management-using-active-learning/results/rq2_3_analysis_combine.py
df = pd.read_csv("weight_tuning_results.csv")
summary = df.groupby("Max_Weight").mean(numeric_only=True).reset_index()


def plot_metric(df, x_col, static_col, adaptive_col, ylabel, title, filename):
    plt.figure(figsize=(9, 5.5))
    
    plt.plot(df[x_col], df[static_col], label="Static Baseline", 
             color="crimson", linestyle="--", marker="o", linewidth=2)
             
    plt.plot(df[x_col], df[adaptive_col], label="Adaptive (AL)", 
             color="tab:blue", linestyle="-", marker="s", linewidth=2.5)
             
    plt.xlabel("Weight ($w$)", fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.title(title, fontsize=12, pad=15)
    plt.xticks(df[x_col])
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300)
    plt.close() 
    print(f"Graph saved as '{filename}'!")


# Visualizations
plots_config = [
    ("Static_Acc", "Adaptive_Acc", "Mean Accuracy", "Accuracy Performance", "accuracy_weight_sweep.png"),
    ("Static_FNR", "Adaptive_FNR", "Mean False Negative Rate", "False Negative Rate Performance", "fnr_weight_sweep.png"),
    ("Static_TNR", "Adaptive_TNR", "Mean True Negative Rate", "True Negative Rate Performance", "tnr_weight_sweep.png"),
    ("Static_TPR", "Adaptive_TPR", "Mean True Positive Rate", "True Positive Rate Performance", "tpr_weight_sweep.png"),
    ("Static_FPR", "Adaptive_FPR", "Mean False Positive Rate", "False Positive Rate Performance", "fpr_weight_sweep.png")
]

for static_col, adaptive_col, ylabel, title, filename in plots_config:
    plot_metric(summary, "Max_Weight", static_col, adaptive_col, ylabel, title, filename)


# Summary Tables

# Define the columns for RQ3
rq3_cols = [
    "Max_Weight", "Static_Acc", "Adaptive_Acc", "Static_BalAcc", "Adaptive_BalAcc",
    "Static_Precision", "Adaptive_Precision", "Static_Recall", "Adaptive_Recall",
    "Static_F1", "Adaptive_F1", "Static_FPR", "Adaptive_FPR", "Static_TPR", 
    "Adaptive_TPR", "Static_TNR", "Adaptive_TNR", "Static_FNR", "Adaptive_FNR"
]

print("RQ3 SUMMARY - CLASSIFICATION PERFORMANCE")
print(summary[rq3_cols].to_string(index=False, float_format="{:.3f}".format))


# Define the columns for RQ2
rq2_cols = ["Max_Weight", "Static_Queries", "Adaptive_Queries", "Static_Redundant", "Adaptive_Redundant"
]

print("RQ2 SUMMARY - QUERY AND REDUNDANCY PERFORMANCE")
print(summary[rq2_cols].to_string(index=False, float_format="{:.1f}".format))