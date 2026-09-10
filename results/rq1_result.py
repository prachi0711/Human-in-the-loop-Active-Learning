import pandas as pd
import numpy as np
from scipy.stats import shapiro
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

col_name = 'latency_ms'
alpha = 0.05
equiv_margin = 1000  # margin for equivalence testing

files = {
    "Simulated": "Dataset/wesad_realtime_log.csv", # logs from the raw wesad dataset running in real-time
    "Live Hardware": "Dataset/emotibit_realtime_log.csv", # logs from emotibit sensor running in realtime
    "Pre-Processed": "Dataset/wesad_ref.csv" # logs from the preprocessed wesad dataset
}

data = {name: pd.read_csv(filepath)[col_name].dropna().values for name, filepath in files.items()}

sim_latency = data["Simulated"]
hw_latency = data["Live Hardware"]
pre_latency = data["Pre-Processed"]

# Normality Test
stat_sim, p_sim = shapiro(sim_latency)
stat_hw, p_hw = shapiro(hw_latency)
stat_we, p_we = shapiro(pre_latency)

print("Normality Check")
print(f"Simulated Stream Normality : p-value = {p_sim:.4f}")
print(f"Live Hardware Normality    : p-value = {p_hw:.4f}")
print(f"Pre-Processed Normality    : p-value = {p_we:.4f}")

def bootstrap_tost(g1, g2, margin, n_resamples=10000, alpha=0.05):
    """
    Performs a bootstrapped Two One-Sided Test for equivalence.
    Returns the p-value and the confidence interval of the differences.
    """
    np.random.seed(42)  # for the random sampling to be consistent every time 
    
    n1, n2 = len(g1), len(g2)
    boot_diffs = np.zeros(n_resamples)
    
    for i in range(n_resamples):
        sample1 = np.random.choice(g1, size=n1, replace=True)
        sample2 = np.random.choice(g2, size=n2, replace=True)
        boot_diffs[i] = np.mean(sample1) - np.mean(sample2)
        
    # how often the bootstrapped differences fall outside your equivalence margin
    p_lower = np.mean(boot_diffs <= -margin)
    p_upper = np.mean(boot_diffs >= margin)
    
    p_tost = max(p_lower, p_upper)
    
    # the Confidence Interval - 90% CI is used for an alpha=0.05
    ci_lower = np.percentile(boot_diffs, alpha * 100)
    ci_upper = np.percentile(boot_diffs, 100 - (alpha * 100))
    
    return p_tost, ci_lower, ci_upper

# Bootstrapped Equivalence Test 
print(f"Bootstrapped Results (Margin: {equiv_margin} ms)")

comparisons = [
    ("Pre-Processed vs. Simulated", pre_latency, sim_latency),
    ("Pre-Processed vs. Live Hardware", pre_latency, hw_latency),
    ("Simulated vs. Live Hardware", sim_latency, hw_latency)
]

# raw p-values and confidence intervals using bootstrapping
results = [bootstrap_tost(g1, g2, equiv_margin) for _, g1, g2 in comparisons]
raw_p_values = [res[0] for res in results]
confidence_intervals = [(res[1], res[2]) for res in results]

reject, corrected_p_values, _, _ = multipletests(raw_p_values, alpha=alpha, method='bonferroni')

for i, (comp_name, _, _) in enumerate(comparisons):
    ci_low, ci_high = confidence_intervals[i]
    
    # If the p-value is 0.0, in 10,000 tries, the difference never exceeded the margin.
    display_p = "< 0.0001" if corrected_p_values[i] == 0 else f"{corrected_p_values[i]:.4e}"
    
    conclusion = "Practically Equivalent" if reject[i] else "Not Equivalent"
    
    print(f"  Corrected P-value: {display_p} -> {conclusion}")
    print(f"  90% Confidence Interval of Difference: [{ci_low:.2f} ms, {ci_high:.2f} ms]")

# Post-hoc analysis
def calculate_tost_power(g1, g2, margin, alpha=0.05):
    """
    Calculates the statistical power of an equivalence test (TOST).
    """
    n1, n2 = len(g1), len(g2)
    diff = abs(np.mean(g1) - np.mean(g2))
    
    # Welch's Standard Error
    se = np.sqrt((np.var(g1, ddof=1) / n1) + (np.var(g2, ddof=1) / n2))
    
    # Z value for the given alpha 
    z_alpha = norm.ppf(1 - alpha)
    
    # the power probability that the CI falls within the margin
    if diff >= margin:
        return 0.0
        
    power = norm.cdf((margin - diff) / se - z_alpha)
    return max(0.0, min(1.0, power))

print(f"\n Statistical Power Analysis (Margin: {equiv_margin} ms)")

for i, (comp_name, g1, g2) in enumerate(comparisons):
    power = calculate_tost_power(g1, g2, equiv_margin, alpha)
    
    print(f"{comp_name}:")
    print(f" Sample Sizes: N1 = {len(g1)}, N2 = {len(g2)}")
    print(f" Statistical Power: {power * 100:.2f}%")
