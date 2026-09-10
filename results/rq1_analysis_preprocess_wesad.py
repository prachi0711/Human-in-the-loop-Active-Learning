import pandas as pd
import joblib
import time
from scipy.stats import entropy

# preprocessed data, it can be found here: https://github.com/prachi0711/Stress-Management-using-Physiological-Signals-and-conversational-agents/tree/main/preprocessing/data
eda = pd.read_csv("preprocessing/data/eda_features_60.csv")
ibi = pd.read_csv("preprocessing/data/ibi_features_60.csv")
df = pd.merge(eda, ibi, on=["subject", "label", "window"])

features = [
 'EDA_Tonic_SD','SCR_Peaks_Amplitude_Mean','SCR_Peaks_N',
 'eda_kurtosis','eda_max','eda_mean','eda_min','eda_range',
 'eda_skewness','eda_std',
 'ibi_max','ibi_mean','ibi_median','ibi_min','ibi_range','ibi_std'
]

scaler = joblib.load("/home/backup_hp/Downloads/R&D/final/rf_scaler.save")
rf     = joblib.load("/home/backup_hp/Downloads/R&D/final/rf_uq_model_60.pkl")

latencies = []

ref = df[features].copy()

for i in range(len(df)):

    start = time.perf_counter()

    X = scaler.transform(df[features].iloc[[i]])
    probs = rf.predict_proba(X)

    end = time.perf_counter()

    latencies.append((end - start) * 1000) 

    ref.loc[i, "prob_baseline"] = probs[:,0]
    ref.loc[i, "prob_stress"]   = probs[:,1]
    ref.loc[i, "entropy"]       = entropy(probs.T)

ref["latency_ms"] = latencies

ref.to_csv("wesad_ref.csv", index=False)
