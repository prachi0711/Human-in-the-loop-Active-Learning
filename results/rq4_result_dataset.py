#!/usr/bin/env python3

import pandas as pd
import ollama
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.stats import binom
from statsmodels.stats.contingency_tables import mcnemar

MODEL_NAME = "qwen2.5:3b"

# Keyword Baseline
def keyword_sad_classifier(text):
    t = str(text).lower()
    stress_words = ["work", "project", "boss", "coworker", "hospital", "pneumonia", "sick", 
                    "angry", "ignored", "cold", "died", "heavy", "fell", "difficult"]
    if any(w in t for w in stress_words):
        return 1
    return 0

# LLM Classifier
def llm_sad_classifier(text):
    system = (
        "You are a binary text classifier. Read the 'Sentence' and determine if it describes a stressor (an event, situation, or feeling causing stress/anxiety).\n"
        "Output EXACTLY '1' if it is a stressor, or '0' if it is NOT a stressor or is just normal conversation.\n"
        "Output NOTHING ELSE. No punctuation."
    ) # set of instruction to the LLM model
    try:
        res = ollama.generate(
            model=MODEL_NAME, 
            system=system, 
            prompt=f"Sentence: '{text}'",
            options={"temperature": 0.0, "num_predict": 10}
        )
        out = res['response'].strip()
        return 1 if "1" in out else 0
    except:
        return 0

# post-hoc analysis
def calculate_mcnemar_power(b, c, alpha=0.05):
    """
    Calculates post-hoc statistical power for McNemar's exact test.
    b: Number of times Model 1 was right and Model 2 was wrong (kw_only).
    c: Number of times Model 2 was right and Model 1 was wrong (llm_only).
    """
    n_discordant = b + c
    
    # If the models never disagreed, power is strictly 0
    if n_discordant == 0:
        return 0.0
        
    p0 = 0.5
    p1 = max(b, c) / n_discordant
    
    # critical boundaries for a two-tailed exact binomial test
    k_lower = binom.ppf(alpha / 2, n_discordant, p0)
    k_upper = binom.ppf(1 - alpha / 2, n_discordant, p0)
    
    # probability of rejecting the null given the observed effect size
    power = binom.cdf(k_lower, n_discordant, p1) + (1 - binom.cdf(k_upper - 1, n_discordant, p1))
    
    return min(1.0, power)

def run_evaluation():
    # Only uses "sentence" and "is_stressor" column
    df = pd.read_csv("Dataset/SAD_llm_dataset.csv").dropna(subset=['sentence', 'is_stressor']).head(234)
    
    keyword_preds = []
    llm_preds = []
    
    print(f"Evaluating phrases")
    
    for index, row in df.iterrows():
        text = row['sentence']
        
        # Keyword Prediction
        kw_pred = keyword_sad_classifier(text)
        keyword_preds.append(kw_pred)
        
        # LLM Prediction
        llm_pred = llm_sad_classifier(text)
        llm_preds.append(llm_pred)

    df['Keyword_Pred'] = keyword_preds
    df['LLM_Pred'] = llm_preds
    
    y_true = df['is_stressor'].astype(int)

    print("RESULTS: GENERAL BENCHMARK (BINARY)")

    # Accuracy & F1
    kw_acc = accuracy_score(y_true, keyword_preds)
    llm_acc = accuracy_score(y_true, llm_preds)
    kw_f1 = f1_score(y_true, keyword_preds, average='macro')
    llm_f1 = f1_score(y_true, llm_preds, average='macro')
    
    print(f"Keyword Baseline -> Accuracy: {kw_acc*100:.2f}% | Macro F1: {kw_f1:.3f}")
    print(f"LLM Classifier   -> Accuracy: {llm_acc*100:.2f}% | Macro F1: {llm_f1:.3f}")
    
    # McNEMAR'S TEST for evaluation
    both_correct = sum((k == t and l == t) for k, l, t in zip(keyword_preds, llm_preds, y_true))
    both_incorrect = sum((k != t and l != t) for k, l, t in zip(keyword_preds, llm_preds, y_true))
    llm_only = sum((l == t and k != t) for k, l, t in zip(keyword_preds, llm_preds, y_true))
    kw_only = sum((k == t and l != t) for k, l, t in zip(keyword_preds, llm_preds, y_true))
    
    table = [[both_correct, kw_only], [llm_only, both_incorrect]]
    result = mcnemar(table, exact=True)
    
    print("STATISTICAL SIGNIFICANCE (McNemar's Test)")
    print(f"LLM was exclusively correct on {llm_only} cases where Keywords failed.")
    print(f"Keywords were exclusively correct on {kw_only} cases where LLM failed.")
    print(f"P-Value: {result.pvalue}")
    
    if result.pvalue < 0.05:
        print("SUCCESS: The LLM is statistically significantly better.")
    else:
        print("Not statistically significant.")

    power = calculate_mcnemar_power(kw_only, llm_only)
    
    print("MCNEMAR'S POWER ANALYSIS")
    print(f"Total Discordant Pairs: {kw_only + llm_only}")
    print(f"Statistical Power: {power * 100:.2f}%")


if __name__ == "__main__":
    run_evaluation()

