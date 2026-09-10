#!/usr/bin/env python3

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
from dialogue_manager import DialogueManager # dialogue manager exists in stress_ws/src/stress_inference_pkg/dialogue_manager.py
from scipy.stats import binom

# Keyword Baseline (Simulates standard rule-based systems)
def keyword_classifier(text):
    t = text.lower()
    if any(w in t for w in ["stop", "pause", "leave"]): return "PAUSE"
    if any(w in t for w in ["resume", "start"]): return "RESUME"
    if any(w in t for w in ["yes", "ok", "sure", "please"]): return "ACCEPTANCE"
    if any(w in t for w in ["no", "thanks"]): return "REJECTION"
    if any(w in t for w in ["stress", "anxious", "panic", "tense", "nervous", "freaking", "overwhelmed"]): return "STRESS"
    if any(w in t for w in ["calm", "well", "good", "fine", "peaceful", "relaxed", "better"]): return "CALM"
    if any(w in t for w in ["maybe", "don't know", "guess", "perhaps", "not sure"]): return "AMBIGUOUS"
    return "UNKNOWN"

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
    df = pd.read_csv("src/stress_inference_pkg/stress_inference_pkg/custom_llm_dataset.csv") 

    # Model used for LLM
    dm = DialogueManager(model_name="qwen2.5:3b")
    
    keyword_preds = []
    llm_preds = []
    
    print(f"Evaluating phrases")
    
    for index, row in df.iterrows():
        text = row['Phrase']
        
        # Keyword Prediction
        kw_pred = keyword_classifier(text)
        keyword_preds.append(kw_pred)
        
        # LLM Prediction
        llm_pred = dm.get_semantic_intent(text, user_id=f"eval_user_{index}")
        llm_preds.append(llm_pred)
        

    df['Keyword_Pred'] = keyword_preds
    df['LLM_Pred'] = llm_preds
    
    y_true = df['Ground_Truth_Label']
    
    print("RESULTS: CUSTOM (8-CLASS INTENT)")
    
    # Accuracy & F1
    kw_acc = accuracy_score(y_true, keyword_preds)
    llm_acc = accuracy_score(y_true, llm_preds)
    kw_f1 = f1_score(y_true, keyword_preds, average='macro', zero_division=0)
    llm_f1 = f1_score(y_true, llm_preds, average='macro', zero_division=0)
    
    print(f"Keyword System -> Accuracy: {kw_acc*100:.2f}% | Macro F1: {kw_f1:.3f}")
    print(f"LLM System -> Accuracy: {llm_acc*100:.2f}% | Macro F1: {llm_f1:.3f}")
    
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

    if result.pvalue < 0.05:
        print("SUCCESS: The LLM is statistically significantly better.")
    else:
        print("Not statistically significant.")

    power = calculate_mcnemar_power(kw_only, llm_only)

    print("MCNEMAR'S POWER ANALYSIS")
    print(f"Total Discordant Pairs: {kw_only + llm_only}")
    print(f"Statistical Power: {power * 100:.2f}%")
    print

if __name__ == "__main__":
    run_evaluation()

