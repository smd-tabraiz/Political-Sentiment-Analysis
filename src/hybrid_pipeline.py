"""
Hybrid Combiner & Advanced NLP Pipeline
1. Hybrid Ensemble Strategy: Combines ML and DL predictions.
2. Advanced NLP: NER (Named Entity Recognition) using SpaCy.
"""

import ast
import spacy
import logging
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Ensure spacy model is installed
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    import sys
    logger.info("Downloading SpaCy model en_core_web_sm...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def ensemble_strategy(row: pd.Series, w_ml: float = 0.7, w_dl: float = 0.3) -> pd.Series:
    """
    SOFT VOTING STRATEGY:
    Combines ML and DL probabilities using weighted averaging.
    """
    # Set adaptive weights based on data source
    # Real data requires more context (DL), Synthetic data is easier for ML
    is_real = row.get("source", "Synthetic").lower() != "synthetic"
    
    if is_real:
        w_ml, w_dl = 0.4, 0.6 # Favor DL context for real data
    else:
        # User defined or default
        pass 

    try:
        # Clean string from numpy types before parsing
        ml_probs_str = str(row["ml_probs"]).replace("np.str_", "")
        dl_probs_str = str(row["dl_probs"]).replace("np.str_", "")
        
        ml_probs = {str(k): v for k, v in ast.literal_eval(ml_probs_str).items()}
        dl_probs = {str(k): v for k, v in ast.literal_eval(dl_probs_str).items()}
        
        # Soft Voting
        combined = {}
        for sentiment in ["negative", "neutral", "positive"]:
            combined[sentiment] = (ml_probs.get(sentiment, 0.0) * w_ml) + \
                                  (dl_probs.get(sentiment, 0.0) * w_dl)
        
        final_label = max(combined, key=combined.get)
        final_conf = round(combined[final_label], 4)
        is_diff = 1 if final_label != row["ml_label"] else 0
        
        # Sarcasm Detection (Heuristic: ML says Positive, DL says Negative)
        is_sarcastic = 0
        if row["ml_label"] == "positive" and row["dl_label"] == "negative" and row["dl_confidence"] > 0.7:
            is_sarcastic = 1
            
    except Exception as e:
        final_label = row.get("ml_label", "neutral")
        final_conf = row.get("ml_confidence", 0.0)
        is_diff = 0
        is_sarcastic = 0
        
    return pd.Series([final_label, final_conf, is_diff, is_sarcastic])

def extract_entities(text: str) -> list:
    if not isinstance(text, str):
        return []
    doc = nlp(text)
    # Extract Persons (PERSON), Organizations (ORG), Locations (GPE, LOC)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]]
    return list(set(entities))

def run_hybrid_pipeline(input_path: str = "data/hybrid_predictions.csv",
                        output_path: str = "data/final_results.csv") -> pd.DataFrame:
    logger.info(f"Loading predictions from {input_path}")
    df = pd.read_csv(input_path)
    
    # 1. Apply Ensemble
    logger.info("Applying Hybrid Ensemble Strategy (Soft Voting)...")
    ensemble_df = df.apply(ensemble_strategy, axis=1, args=(0.6, 0.4))
    df[["final_sentiment", "final_confidence", "is_different_from_ml", "is_sarcastic"]] = ensemble_df
    
    diff_count = df["is_different_from_ml"].sum()
    logger.info(f"Hybrid logic changed {diff_count} predictions ({diff_count/len(df):.1%}) compared to ML.")
    
    # 2. Extract Entities
    logger.info("Extracting Named Entities (NER)...")
    tqdm.pandas(desc="NER Extraction")
    df["extracted_entities"] = df["text"].progress_apply(extract_entities)
    
    # Save Final Dataset
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Final dataset saved to {output_path}")
    
    # Print accuracy if true sentiment labels are available
    if "true_sentiment" in df.columns:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        labeled = df[
            df["true_sentiment"].notna() &
            df["true_sentiment"].isin(["positive", "negative", "neutral"])
        ].copy()
        
        if len(labeled) >= 10:
            y_true = labeled["true_sentiment"]
            
            perf = {}
            for col_prefix, label_col in [("ML", "ml_label"), ("DL", "dl_label"), ("Hybrid", "final_sentiment")]:
                acc = accuracy_score(y_true, labeled[label_col])
                p, r, f1, _ = precision_recall_fscore_support(y_true, labeled[label_col], average="weighted")
                cm = confusion_matrix(y_true, labeled[label_col], labels=["negative", "neutral", "positive"])
                
                perf[col_prefix] = {
                    "accuracy": round(float(acc), 4),
                    "precision": round(float(p), 4),
                    "recall": round(float(r), 4),
                    "f1": round(float(f1), 4),
                    "confusion_matrix": cm.tolist()
                }
            
            logger.info(f"--- Performance Comparison (on {len(labeled)} labeled rows) ---")
            for m, s in perf.items():
                logger.info(f"{m:6} | Acc: {s['accuracy']:.4f} | F1: {s['f1']:.4f}")

            import json
            with open("data/summary_metrics.json", "w") as f:
                json.dump(perf, f, indent=4)
            
            # Keep legacy text file for safety
            with open("data/summary_metrics.txt", "w") as f:
                for m, s in perf.items():
                    f.write(f"{m} Accuracy: {s['accuracy']}\n")
        else:
            logger.info("Not enough labeled rows for accuracy comparison.")
            with open("data/summary_metrics.txt", "w") as f:
                f.write(f"ML Accuracy: N/A\n")
                f.write(f"DL Accuracy: N/A\n")
                f.write(f"Hybrid Accuracy: N/A\n")
                f.write(f"Total Samples: {len(df)}\n")
            
    return df
            
    return df

if __name__ == "__main__":
    run_hybrid_pipeline()
