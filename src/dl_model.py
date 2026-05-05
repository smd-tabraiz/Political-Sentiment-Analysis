"""
Deep Learning Sentiment Classifier
Uses a pre-trained Transformer (cardiffnlp/twitter-roberta-base-sentiment-latest)
via Hugging Face pipeline for inference.
Returns per-sample predictions with confidence scores.
"""

import os
import logging
import torch
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DEVICE = 0 if torch.cuda.is_available() else -1

class DLSentimentClassifier:
    def __init__(self):
        logger.info(f"Loading DL model: {MODEL_NAME} (Device: {'GPU' if DEVICE == 0 else 'CPU'})")
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            device=DEVICE,
            max_length=512,
            truncation=True,
            top_k=None # Get all probabilities
        )

    def _map_label(self, label: str) -> str:
        # Map RoBERTa labels to our format
        mapping = {
            "positive": "positive",
            "neutral": "neutral",
            "negative": "negative"
        }
        return mapping.get(label.lower(), "neutral")

    def predict(self, texts: list) -> list[dict]:
        logger.info(f"Running DL predictions on {len(texts)} texts...")
        results = []
        
        # Process in batches for progress bar and efficiency
        batch_size = 32
        for i in tqdm(range(0, len(texts), batch_size), desc="DL Inference"):
            batch = texts[i:i+batch_size]
            # Handle potential None or empty strings
            clean_batch = [str(t) if t and str(t).strip() else "neutral" for t in batch]
            
            try:
                preds = self.pipeline(clean_batch)
                
                for pred_list in preds:
                    # pred_list is a list of dicts: [{'label': 'positive', 'score': 0.9}, ...]
                    # We want to extract the top label and all probabilities
                    probs = {self._map_label(p["label"]): round(p["score"], 4) for p in pred_list}
                    top_label = max(probs, key=probs.get)
                    top_score = probs[top_label]
                    
                    results.append({
                        "dl_label": top_label,
                        "dl_confidence": top_score,
                        "dl_probs": probs
                    })
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # Fallback for errors
                for _ in range(len(batch)):
                    results.append({
                        "dl_label": "neutral",
                        "dl_confidence": 0.0,
                        "dl_probs": {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
                    })
                    
        return results

def run_dl_inference(input_path: str = "data/ml_predictions.csv", 
                     output_path: str = "data/hybrid_predictions.csv") -> pd.DataFrame:
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    classifier = DLSentimentClassifier()
    
    # Use the cleaned text for DL as well, or original text depending on model.
    # RoBERTa often works well with minimally cleaned text, but cleaned_text is safer.
    # We will use original text since RoBERTa can handle syntax, but let's use cleaned for consistency.
    texts = df["cleaned_text"].tolist()
    
    predictions = classifier.predict(texts)
    
    df["dl_label"] = [p["dl_label"] for p in predictions]
    df["dl_confidence"] = [p["dl_confidence"] for p in predictions]
    df["dl_probs"] = [str(p["dl_probs"]) for p in predictions]
    
    df.to_csv(output_path, index=False)
    logger.info(f"✅ DL predictions added and saved to {output_path}")
    return df

if __name__ == "__main__":
    run_dl_inference()
