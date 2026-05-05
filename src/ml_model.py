"""
Traditional ML Sentiment Classifier
Uses TF-IDF vectorization with Logistic Regression, Naive Bayes, and SVM.
Returns per-sample predictions with confidence scores.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score, precision_recall_fscore_support)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
MODEL_DIR = "models/ml"


# ─── TF-IDF + Classifier Pipeline ────────────────────────────────────────────
def build_pipeline(classifier: str = "lr") -> Pipeline:
    """
    classifier: 'lr' (Logistic Regression), 'nb' (Naive Bayes), 'svm' (SVM)
    """
    # 1. MUCH FEWER FEATURES (1000 instead of 50000)
    # 2. Only Unigrams (1,1) instead of (1,3) to prevent memorizing phrases
    # EXTREME CONSTRAINT: Only 200 words allowed. 
    # This prevents the model from memorizing specific template keywords.
    tfidf = TfidfVectorizer(
        ngram_range=(1, 1),
        max_features=200,
        stop_words="english"
    )

    if classifier == "lr":
        # C=0.01 is VERY strong regularization
        clf = LogisticRegression(
            max_iter=1000, C=0.01, solver="lbfgs",
            random_state=42
        )
    elif classifier == "nb":
        clf = MultinomialNB(alpha=2.0) # Increased smoothing
    elif classifier == "svm":
        clf = SVC(C=0.01, probability=True, kernel='linear', random_state=42)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    return Pipeline([("tfidf", tfidf), ("clf", clf)])


# ─── Trainer ─────────────────────────────────────────────────────────────────
class MLSentimentClassifier:
    def __init__(self, classifier: str = "lr"):
        self.classifier_name = classifier
        self.pipeline = build_pipeline(classifier)
        self.le = LabelEncoder()
        self.is_trained = False
        self.metrics: dict = {}

    def fit(self, X_train: list, y_train: list) -> "MLSentimentClassifier":
        logger.info(f"Training {self.classifier_name.upper()} pipeline...")
        y_enc = self.le.fit_transform(y_train)
        self.pipeline.fit(X_train, y_enc)
        self.is_trained = True
        logger.info("✅ Training complete.")
        return self

    def evaluate(self, X_test: list, y_test: list, X_train: list = None, y_train: list = None) -> dict:
        y_enc = self.le.transform(y_test)
        y_pred = self.pipeline.predict(X_test)

        acc = accuracy_score(y_enc, y_pred)
        # Calculate Precision, Recall, F1
        prec, rec, f1, _ = precision_recall_fscore_support(y_enc, y_pred, average='weighted')
        
        if X_train is not None and y_train is not None:
            train_enc = self.le.transform(y_train)
            train_pred = self.pipeline.predict(X_train)
            train_acc = accuracy_score(train_enc, train_pred)
            logger.info(f"  {self.classifier_name.upper()} Overfitting Check: Train={train_acc:.4f}, Val={acc:.4f}")
            if (train_acc - acc) > 0.15:
                logger.warning(f"  ⚠️ {self.classifier_name.upper()} may be OVERFITTING! (Gap: {train_acc-acc:.4f})")

        report = classification_report(y_enc, y_pred,
                                        target_names=self.le.classes_,
                                        output_dict=True)
        cm = confusion_matrix(y_enc, y_pred)

        self.metrics = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_weighted": round(float(f1), 4),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }
        logger.info(f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
        
        # --- FEATURE IMPORTANCE INSPECTION (Overfitting check) ---
        try:
            tfidf = self.pipeline.named_steps["tfidf"]
            clf = self.pipeline.named_steps["clf"]
            feature_names = tfidf.get_feature_names_out()
            
            if hasattr(clf, "coef_"):
                logger.info("Top words per class (Top 5):")
                for i, class_label in enumerate(self.le.classes_):
                    # For multi-class LR, coef_ has shape (n_classes, n_features)
                    coefs = clf.coef_[i] if len(clf.coef_) > 1 else clf.coef_[0]
                    top_indices = np.argsort(coefs)[-5:][::-1]
                    top_words = [feature_names[idx] for idx in top_indices]
                    logger.info(f"  [{class_label}]: {', '.join(top_words)}")
        except Exception as e:
            logger.debug(f"Could not extract feature importance: {e}")

        return self.metrics

    def predict(self, texts: list) -> list[dict]:
        """Returns list of dicts with label and confidence."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")
        probs   = self.pipeline.predict_proba(texts)
        classes = self.le.classes_
        results = []
        for prob_row in probs:
            idx   = np.argmax(prob_row)
            label = classes[idx]
            conf  = round(float(prob_row[idx]), 4)
            all_probs = {c: round(float(p), 4) for c, p in zip(classes, prob_row)}
            results.append({
                "ml_label":      label,
                "ml_confidence": conf,
                "ml_probs":      all_probs,
            })
        return results

    def cross_validate(self, X: list, y: list, cv: int = 5) -> dict:
        y_enc = self.le.fit_transform(y)
        skf   = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.pipeline, X, y_enc, cv=skf,
                                  scoring="f1_weighted", n_jobs=1)
        return {"cv_mean": round(scores.mean(), 4), "cv_std": round(scores.std(), 4)}

    def save(self, path: str = None):
        path = path or f"{MODEL_DIR}/{self.classifier_name}_pipeline.pkl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "le": self.le,
                     "metrics": self.metrics}, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "MLSentimentClassifier":
        data = joblib.load(path)
        instance = cls.__new__(cls)
        instance.pipeline        = data["pipeline"]
        instance.le              = data["le"]
        instance.metrics         = data.get("metrics", {})
        instance.is_trained      = True
        instance.classifier_name = path.split("/")[-1].split("_")[0]
        return instance


# ─── Main Training Script ─────────────────────────────────────────────────────
def train_ml_models(preprocessed_path: str = "data/preprocessed.csv") -> dict:
    logger.info(f"Loading preprocessed data from {preprocessed_path}")
    df = pd.read_csv(preprocessed_path)
    df = df[df["cleaned_text"].notna() & (df["cleaned_text"].str.strip() != "")]

    # ── Handle real-world data: only train on rows that have labels ────────
    labeled_mask = df["true_sentiment"].notna() & df["true_sentiment"].isin(
        ["positive", "negative", "neutral"]
    )
    labeled_df = df[labeled_mask].copy()

    if len(labeled_df) < 50:
        logger.warning(
            f"Only {len(labeled_df)} labeled rows found. "
            "Need at least 50 for training. Skipping ML training — will use DL only."
        )
        # Still generate placeholder ML predictions so the pipeline continues
        df["ml_label"]      = "neutral"
        df["ml_confidence"] = 0.0
        df["ml_probs"]      = str({"negative": 0.0, "neutral": 1.0, "positive": 0.0})
        df["split"]         = "unlabeled"
        df.to_csv("data/ml_predictions.csv", index=False)
        return {"models": {}, "metrics": {}, "best": None, "df": df}

    X = labeled_df["cleaned_text"].tolist()
    y = labeled_df["true_sentiment"].tolist()

    # 1. First split: Train (70%) and Temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    # 2. Second split: Validation (15%) and Test (15%)
    # test_size=0.5 because 0.5 * 0.3 = 0.15
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    logger.info(f"Split sizes: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

    results = {}
    all_metrics = {}

    for clf_name in ["lr", "svm", "nb"]:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {clf_name.upper()} classifier")
        clf = MLSentimentClassifier(classifier=clf_name)
        # Fit on training data
        clf.fit(X_train, y_train)
        
        # Overfitting check
        train_metrics = clf.evaluate(X_train, y_train)
        test_metrics = clf.evaluate(X_test, y_test)
        
        train_acc = train_metrics["accuracy"]
        test_acc = test_metrics["accuracy"]
        logger.info(f"  {clf_name.upper()} Check -> Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        if (train_acc - test_acc) > 0.15:
            logger.warning(f"  ⚠️ {clf_name.upper()} shows signs of OVERFITTING!")

        cv_scores = clf.cross_validate(X, y)
        test_metrics.update(cv_scores)
        clf.save()
        results[clf_name] = clf
        all_metrics[clf_name] = test_metrics
        logger.info(f"CV Score: {cv_scores['cv_mean']:.4f} ± {cv_scores['cv_std']:.4f}")

    # Pick best model by F1
    best_name = max(all_metrics, key=lambda k: all_metrics[k]["f1_weighted"])
    logger.info(f"\n🏆 Best ML model: {best_name.upper()} (F1={all_metrics[best_name]['f1_weighted']})")

    # Save metrics
    import json
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        serializable = {}
        for k, v in all_metrics.items():
            serializable[k] = {kk: vv for kk, vv in v.items() if kk != "confusion_matrix"}
            serializable[k]["confusion_matrix"] = v["confusion_matrix"]
        json.dump({"metrics": serializable, "best_model": best_name}, f, indent=2)

    # Run predictions on ENTIRE dataset (labeled + unlabeled)
    logger.info("Running ML predictions on full dataset...")
    best_clf = results[best_name]
    all_preds = best_clf.predict(df["cleaned_text"].tolist())
    df["ml_label"]      = [p["ml_label"]      for p in all_preds]
    df["ml_confidence"] = [p["ml_confidence"] for p in all_preds]
    df["ml_probs"]      = [str(p["ml_probs"]) for p in all_preds]

    # Mark split
    df["split"] = "unlabeled"
    df.loc[labeled_mask, "split"] = "labeled"

    df.to_csv("data/ml_predictions.csv", index=False)
    logger.info(f"✅ ML predictions saved to data/ml_predictions.csv ({len(df)} rows)")

    return {"models": results, "metrics": all_metrics, "best": best_name, "df": df}


if __name__ == "__main__":
    results = train_ml_models()
