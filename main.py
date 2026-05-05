"""
Main entry point for running the entire pipeline.
Uses real social media data (Twitter/Reddit/HuggingFace) instead of synthetic data.
"""

import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from data.collect_real_data import collect_real_data
from src.preprocessing import run_preprocessing
from src.ml_model import train_ml_models
from src.dl_model import run_dl_inference
from src.hybrid_pipeline import run_hybrid_pipeline


def run_all(data_source: str = None, max_samples: int = 3000):
    """
    data_source: "twitter", "reddit", "huggingface", "all", or None (uses .env)
    """
    try:
        logger.info(f"=== 1. Collecting Data (Source: {data_source or 'Auto'}) ===")
        os.makedirs("data", exist_ok=True)
        if data_source == "synthetic":
            from data.generate_dataset import generate_dataset
            df = generate_dataset(max_samples)
            df.to_csv("data/political_social_media.csv", index=False)
            logger.info(f"Generated {len(df)} synthetic samples.")
        else:
            df = collect_real_data(source=data_source, max_samples=max_samples)

        logger.info("=== 2. Preprocessing Data ===")
        run_preprocessing("data/political_social_media.csv", "data/preprocessed.csv")

        logger.info("=== 3. Training & Running ML Models ===")
        train_ml_models("data/preprocessed.csv")

        # Memory Check for Cloud
        if os.getenv("DEPLOYMENT_ENV") == "cloud":
            import psutil
            mem = psutil.virtual_memory()
            logger.info(f"Memory Check: {mem.percent}% used ({mem.available / (1024**2):.1f}MB available)")
            if mem.available < 800 * 1024 * 1024: # Less than 800MB
                logger.warning("Low memory detected! DL Inference might fail.")

        logger.info("=== 4. Running DL Inference (RoBERTa) ===")
        run_dl_inference("data/ml_predictions.csv", "data/hybrid_predictions.csv")

        logger.info("=== 5. Running Hybrid Ensemble & NER ===")
        run_hybrid_pipeline("data/hybrid_predictions.csv", "data/final_results.csv")

        logger.info("=== Pipeline Complete! ===")
        
    except Exception as e:
        logger.error(f"PIPELINE CRITICAL FAILURE: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise e


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=None,
                        choices=["twitter", "reddit", "huggingface", "synthetic", "all"])
    parser.add_argument("--max", type=int, default=3000)
    args = parser.parse_args()

    run_all(data_source=args.source, max_samples=args.max)
