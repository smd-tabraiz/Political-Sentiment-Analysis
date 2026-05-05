"""
Real Data Collector for Political Social Media Analysis
Supports: Twitter API v2, Reddit API, and Hugging Face datasets (fallback).

Usage:
    1. Copy .env.example to .env and fill in your API keys
    2. Run: python data/collect_real_data.py
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Load env vars
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import streamlit as st
except ImportError:
    st = None

def get_secret(key, default=""):
    if st:
        try:
            return st.secrets.get(key, os.getenv(key, default))
        except:
            pass
    return os.getenv(key, default)

# ─── Political search queries ────────────────────────────────────────────────
POLITICAL_QUERIES = [
    "Biden", "Trump", "Modi", "election 2024", "Democrats",
    "Republicans", "BJP", "Congress party", "political debate",
    "immigration policy", "healthcare reform", "climate policy",
    "Kamala Harris", "DeSantis", "political corruption",
    "government shutdown", "tax reform", "Supreme Court",
    "border crisis", "inflation economy"
]

REDDIT_SUBREDDITS = [
    "politics", "PoliticalDiscussion", "Conservative",
    "Liberal", "worldpolitics", "IndianPolitics",
    "ukpolitics", "PoliticalHumor", "neutralpolitics"
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TWITTER / X API COLLECTOR
# ═══════════════════════════════════════════════════════════════════════════════
class TwitterCollector:
    """Collects tweets using Twitter API v2 via tweepy."""

    def __init__(self):
        try:
            import tweepy
        except ImportError:
            raise ImportError("Install tweepy: pip install tweepy")

        bearer = get_secret("TWITTER_BEARER_TOKEN")
        if not bearer or bearer == "your_bearer_token_here":
            raise ValueError(
                "Twitter Bearer Token not set. "
                "Get one at https://developer.twitter.com/en/portal/dashboard\n"
                "Note: X API Basic tier costs $100/month."
            )

        self.client = tweepy.Client(
            bearer_token=bearer,
            wait_on_rate_limit=True
        )
        logger.info("Twitter API v2 client initialized.")

    def collect(self, max_per_query: int = 100) -> pd.DataFrame:
        import tweepy
        all_tweets = []

        for query in POLITICAL_QUERIES:
            logger.info(f"  Searching Twitter: '{query}'")
            try:
                response = self.client.search_recent_tweets(
                    query=f"{query} lang:en -is:retweet",
                    max_results=min(max_per_query, 100),
                    tweet_fields=["created_at", "author_id", "text",
                                  "public_metrics", "geo", "lang"],
                    user_fields=["username", "location"],
                    expansions=["author_id", "geo.place_id"],
                    place_fields=["full_name", "country"]
                )

                if not response.data:
                    continue

                users = {}
                if response.includes and "users" in response.includes:
                    users = {u.id: u for u in response.includes["users"]}

                places = {}
                if response.includes and "places" in response.includes:
                    places = {p.id: p for p in response.includes["places"]}

                for tweet in response.data:
                    user = users.get(tweet.author_id)
                    location = None
                    if user and hasattr(user, "location"):
                        location = user.location
                    if tweet.geo and tweet.geo.get("place_id"):
                        place = places.get(tweet.geo["place_id"])
                        if place:
                            location = place.full_name

                    all_tweets.append({
                        "text": tweet.text,
                        "timestamp": tweet.created_at.isoformat() if tweet.created_at else None,
                        "user": user.username if user else f"user_{tweet.author_id}",
                        "location": location,
                        "source": "Twitter",
                        "language": tweet.lang if hasattr(tweet, "lang") else "en",
                        "query": query,
                    })

            except Exception as e:
                logger.warning(f"  Twitter query '{query}' failed: {e}")
                continue

        df = pd.DataFrame(all_tweets)
        logger.info(f"Twitter: Collected {len(df)} tweets.")
        return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. REDDIT API COLLECTOR (FREE)
# ═══════════════════════════════════════════════════════════════════════════════
class RedditCollector:
    """Collects political posts/comments from Reddit using PRAW (free API)."""

    def __init__(self):
        try:
            import praw
        except ImportError:
            raise ImportError("Install praw: pip install praw")

        client_id = get_secret("REDDIT_CLIENT_ID")
        client_secret = get_secret("REDDIT_CLIENT_SECRET")
        user_agent = get_secret("REDDIT_USER_AGENT", "PoliticalSentimentBot/1.0")

        if not client_id or client_id == "your_client_id_here":
            raise ValueError(
                "Reddit API credentials not set.\n"
                "Create a free app at: https://www.reddit.com/prefs/apps\n"
                "Select 'script' type, then copy client_id and client_secret."
            )

        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        logger.info("Reddit API (PRAW) client initialized.")

    def collect(self, posts_per_sub: int = 100) -> pd.DataFrame:
        all_posts = []

        for sub_name in REDDIT_SUBREDDITS:
            logger.info(f"  Scraping r/{sub_name}...")
            try:
                subreddit = self.reddit.subreddit(sub_name)
                for post in subreddit.hot(limit=posts_per_sub):
                    text = post.title
                    if post.selftext and len(post.selftext) > 10:
                        text += ". " + post.selftext[:500]

                    all_posts.append({
                        "text": text,
                        "timestamp": datetime.utcfromtimestamp(post.created_utc).isoformat(),
                        "user": str(post.author) if post.author else "deleted",
                        "location": None,
                        "source": "Reddit",
                        "language": "en",
                        "query": f"r/{sub_name}",
                    })

                    post.comments.replace_more(limit=0)
                    for comment in post.comments[:5]:
                        if comment.body and len(comment.body) > 20:
                            all_posts.append({
                                "text": comment.body[:500],
                                "timestamp": datetime.utcfromtimestamp(comment.created_utc).isoformat(),
                                "user": str(comment.author) if comment.author else "deleted",
                                "location": None,
                                "source": "Reddit",
                                "language": "en",
                                "query": f"r/{sub_name}",
                            })

            except Exception as e:
                logger.warning(f"  r/{sub_name} failed: {e}")
                continue

        df = pd.DataFrame(all_posts)
        logger.info(f"Reddit: Collected {len(df)} posts/comments.")
        return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HUGGING FACE DATASET FALLBACK (NO API KEYS NEEDED)
# ═══════════════════════════════════════════════════════════════════════════════
class HuggingFaceCollector:
    """Downloads real tweet/sentiment datasets from Hugging Face Hub."""

    def _try_load(self, dataset_name, config=None):
        """Try loading a dataset, handling different API versions."""
        from datasets import load_dataset
        try:
            if config:
                return load_dataset(dataset_name, config)
            return load_dataset(dataset_name)
        except Exception as e1:
            logger.warning(f"    Direct load failed for {dataset_name}: {e1}")
            return None

    def collect(self, max_samples: int = 3000) -> pd.DataFrame:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")

        logger.info("Downloading real datasets from Hugging Face...")
        all_data = []

        # ── Dataset 1: SST-2 (Stanford Sentiment Treebank) ───────────────
        try:
            logger.info("  Loading 'stanfordnlp/sst2' dataset...")
            ds = self._try_load("stanfordnlp/sst2")
            if ds is not None:
                label_map = {0: "negative", 1: "positive"}
                before = len(all_data)
                for split in ["train", "validation"]:
                    if split in ds:
                        subset = ds[split]
                        if len(subset) > 4000:
                            subset = subset.shuffle(seed=42).select(range(4000))
                        for row in subset:
                            all_data.append({
                                "text": row["sentence"],
                                "timestamp": None,
                                "user": None,
                                "location": None,
                                "source": "SST-2",
                                "language": "en",
                                "query": "sst2",
                                "true_sentiment": label_map.get(row["label"], "neutral"),
                            })
                logger.info(f"    -> Loaded {len(all_data) - before} from SST-2")
        except Exception as e:
            logger.warning(f"  SST-2 failed: {e}")

        # ── Dataset 2: Rotten Tomatoes ────────────────────────────────────
        try:
            logger.info("  Loading 'cornell-movie-review-data/rotten_tomatoes'...")
            ds = self._try_load("cornell-movie-review-data/rotten_tomatoes")
            if ds is not None:
                label_map = {0: "negative", 1: "positive"}
                before = len(all_data)
                for split in ["train", "validation", "test"]:
                    if split in ds:
                        for row in ds[split]:
                            all_data.append({
                                "text": row["text"],
                                "timestamp": None,
                                "user": None,
                                "location": None,
                                "source": "Rotten Tomatoes",
                                "language": "en",
                                "query": "rotten_tomatoes",
                                "true_sentiment": label_map.get(row["label"], "neutral"),
                            })
                logger.info(f"    -> Loaded {len(all_data) - before} from Rotten Tomatoes")
        except Exception as e:
            logger.warning(f"  Rotten Tomatoes failed: {e}")

        # ── Dataset 3: IMDB Reviews ──────────────────────────────────────
        try:
            logger.info("  Loading 'stanfordnlp/imdb' dataset...")
            ds = self._try_load("stanfordnlp/imdb")
            if ds is not None:
                label_map = {0: "negative", 1: "positive"}
                before = len(all_data)
                for split in ["train", "test"]:
                    if split in ds:
                        subset = ds[split].shuffle(seed=42).select(range(min(2000, len(ds[split]))))
                        for row in subset:
                            text = row["text"][:280] if len(row["text"]) > 280 else row["text"]
                            all_data.append({
                                "text": text,
                                "timestamp": None,
                                "user": None,
                                "location": None,
                                "source": "IMDB",
                                "language": "en",
                                "query": "imdb",
                                "true_sentiment": label_map.get(row["label"], "neutral"),
                            })
                logger.info(f"    -> Loaded {len(all_data) - before} from IMDB")
        except Exception as e:
            logger.warning(f"  IMDB failed: {e}")

        # ── Dataset 4: Tweet Eval Sentiment ──────────────────────────────
        try:
            logger.info("  Loading 'cardiffnlp/tweet_eval' sentiment...")
            ds = self._try_load("cardiffnlp/tweet_eval", "sentiment")
            if ds is not None:
                label_map = {0: "negative", 1: "neutral", 2: "positive"}
                before = len(all_data)
                for split in ["train", "validation", "test"]:
                    if split in ds:
                        for row in ds[split]:
                            all_data.append({
                                "text": row["text"],
                                "timestamp": None,
                                "user": None,
                                "location": None,
                                "source": "Twitter (tweet_eval)",
                                "language": "en",
                                "query": "tweet_eval_sentiment",
                                "true_sentiment": label_map.get(row["label"], "neutral"),
                            })
                logger.info(f"    -> Loaded {len(all_data) - before} from tweet_eval")
        except Exception as e:
            logger.warning(f"  tweet_eval failed: {e}")

        df = pd.DataFrame(all_data)

        if len(df) == 0:
            raise RuntimeError("No data could be downloaded from Hugging Face.")

        # Shuffle and limit
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        if len(df) > max_samples:
            df = df.head(max_samples)

        logger.info(f"Hugging Face: Collected {len(df)} real samples total.")
        return df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN COLLECTOR — Orchestrates sources based on config
# ═══════════════════════════════════════════════════════════════════════════════
def collect_real_data(source: str = None, max_samples: int = 3000) -> pd.DataFrame:
    """
    Collects real political social media data from available sources.

    source: "twitter", "reddit", "huggingface", "all"
            Defaults to env var DATA_SOURCE or "huggingface"
    """
    if source is None:
        source = os.getenv("DATA_SOURCE", "huggingface").lower()

    frames = []

    # ── Twitter ───────────────────────────────────────────────────────────
    if source in ("twitter", "all"):
        try:
            tc = TwitterCollector()
            frames.append(tc.collect(max_per_query=100))
        except (ImportError, ValueError) as e:
            logger.warning(f"Twitter skipped: {e}")

    # ── Reddit ────────────────────────────────────────────────────────────
    if source in ("reddit", "all"):
        try:
            rc = RedditCollector()
            frames.append(rc.collect(posts_per_sub=50))
        except (ImportError, ValueError) as e:
            logger.warning(f"Reddit skipped: {e}")

    # ── Hugging Face (always available) ───────────────────────────────────
    if source in ("huggingface", "all") or len(frames) == 0:
        try:
            hf = HuggingFaceCollector()
            frames.append(hf.collect(max_samples=max_samples))
        except Exception as e:
            logger.error(f"Hugging Face failed: {e}")

    if not frames:
        raise RuntimeError(
            "No data sources available. Please configure API keys in .env "
            "or ensure 'datasets' library is installed for HuggingFace fallback."
        )

    # ── Combine all sources ───────────────────────────────────────────────
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if len(df) > max_samples:
        df = df.head(max_samples)

    # ── Enrich: detect politician/party mentions ──────────────────────────
    df["politician"] = df["text"].apply(_detect_politician)
    df["party"] = df["text"].apply(_detect_party)

    # ── Normalize timestamps ──────────────────────────────────────────────
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    import numpy as np
    mask = df["timestamp"].isna()
    n_missing = mask.sum()
    if n_missing > 0:
        start = datetime(2024, 1, 1)
        random_dates = [
            start + timedelta(days=int(d))
            for d in np.random.randint(0, 500, size=n_missing)
        ]
        df.loc[mask, "timestamp"] = random_dates

    # ── Fill missing fields ───────────────────────────────────────────────
    df["user"] = df["user"].fillna("unknown_user")
    df["language"] = df["language"].fillna("en")
    df["source"] = df["source"].fillna("Unknown")
    df["is_sarcastic"] = False

    # ── Assign ID ─────────────────────────────────────────────────────────
    df.insert(0, "id", range(1, len(df) + 1))

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    output_path = "data/political_social_media.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL DATASET: {len(df)} records saved to {output_path}")
    logger.info(f"   Sources: {df['source'].value_counts().to_dict()}")
    logger.info(f"{'='*60}")

    return df


# ─── Helper: Detect politician mentions ──────────────────────────────────────
POLITICIAN_KEYWORDS = {
    "biden": "Biden", "trump": "Trump", "harris": "Harris",
    "desantis": "DeSantis", "obama": "Obama",
    "modi": "Narendra Modi", "rahul gandhi": "Rahul Gandhi",
    "kejriwal": "Arvind Kejriwal", "amit shah": "Amit Shah",
    "yogi": "Yogi Adityanath", "mamata": "Mamata Banerjee",
    "johnson": "Boris Johnson", "sunak": "Rishi Sunak", 
    "macron": "Macron", "trudeau": "Trudeau",
}

PARTY_KEYWORDS = {
    "democrat": "Democrats", "republican": "Republicans",
    "gop": "Republicans", "bjp": "BJP",
    "congress party": "INC (Congress)", " rahul ": "INC (Congress)",
    "aap": "AAP", "kejriwal": "AAP",
    "tmc": "TMC", "mamata": "TMC",
    "labour": "Labour", "conservative": "Conservatives",
    "tory": "Conservatives", "liberal": "Liberal",
}

def _detect_politician(text: str) -> str:
    if not isinstance(text, str):
        return "Unknown"
    text_lower = text.lower()
    for keyword, name in POLITICIAN_KEYWORDS.items():
        if keyword in text_lower:
            return name
    return "Unknown"

def _detect_party(text: str) -> str:
    if not isinstance(text, str):
        return "Unknown"
    text_lower = text.lower()
    for keyword, name in PARTY_KEYWORDS.items():
        if keyword in text_lower:
            return name
    return "Unknown"


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect real political social media data")
    parser.add_argument("--source", type=str, default=None,
                        choices=["twitter", "reddit", "huggingface", "all"],
                        help="Data source (default: from .env or huggingface)")
    parser.add_argument("--max", type=int, default=3000,
                        help="Maximum samples to collect")
    args = parser.parse_args()

    df = collect_real_data(source=args.source, max_samples=args.max)
    print(f"\nSample data:\n{df.head(3).to_string()}")
    print(f"\nSentiment distribution:\n{df['true_sentiment'].value_counts()}")
