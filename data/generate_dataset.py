"""
Political Social Media Dataset Generator
Generates a realistic synthetic dataset for political sentiment analysis
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
import os

random.seed(42)
np.random.seed(42)

# ─── Political Entities ───────────────────────────────────────────────────────
POLITICIANS = [
    "Biden", "Trump", "Harris", "DeSantis", "Obama", "Modi", "Rahul Gandhi", 
    "Arvind Kejriwal", "Amit Shah", "Mamata Banerjee", "Yogi Adityanath",
    "Sunak", "Macron", "Trudeau"
]

PARTIES = [
    "Democrats", "Republicans", "Labour", "Conservatives", 
    "BJP", "INC (Congress)", "AAP", "TMC", "LibDems"
]

LOCATIONS = [
    "Sydney", "Berlin", "Texas", "California", "Florida",
    "Maharashtra", "Delhi", "Gujarat"
]

HASHTAGS_POSITIVE = [
    "#MAGA", "#BidenHarris2024", "#ModiForIndia", "#UnitedWeStand",
    "#Progress", "#HopeAndChange", "#BuildBackBetter", "#IndiaFirst",
    "#Democracy", "#PeaceAndProsperity"
]

HASHTAGS_NEGATIVE = [
    "#Corrupt", "#Failed", "#Resign", "#NeverAgain",
    "#Liar", "#Scam", "#Exposed", "#EndThis",
    "#Fraud", "#Accountability"
]

HASHTAGS_NEUTRAL = [
    "#Election2024", "#PoliticsDaily", "#NewsUpdate",
    "#BreakingNews", "#IndiaNews", "#USPolitics",
    "#GlobalPolitics", "#PolicyDebate", "#VoterRights"
]

# ─── Template Sentences ────────────────────────────────────────────────────────
POSITIVE_TEMPLATES = [
    "{politician} delivered an outstanding speech on economic recovery today. Real leadership! {hashtag}",
    "So proud of what {party} has accomplished this term. Unemployment is down! {hashtag}",
    "{politician} just announced a major healthcare reform. This is what we voted for! {hashtag}",
    "Watching {politician} handle international relations with such grace and dignity. {hashtag}",
    "{party}'s new budget plan is exactly what the middle class needs. Great work! {hashtag}",
    "The policies of {politician} have genuinely improved lives in {location}. Keep it up! {hashtag}",
    "{politician} secured a historic trade deal today. This will create millions of jobs! {hashtag}",
    "Under {politician}'s leadership, crime rates in {location} have dropped significantly. {hashtag}",
    "{party} consistently delivers on its promises. Strong governance is visible. {hashtag}",
    "Incredible infrastructure development happening under {politician}'s watch in {location}. {hashtag}",
    "The education reforms by {party} are transforming schools across the nation. {hashtag}",
    "{politician} shows true compassion for ordinary citizens. A leader for the people! {hashtag}",
    "Record GDP growth under {party} leadership. The economy is thriving! {hashtag}",
    "{politician} united the divided nation with his remarkable speech yesterday. {hashtag}",
    "Proud to be a {party} supporter. Our values align with national progress. {hashtag}",
    "Excellent initiative by {politician} to support rural development. {hashtag}",
    "The new {party} policy is a breath of fresh air for the industry. {hashtag}",
    "Finally, a leader like {politician} who understands the needs of the youth. {hashtag}",
]

NEGATIVE_TEMPLATES = [
    "{politician} has completely failed the working class. Another broken promise. {hashtag}",
    "Can't believe {party} raised taxes again while corruption runs rampant in {location}. {hashtag}",
    "{politician}'s policies are destroying small businesses across {location}. Unacceptable! {hashtag}",
    "The lies told by {politician} during the election are now crystal clear. Disgraceful. {hashtag}",
    "{party} has been in power for years and {location} is still in shambles. {hashtag}",
    "Scandal after scandal with {politician}. When will the accountability come? {hashtag}",
    "Inflation is through the roof under {party}'s mismanagement. Families are suffering. {hashtag}",
    "{politician} sold out to corporate interests. The common man means nothing to them. {hashtag}",
    "Healthcare is a mess and {party} has done nothing meaningful to fix it. {hashtag}",
    "{politician} is the worst thing to happen to {location} in decades. Step down NOW. {hashtag}",
    "Corruption in {party} is at an all-time high. Where is the accountability? {hashtag}",
    "{politician}'s foreign policy has embarrassed us on the world stage. Shameful. {hashtag}",
    "The poor get poorer under {party}'s watch. Trickle-down economics is a lie! {hashtag}",
    "How many more scandals before {politician} is held responsible? Enough is enough! {hashtag}",
    "{party} is tearing this country apart with its divisive rhetoric. {hashtag}",
    "Extremely disappointed with {politician}'s lack of action on climate change. {hashtag}",
    "The latest {party} scandal is just the tip of the iceberg. {hashtag}",
    "Wait, {politician} actually thinks this plan will work? It's a disaster. {hashtag}",
    "Another day, another failure from the {party} administration in {location}. {hashtag}",
]

NEUTRAL_TEMPLATES = [
    "{politician} held a press conference today addressing the ongoing budget negotiations. {hashtag}",
    "{party} released its new policy framework for the upcoming term. {hashtag}",
    "Voters in {location} will head to the polls next week as {politician} campaigns. {hashtag}",
    "The debate between {politician} and the opposition drew millions of viewers. {hashtag}",
    "{party} announced cabinet reshuffle ahead of the parliamentary session. {hashtag}",
    "Analysts are divided on {politician}'s recent economic proposals. {hashtag}",
    "Survey shows mixed public opinion on {party}'s latest immigration policy. {hashtag}",
    "{politician} met with world leaders at the G20 summit in {location}. {hashtag}",
    "Experts weigh in on {party}'s five-year development plan. {hashtag}",
    "Local elections in {location} expected to be closely contested by {party}. {hashtag}",
    "{politician} addressed parliament on the state of national security. {hashtag}",
    "{party} committee to review healthcare spending over the next quarter. {hashtag}",
    "Opposition responds to {politician}'s budget address with mixed reactions. {hashtag}",
    "New poll data shows shifting voter preferences in {location} region. {hashtag}",
    "{politician} signs executive order; legal challenges anticipated from {party}. {hashtag}",
]

SARCASTIC_TEMPLATES = [
    "Oh wow, {politician} solved climate change with a 2-page memo. Genius! {hashtag}",
    "Sure, {party} really cares about the poor. That's why they cut benefits AGAIN. {hashtag}",
    "Totally believe {politician} when they say they're 'listening to the people'. {hashtag}",
    "Yeah {party}, another tax cut for billionaires will definitely fix inequality. Right. {hashtag}",
    "{politician}'s latest speech: another masterpiece of saying nothing for 45 minutes. {hashtag}",
]

# ─── Overlapping Keywords (To make it harder) ────────────────────────────────
# These words will appear in ALL sentiments to prevent simple keyword cheating
NEUTRAL_WORDS = ["policy", "government", "today", "statement", "news", "official", "update"]

# ─── Ambiguous/Mixed Templates ───────────────────────────────────────────────
AMBIGUOUS_TEMPLATES = [
    "I used to support {party}, but after {politician}'s recent speech in {location}, I'm not so sure anymore.",
    "Interesting to see {politician} and {party} working together. Not sure if this is good or bad. {hashtag}",
    "The new {party} bill has some good points but also many flaws. A mixed bag for {location}.",
    "Everyone is talking about {politician}. Some love it, some hate it. What do you think? {hashtag}",
    "While {politician} claims success, the situation in {location} remains complicated. {hashtag}"
]

# ─── Generator Function with Noise ────────────────────────────────────────────
def generate_dataset(n_samples: int = 3000) -> pd.DataFrame:
    records = []
    start_date = datetime(2024, 1, 1)
    
    # Sentiments including Ambiguous (mapped to Neutral for classification)
    sentiment_types = ["positive", "negative", "neutral", "ambiguous"]
    sentiments = random.choices(sentiment_types, weights=[0.3, 0.3, 0.3, 0.1], k=n_samples)

    for i, sentiment in enumerate(sentiments):
        politician = random.choice(POLITICIANS)
        party      = random.choice(PARTIES)
        location   = random.choice(LOCATIONS)
        
        # Select base template
        if sentiment == "positive":
            template = random.choice(POSITIVE_TEMPLATES)
            hashtag  = random.choice(HASHTAGS_POSITIVE + HASHTAGS_NEUTRAL) # Mixed hashtags
        elif sentiment == "negative":
            template = random.choice(NEGATIVE_TEMPLATES)
            hashtag  = random.choice(HASHTAGS_NEGATIVE + HASHTAGS_NEUTRAL)
        elif sentiment == "ambiguous":
            template = random.choice(AMBIGUOUS_TEMPLATES)
            hashtag  = random.choice(HASHTAGS_NEUTRAL)
            sentiment = "neutral" # Ground truth for classification
        else:
            template = random.choice(NEUTRAL_TEMPLATES)
            hashtag  = random.choice(HASHTAGS_NEUTRAL)

        text = template.format(politician=politician, party=party,
                               location=location, hashtag=hashtag)
        
        # --- HARD MODE: Word Shuffling & Noise ---
        words = text.split()
        
        # 1. Randomly swap synonyms or shuffle chunks
        if random.random() < 0.3:
            idx = random.randint(0, max(0, len(words)-3))
            chunk = words[idx:idx+3]
            random.shuffle(chunk)
            words[idx:idx+3] = chunk
            
        # 2. Drop a random word (mimics telegram-style tweets)
        if random.random() < 0.2:
            words.pop(random.randint(0, len(words)-1))
            
        # 3. Inject Typos/Slang (mimics messy social media)
        if random.random() < 0.15:
            typo_idx = random.randint(0, len(words)-1)
            words[typo_idx] = words[typo_idx].replace("th", "ht").replace("ing", "in'").replace("s", "z")

        text = " ".join(words)
        
        # Add random neutral words to "dilute" the sentiment signal
        if random.random() < 0.4:
            text = f"{random.choice(NEUTRAL_WORDS)} {text}"

        # Inject Typos
        if random.random() < 0.1:
            text = text.replace("the", "teh").replace("a ", "4 ")

        records.append({
            "id": i + 1,
            "text": text,
            "timestamp": start_date + timedelta(days=random.randint(0, 400)),
            "user": f"user_{random.randint(100, 999)}",
            "location": location,
            "politician": politician,
            "party": party,
            "source": "Synthetic",
            "true_sentiment": sentiment,
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3000)
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    df = generate_dataset(args.samples)
    df.to_csv("data/political_social_media.csv", index=False)
    print(f"✅ Dataset generated: {len(df)} records in data/political_social_media.csv")
    print(df["true_sentiment"].value_counts())
