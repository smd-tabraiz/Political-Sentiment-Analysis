# 📊 Political Sentiment Analysis: Hybrid ML & DL Approach

## 1. Project Overview
This project implements a robust **Hybrid Sentiment Analysis System** designed specifically for political social media data. It addresses the complexities of political discourse—including sarcasm, noise, and complex sentence structures—by combining the speed and statistical strength of Traditional Machine Learning (ML) with the deep contextual understanding of Transformer-based Deep Learning (DL) models.

## 2. Key Features
- **Data Collection & Generation**: Capability to collect real social media data (Twitter, Reddit, HuggingFace) or generate a realistic synthetic dataset.
- **Advanced Preprocessing**: Normalizes political entities, removes noise, handles emojis, and applies tokenization and lemmatization.
- **Hybrid Ensembling**: Combines predictions from both Traditional ML (TF-IDF + Logistic Regression, Naive Bayes, SVM) and Deep Learning (RoBERTa fine-tuned on Twitter sentiment) using a probability-weighted approach.
- **Advanced NLP (NER)**: Extracts Named Entities (Politicians, Parties, Locations) using SpaCy to allow entity-wise sentiment tracking.
- **Analytical Dashboard**: A Streamlit application offering interactive visualizations of processed data, sentiment distributions, and model performance.

## 3. Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd Political-Sentiment-Analysis
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install SpaCy model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Environment Setup:**
   Copy `.env.example` to `.env` and fill in any required API keys (e.g., for Twitter/Reddit data collection).
   ```bash
   cp .env.example .env
   ```

## 4. Usage

### Running the Full Pipeline
You can run the entire pipeline from data collection/generation to hybrid inference using `main.py`.

```bash
# Run with synthetic data (default 3000 samples)
python main.py --source synthetic

# Run with HuggingFace dataset
python main.py --source huggingface

# Run with other sources
python main.py --source twitter
```

### Running the Dashboard
After running the pipeline and generating predictions, you can visualize the results via the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

## 5. Pipeline Architecture
1. **Data Collection**: Retrieves raw text from chosen sources.
2. **Preprocessing**: Cleans text, standardizes entities, and prepares features.
3. **ML Baseline**: Trains and predicts using TF-IDF + traditional classifiers.
4. **DL Refinement**: Runs inference using `cardiffnlp/twitter-roberta-base-sentiment-latest`.
5. **Hybrid Ensembling**: Averages probabilities from ML and DL models.
6. **Dashboard**: Presents interactive analytics.

## 6. Project Structure
```
.
├── .streamlit/             # Streamlit configuration
├── data/                   # Data processing and generation scripts, and datasets
├── models/                 # Saved models and scalers
├── src/                    # Source code (preprocessing, ML, DL, hybrid pipeline)
├── main.py                 # Main execution script
├── dashboard.py            # Streamlit dashboard application
├── requirements.txt        # Project dependencies
├── SUMMARY_REPORT.md       # Detailed technical summary report
└── README.md               # Project documentation
```

## 7. More Information
For a more detailed technical dive into the models, strengths/weaknesses, and biases, please see the `SUMMARY_REPORT.md`.
