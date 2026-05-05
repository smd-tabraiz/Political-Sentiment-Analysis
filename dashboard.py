import os
import ast
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import subprocess

# --- DEPLOYMENT HELPER ---
def get_secret(key, default=None):
    """Get secret from streamlit secrets (cloud) or env vars (local)."""
    try:
        return st.secrets[key]
    except (KeyError, AttributeError, FileNotFoundError):
        return os.getenv(key, default)

# Page config
st.set_page_config(page_title="Hybrid Political Sentiment AI", page_icon="🏛️", layout="wide")

# Premium CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    * {
        font-family: 'Outfit', sans-serif;
    }

    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Responsive Metrics Grid */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px !important;
        transition: transform 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.05);
    }

    .metric-box {
        background: linear-gradient(135deg, rgba(31, 119, 180, 0.15) 0%, rgba(31, 119, 180, 0.05) 100%);
        padding: 24px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        border: 1px solid rgba(31, 119, 180, 0.3);
        margin-bottom: 20px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1f77b4, #64b5f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
    }

    .insight-card {
        background: #1a1c24;
        padding: 25px;
        border-radius: 18px;
        border-left: 6px solid #1f77b4;
        margin: 15px 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    }
    
    .insight-card h4 {
        color: #64b5f6 !important;
        margin-top: 0;
        font-weight: 700;
    }

    /* Mobile specific adjustments */
    @media (max-width: 640px) {
        .metric-value {
            font-size: 1.8rem;
        }
        .stPlotlyChart {
            height: 300px !important;
        }
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
    }

    /* Premium Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Custom Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        background: linear-gradient(90deg, #1f77b4, #0b5394);
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        opacity: 0.9;
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    path = get_secret("RESULTS_CSV_PATH", "data/final_results.csv")
    if not os.path.exists(path):
        # Fallback for fresh deployment
        if os.path.exists("data/political_social_media.csv"):
            return pd.read_csv("data/political_social_media.csv")
        return None
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    def parse_entities(x):
        try:
            return ast.literal_eval(x)
        except:
            return []
    
    if "extracted_entities" in df.columns:
        df["extracted_entities"] = df["extracted_entities"].apply(parse_entities)
    return df

def get_advanced_metrics():
    path = "data/summary_metrics.json"
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        # Added a timestamp check or just direct read to avoid stale data
        return json.load(f)

def plot_confusion_matrix(cm, labels, title):
    fig = px.imshow(cm,
                    x=labels,
                    y=labels,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='Blues',
                    title=title)
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    return fig

def main():
    df = load_data()
    all_metrics = get_advanced_metrics()
    
    if df is None or all_metrics is None:
        st.error("Data or Metrics not found. Please run the full pipeline first.")
        return
        
    politicians_list = sorted([p for p in df["politician"].unique() if p != "Unknown"])
    parties_list = sorted([p for p in df["party"].unique() if p != "Unknown"])
    
    st.title("🏛️ Hybrid Political Sentiment Analysis AI")
    st.markdown("### Production-Grade Pipeline Evaluation & Insights")

    # --- EXECUTIVE SUMMARY ---
    with st.container():
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        s_col1.metric("Total Analyzed", f"{len(df):,}")
        s_col2.metric("Politicians tracked", len(politicians_list))
        s_col3.metric("System Accuracy", f"{all_metrics['Hybrid']['accuracy']:.1%}")
        s_col4.metric("DL Contribution", f"{df['is_different_from_ml'].mean():.1%}")
    
    st.markdown("---")

    # --- Sidebar Filters ---
    st.sidebar.header("🔍 Filters")
    
    if st.sidebar.button("🔥 Force Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Data Source Management")
    source_choice = st.sidebar.radio(
        "Select Active Pipeline Source:",
        ["Synthetic Data Generator", "Real-World APIs / HF"],
        index=0,
        help="Choose 'Synthetic' for testing or 'Real-World' for live social media data."
    )
    
    source_map = {
        "Synthetic Data Generator": "synthetic",
        "Real-World APIs / HF": "all"
    }
    
    if st.sidebar.button("🛠️ Re-Run Full Analysis Pipeline"):
        with st.sidebar.status("🔄 Running Analysis...", expanded=True) as status:
            import sys
            # Limit samples on cloud to avoid OOM
            max_samples = 300 if get_secret("DEPLOYMENT_ENV") == "cloud" else 1000
            
            cmd = [sys.executable, "main.py", "--source", source_map[source_choice], "--max", str(max_samples)]
            st.code(" ".join(cmd))
            
            try:
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Create a placeholder for real-time logs
                log_placeholder = st.empty()
                full_log = ""
                
                for line in process.stdout:
                    full_log += line
                    log_placeholder.code(full_log[-1000:]) # Show last 1000 chars of logs
                
                process.wait()
                
                if process.returncode == 0:
                    status.update(label="✅ Pipeline Complete!", state="complete", expanded=False)
                    st.cache_data.clear()
                    st.rerun()
                else:
                    status.update(label="❌ Pipeline Failed", state="error")
                    st.error(f"Pipeline exited with code {process.returncode}. Check logs above for details.")
            except Exception as e:
                status.update(label="❌ Execution Error", state="error")
                st.error(f"Failed to start pipeline: {e}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 Model Tester")
    user_text = st.sidebar.text_area("Test your own political text:", placeholder="e.g., Oh great, another tax hike. Just what we needed!")
    
    if user_text:
        # Load models for real-time testing
        from src.ml_model import MLSentimentClassifier
        from src.dl_model import DLSentimentClassifier
        from src.hybrid_pipeline import ensemble_strategy
        
        try:
            # We assume best model exists
            ml_clf = MLSentimentClassifier.load(f"models/ml/{all_metrics['ML'].get('best_model', 'lr')}_pipeline.pkl")
            dl_clf = DLSentimentClassifier()
            
            ml_res = ml_clf.predict([user_text])[0]
            dl_res = dl_clf.predict([user_text])[0]
            
            # Combine
            row = pd.Series({
                "ml_label": ml_res["ml_label"],
                "ml_confidence": ml_res["ml_confidence"],
                "ml_probs": str(ml_res["ml_probs"]),
                "dl_label": dl_res["dl_label"],
                "dl_confidence": dl_res["dl_confidence"],
                "dl_probs": str(dl_res["dl_probs"])
            })
            hybrid_res = ensemble_strategy(row, 0.6, 0.4)
            
            st.sidebar.success(f"Final Sentiment: **{hybrid_res[0].upper()}**")
            st.sidebar.progress(hybrid_res[1], text=f"Confidence: {hybrid_res[1]:.1%}")
            
            # Sarcasm Check
            if ml_res["ml_label"] != dl_res["dl_label"] and (ml_res["ml_confidence"] > 0.6 and dl_res["dl_confidence"] > 0.6):
                st.sidebar.warning("⚠️ Potential Sarcasm Detected!")
        except Exception as e:
            st.sidebar.error(f"Test Error: {e}")
    
    if st.sidebar.button("🗑️ Clear All Filters"):
        st.rerun()

    politicians_list = sorted([p for p in df["politician"].unique() if p != "Unknown"])
    parties_list = sorted([p for p in df["party"].unique() if p != "Unknown"])

    st.sidebar.info(f"📊 Loaded {len(df)} records")
    st.sidebar.info(f"👥 {len(politicians_list)} Politicians found")

    selected_pols = st.sidebar.multiselect("👤 Select Politicians", options=politicians_list)
    selected_parties = st.sidebar.multiselect("🎯 Select Parties", options=parties_list)

    filtered_df = df.copy()
    if selected_pols:
        filtered_df = filtered_df[filtered_df["politician"].isin(selected_pols)]
    if selected_parties:
        filtered_df = filtered_df[filtered_df["party"].isin(selected_parties)]

    # --- 1. PERFORMANCE COMPARISON ---
    st.header("🚀 1. Model Reliability & Performance")
    
    # Identify best model
    best_model = max(all_metrics, key=lambda x: all_metrics[x]['f1'])
    
    m_cols = st.columns(3)
    for i, (name, stats) in enumerate(all_metrics.items()):
        with m_cols[i]:
            is_best = (name == best_model)
            badge = '<span class="best-badge">🏆 BEST MODEL</span>' if is_best else ""
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">{name} Model {badge}</div>
                <div class="metric-value">{stats['f1']:.2%}</div>
                <div class="metric-label">Weighted F1-Score</div>
                <hr>
                <div style="display: flex; justify-content: space-around;">
                    <div><small>Acc</small><br><b>{stats['accuracy']:.2%}</b></div>
                    <div><small>Prec</small><br><b>{stats['precision']:.2%}</b></div>
                    <div><small>Rec</small><br><b>{stats['recall']:.2%}</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Performance Comparison Chart
    st.subheader("Model Metric Comparison")
    comparison_data = []
    for name, stats in all_metrics.items():
        for metric in ["accuracy", "precision", "recall", "f1"]:
            comparison_data.append({"Model": name, "Metric": metric.capitalize(), "Score": stats[metric]})
    
    comp_df = pd.DataFrame(comparison_data)
    fig_comp = px.bar(comp_df, x="Metric", y="Score", color="Model", barmode="group",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_comp.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig_comp, use_container_width=True)

    # --- 2. ERROR ANALYSIS (Confusion Matrices) ---
    st.header("🧠 2. Error Analysis & Bias Detection")
    st.markdown("""
    Evaluating the confusion matrices helps identify where models struggle. 
    For example, ML often confuses **Neutral** with **Positive** in political context.
    """)
    
    cm_tabs = st.tabs(["ML Errors", "DL Errors", "Hybrid Errors"])
    labels = ["Negative", "Neutral", "Positive"]
    
    with cm_tabs[0]:
        st.plotly_chart(plot_confusion_matrix(all_metrics['ML']['confusion_matrix'], labels, "ML Confusion Matrix"), use_container_width=True)
    with cm_tabs[1]:
        st.plotly_chart(plot_confusion_matrix(all_metrics['DL']['confusion_matrix'], labels, "DL Confusion Matrix"), use_container_width=True)
    with cm_tabs[2]:
        st.plotly_chart(plot_confusion_matrix(all_metrics['Hybrid']['confusion_matrix'], labels, "Hybrid Confusion Matrix"), use_container_width=True)

    # --- 3. SENTIMENT INSIGHTS ---
    st.header("📈 3. Sentiment Insights")
    
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        # Time Series
        filtered_df['month'] = filtered_df['timestamp'].dt.to_period('M').astype(str)
        time_df = filtered_df.groupby(['month', 'final_sentiment']).size().reset_index(name='count')
        fig_time = px.line(time_df, x='month', y='count', color='final_sentiment', 
                           title="Monthly Sentiment Trend", color_discrete_map={"positive": "#2ca02c", "neutral": "#7f7f7f", "negative": "#d62728"}, markers=True)
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col_b:
        # Pie Chart
        fig_pie = px.pie(filtered_df, names="final_sentiment", title="Final Sentiment Distribution",
                         color="final_sentiment", color_discrete_map={"positive": "#2ca02c", "neutral": "#7f7f7f", "negative": "#d62728"})
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- 4. ENTITY & CONTEXTUAL ANALYSIS ---
    st.header("🔍 4. Entity & Contextual Analysis")
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        pol_sent = filtered_df.groupby(['politician', 'final_sentiment']).size().reset_index(name='count')
        fig_pol = px.bar(pol_sent, x='politician', y='count', color='final_sentiment', 
                         title="Sentiment by Politician", barmode='group')
        st.plotly_chart(fig_pol, use_container_width=True)
        
    with col_e2:
        # Word Cloud
        st.subheader("Top Keywords")
        text = " ".join(filtered_df["cleaned_text"].dropna())
        if text:
            wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate(text)
            fig_wc, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig_wc)

    # --- 5. DATA QUALITY & BIAS ---
    st.header("📊 5. Data Quality & Bias Detection")
    dq_col1, dq_col2 = st.columns(2)
    with dq_col1:
        st.subheader("Source Distribution")
        target_df = filtered_df if "source" in filtered_df.columns else df
        if "source" in target_df.columns:
            source_counts = target_df["source"].value_counts()
            fig_source = px.bar(source_counts, orientation='h', title="Samples by Source", color=source_counts.index)
            st.plotly_chart(fig_source, use_container_width=True)
        else:
            st.info("Source information not available in this dataset.")
    
    with dq_col2:
        st.subheader("Confidence Calibration")
        fig_conf = px.histogram(df, x="final_confidence", nbins=20, title="Confidence Distribution", color="final_sentiment")
        st.plotly_chart(fig_conf, use_container_width=True)

    # --- 6. EXPERT ANALYSIS (TEXTUAL INSIGHTS) ---
    st.header("📜 6. Expert Analysis & Recommendations")
    
    st.markdown(f"""
    <div class="insight-card">
        <h4>Why the Hybrid Model Wins</h4>
        <p>The <b>Hybrid Model</b> (weighted 30% ML / 70% DL) leverages the statistical speed of TF-IDF and the contextual depth of <b>RoBERTa-Transformer</b>.</p>
        <ul>
            <li><b>ML Strengths:</b> Excellent at identifying explicit sentiment markers (e.g., "awful", "great") but fails at sarcasm.</li>
            <li><b>DL Strengths:</b> Understands political nuances, negation, and complex sentence structures.</li>
            <li><b>Performance Gain:</b> The hybrid ensemble showed a <b>{ (all_metrics['Hybrid']['f1'] - all_metrics['ML']['f1'])*100:.1f}%</b> F1-score improvement over the ML baseline.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### Best Practices for Future Deployment:
    1. **Address Data Leakage:** Ensure training data doesn't contain tokens that are specific to the test set (e.g., specific dates or IDs).
    2. **Regular Retraining:** Political sentiment shifts rapidly. Models should be fine-tuned on recent data every 30 days.
    3. **Confidence Thresholding:** For automated moderation, only trust predictions with confidence > 85%.
    """)

    # --- Data Explorer ---
    with st.expander("📂 View Raw Data & Confidence Scores"):
        st.dataframe(filtered_df[["text", "politician", "party", "true_sentiment", "final_sentiment", "final_confidence"]].head(100), use_container_width=True)

if __name__ == "__main__":
    main()
