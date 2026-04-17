import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

# =========================================================
# Page Config
# =========================================================
st.set_page_config(
    page_title="Financial NLP Dashboard",
    page_icon="📊",
    layout="wide",
)

# =========================================================
# Editable Config
# =========================================================
SENTIMENT_MODEL_NAME = "sunpeishan/finetuned-finbert-sentiment-plp"
TOPIC_MODEL_URL = "https://huggingface.co/huiwen999/BERTopic/resolve/main/model_bundle.pkl"
ENABLE_TOPIC_MODEL = True  # topic model is loaded from a Hugging Face file URL

# Optional label maps for topic model output
TOPIC_ID_TO_NAME = {
    0: "Cybersecurity Risk",
    1: "Revenue Recognition",
    5: "Contingent Liability Risk",
    6: "Litigation Risk",
    8: "Tax Uncertainty Risk",
    12: "Credit / Covenant Risk",
    16: "Financing / Debt Risk",
}

TOPIC_ID_TO_GROUP = {
    0: "Risk",
    1: "Non-risk",
    5: "Risk",
    6: "Risk",
    8: "Risk",
    12: "Risk",
    16: "Risk",
}

# =========================================================
# Styling
# =========================================================
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    }
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1300px;
    }
    .hero {
        padding: 1.5rem 1.6rem;
        border-radius: 22px;
        background: white;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem 1.2rem;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        text-align: center;
    }
    .section-card {
        background: white;
        padding: 1.2rem;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    }
    .small-note {
        color: #475569;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Cached Loaders
# =========================================================
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model=SENTIMENT_MODEL_NAME)


@st.cache_resource
def load_topic_model_from_url():
    if not ENABLE_TOPIC_MODEL:
        return None

    import requests
    import pickle
    import tempfile
    from pathlib import Path

    response = requests.get(TOPIC_MODEL_URL, timeout=120)
    response.raise_for_status()

    temp_dir = Path(tempfile.gettempdir()) / "financial_nlp_dashboard"
    temp_dir.mkdir(parents=True, exist_ok=True)
    model_path = temp_dir / "model_bundle.pkl"
    model_path.write_bytes(response.content)

    with open(model_path, "rb") as f:
        topic_model = pickle.load(f)

    return topic_model


sentiment_pipeline = load_sentiment_pipeline()
topic_model = load_topic_model_from_url()

# =========================================================
# Helpers
# =========================================================
def normalize_sentiment_label(label: str) -> str:
    label = str(label).lower()
    mapping = {
        "positive": "Positive",
        "negative": "Negative",
        "neutral": "Neutral",
        "label_0": "Negative",
        "label_1": "Neutral",
        "label_2": "Positive",
    }
    return mapping.get(label, label.title())


def predict_sentiment(text: str):
    result = sentiment_pipeline(text)[0]
    return {
        "label": normalize_sentiment_label(result["label"]),
        "score": float(result["score"]),
    }


def normalize_topic_output(raw_result: dict):
    raw_label = raw_result.get("label", "Unknown")
    raw_score = float(raw_result.get("score", 0.0))

    # If topic model returns LABEL_0 / numeric id strings, try mapping to business label
    topic_name = raw_label
    topic_group = "Unknown"

    try:
        if str(raw_label).lower().startswith("label_"):
            topic_id = int(str(raw_label).split("_")[-1])
            topic_name = TOPIC_ID_TO_NAME.get(topic_id, f"Topic {topic_id}")
            topic_group = TOPIC_ID_TO_GROUP.get(topic_id, "Unknown")
        elif str(raw_label).isdigit():
            topic_id = int(raw_label)
            topic_name = TOPIC_ID_TO_NAME.get(topic_id, f"Topic {topic_id}")
            topic_group = TOPIC_ID_TO_GROUP.get(topic_id, "Unknown")
        else:
            topic_group = "Risk" if "risk" in str(raw_label).lower() else "Non-risk"
    except Exception:
        pass

    return {
        "label": topic_name,
        "group": topic_group,
        "score": raw_score,
    }


def predict_topic(text: str):
    if topic_model is None:
        return {
            "label": "Topic model placeholder",
            "group": "Pending",
            "score": 0.0,
        }

    try:
        topics, probs = topic_model.transform([text])
        topic_id = topics[0]
        topic_name = TOPIC_ID_TO_NAME.get(topic_id, f"Topic {topic_id}")
        topic_group = TOPIC_ID_TO_GROUP.get(topic_id, "Unknown")

        topic_score = 0.0
        if probs is not None:
            try:
                topic_score = float(max(probs[0])) if hasattr(probs[0], "__iter__") else float(probs[0])
            except Exception:
                topic_score = 0.0

        return {
            "label": topic_name,
            "group": topic_group,
            "score": topic_score,
        }
    except Exception:
        return {
            "label": "Topic inference failed",
            "group": "Unknown",
            "score": 0.0,
        }


def analyze_text(text: str):
    sentiment = predict_sentiment(text)
    topic = predict_topic(text)

    risk_signal = "High"
    if sentiment["label"] == "Positive":
        risk_signal = "Low"
    elif sentiment["label"] == "Neutral":
        risk_signal = "Medium"

    if topic["group"] == "Risk" and sentiment["label"] == "Negative":
        risk_signal = "High"

    return {
        "text": text,
        "sentiment": sentiment["label"],
        "sentiment_confidence": sentiment["score"],
        "topic": topic["label"],
        "topic_group": topic["group"],
        "topic_confidence": topic["score"],
        "risk_signal": risk_signal,
    }


def batch_analyze(df: pd.DataFrame, text_col: str):
    rows = []
    for text in df[text_col].fillna("").astype(str):
        if text.strip():
            rows.append(analyze_text(text))
    return pd.DataFrame(rows)


def plot_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, ylim=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df[x_col], df[y_col])
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if ylim is not None:
        ax.set_ylim(*ylim)
    plt.xticks(rotation=20, ha="right")
    st.pyplot(fig)

# =========================================================
# Header
# =========================================================
st.markdown(
    f"""
    <div class="hero">
        <h1 style="margin-bottom:0.4rem;">📊 Financial NLP Dashboard</h1>
        <p class="small-note">
            Front-end dashboard for financial text analysis. The sentiment module is already connected to Peishan's Hugging Face model,
            and the topic modeling module is reserved for a future Hugging Face deployment.
        </p>
        <p class="small-note"><b>Sentiment model:</b> {SENTIMENT_MODEL_NAME}</p>
        <p class="small-note"><b>Topic model:</b> {TOPIC_MODEL_URL if ENABLE_TOPIC_MODEL else 'Placeholder mode (not enabled yet)'}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Model Configuration")
    st.write("**Sentiment**")
    st.code(SENTIMENT_MODEL_NAME)
    st.write("**Topic modeling**")
    st.code(TOPIC_MODEL_URL if ENABLE_TOPIC_MODEL else "Waiting for Hugging Face topic model")
    st.markdown("---")
    st.write("Recommended workflow:")
    st.write("1. Keep sentiment live now")
    st.write("2. Swap topic placeholder later")
    st.write("3. Use batch upload for final presentation")

# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "Live Analysis",
    "Batch Analysis",
    "System Overview",
])

# =========================================================
# Tab 1 - Live Analysis
# =========================================================
with tab1:
    left, right = st.columns([1.15, 1])

    with left:
        st.subheader("Input Financial Text")
        text = st.text_area(
            "Text",
            value="The company faces elevated litigation exposure and continuing regulatory uncertainty.",
            height=180,
            label_visibility="collapsed",
        )
        run_live = st.button("Run Analysis", use_container_width=True)

    with right:
        st.subheader("Prediction Output")
        if run_live:
            if not text.strip():
                st.warning("Please enter some text first.")
            else:
                result = analyze_text(text)

                c1, c2, c3 = st.columns(3)
                c1.markdown(f'<div class="metric-card"><h3>{result["sentiment"]}</h3><p>Sentiment</p></div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="metric-card"><h3>{result["topic"]}</h3><p>Topic</p></div>', unsafe_allow_html=True)
                c3.markdown(f'<div class="metric-card"><h3>{result["risk_signal"]}</h3><p>Risk Signal</p></div>', unsafe_allow_html=True)

                st.markdown("### Confidence Overview")
                conf_df = pd.DataFrame(
                    {
                        "Module": ["Sentiment", "Topic"],
                        "Confidence": [result["sentiment_confidence"], result["topic_confidence"]],
                    }
                )
                plot_bar(conf_df, "Module", "Confidence", "Module Confidence", ylim=(0, 1))

                st.markdown("### Detailed Output")
                st.dataframe(pd.DataFrame([result]), use_container_width=True)
        else:
            st.markdown(
                '<div class="section-card"><p class="small-note">Run the dashboard to see sentiment prediction now. Topic modeling will activate once your Hugging Face topic model is ready.</p></div>',
                unsafe_allow_html=True,
            )

# =========================================================
# Tab 2 - Batch Analysis
# =========================================================
with tab2:
    st.subheader("Batch CSV Analysis")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview")
        st.dataframe(df.head(), use_container_width=True)

        candidate_cols = [c for c in df.columns if c.lower() in ["text", "sentence", "content", "paragraph", "headline", "notes"]]
        default_idx = list(df.columns).index(candidate_cols[0]) if candidate_cols else 0
        text_col = st.selectbox("Select text column", list(df.columns), index=default_idx)

        if st.button("Run Batch Analysis", use_container_width=True):
            result_df = batch_analyze(df, text_col)

            if result_df.empty:
                st.warning("No valid text found in selected column.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(f'<div class="metric-card"><h3>{len(result_df)}</h3><p>Total Records</p></div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="metric-card"><h3>{(result_df["sentiment"] == "Negative").sum()}</h3><p>Negative</p></div>', unsafe_allow_html=True)
                c3.markdown(f'<div class="metric-card"><h3>{(result_df["topic_group"] == "Risk").sum()}</h3><p>Risk Topics</p></div>', unsafe_allow_html=True)
                c4.markdown(f'<div class="metric-card"><h3>{result_df["topic"].nunique()}</h3><p>Unique Topics</p></div>', unsafe_allow_html=True)

                st.markdown("### Output Table")
                st.dataframe(result_df, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    sent_dist = result_df["sentiment"].value_counts().reset_index()
                    sent_dist.columns = ["sentiment", "count"]
                    plot_bar(sent_dist, "sentiment", "count", "Sentiment Distribution")

                with col2:
                    topic_dist = result_df["topic"].value_counts().head(10).reset_index()
                    topic_dist.columns = ["topic", "count"]
                    plot_bar(topic_dist, "topic", "count", "Top Topic Distribution")

                csv_data = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Results CSV",
                    csv_data,
                    "financial_nlp_results.csv",
                    "text/csv",
                    use_container_width=True,
                )
    else:
        st.info("Upload a CSV file containing one text column for batch analysis.")

# =========================================================
# Tab 3 - System Overview
# =========================================================
with tab3:
    st.subheader("System Overview")
    st.markdown(
        """
        ### Current integration
        - **Sentiment module**: live Hugging Face model from Peishan
        - **Topic module**: remote BERTopic pickle loaded from a Hugging Face file URL
        - **UI layer**: Streamlit dashboard for live demo and batch presentation

        ### How to activate topic modeling later
        1. Keep the current Hugging Face `model_bundle.pkl` file URL
        2. Make sure the environment includes BERTopic dependencies
        3. Adjust topic label mapping if needed
        4. Re-run the app to activate topic inference

        ### Suggested final demo flow
        1. Enter one financial sentence and run live analysis
        2. Show sentiment output from FinBERT
        3. Show topic section as placeholder now, or live once deployed
        4. Upload CSV and present distribution charts
        """
    )
