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
# Model Config
# =========================================================
SENTIMENT_MODEL_NAME = "sunpeishan/finetuned-finbert-sentiment-plp"
TOPIC_MODEL_REPO_ID = "huiwen999/BERTopic"
ENABLE_TOPIC_MODEL = True

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
        min-height: 115px;
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
    return pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL_NAME,
    )


@st.cache_resource
def load_topic_model_from_hf():
    """
    Load BERTopic model and topic mappings from Hugging Face Hub.
    This uses BERTopic.load(repo_id), not pickle.load(model_bundle.pkl).
    """
    if not ENABLE_TOPIC_MODEL:
        return None, {}, {}

    try:
        import json
        from bertopic import BERTopic
        from huggingface_hub import hf_hub_download

        model = BERTopic.load(TOPIC_MODEL_REPO_ID)

        label_path = hf_hub_download(
            repo_id=TOPIC_MODEL_REPO_ID,
            filename="topic_label_map.json",
            repo_type="model",
        )

        group_path = hf_hub_download(
            repo_id=TOPIC_MODEL_REPO_ID,
            filename="group_map.json",
            repo_type="model",
        )

        with open(label_path, "r", encoding="utf-8") as f:
            topic_label_map = json.load(f)

        with open(group_path, "r", encoding="utf-8") as f:
            group_map = json.load(f)

        return model, topic_label_map, group_map

    except Exception as e:
        st.warning(
            f"Topic model loading failed: {type(e).__name__}. "
            "The app will continue without topic inference."
        )
        return None, {}, {}


sentiment_pipeline = load_sentiment_pipeline()
topic_model, topic_label_map, group_map = load_topic_model_from_hf()

# =========================================================
# Helper Functions
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


def predict_topic(text: str):
    if topic_model is None:
        return {
            "label": "Topic model unavailable",
            "group": "Pending",
            "score": 0.0,
        }

    try:
        topics, probs = topic_model.transform([text])
        topic_id = int(topics[0])

        topic_name = topic_label_map.get(str(topic_id), f"Topic {topic_id}")
        topic_group = group_map.get(str(topic_id), "Unknown")

        topic_score = 0.0
        if probs is not None:
            try:
                topic_score = float(max(probs[0]))
            except Exception:
                topic_score = 0.0

        return {
            "label": topic_name,
            "group": topic_group,
            "score": topic_score,
        }

    except Exception as e:
        return {
            "label": f"Topic inference failed: {type(e).__name__}",
            "group": "Unknown",
            "score": 0.0,
        }


def analyze_text(text: str):
    sentiment = predict_sentiment(text)
    topic = predict_topic(text)

    risk_signal = "Medium"

    if sentiment["label"] == "Positive":
        risk_signal = "Low"
    elif sentiment["label"] == "Negative":
        risk_signal = "High"

    if topic["group"] == "Risk" and sentiment["label"] == "Negative":
        risk_signal = "High"
    elif topic["group"] == "Risk" and sentiment["label"] == "Neutral":
        risk_signal = "Medium"

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
            This dashboard integrates Peishan's fine-tuned FinBERT sentiment model and Huiwen's BERTopic model
            for financial text analysis.
        </p>
        <p class="small-note"><b>Sentiment model:</b> {SENTIMENT_MODEL_NAME}</p>
        <p class="small-note"><b>Topic model:</b> {TOPIC_MODEL_REPO_ID}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.header("Model Configuration")

    st.write("**Sentiment Model**")
    st.code(SENTIMENT_MODEL_NAME)

    st.write("**Topic Model**")
    st.code(TOPIC_MODEL_REPO_ID)

    st.markdown("---")
    st.subheader("Model Status")

    if sentiment_pipeline is not None:
        st.success("Peishan FinBERT loaded")
    else:
        st.error("Peishan FinBERT failed")

    if topic_model is not None:
        st.success("BERTopic model loaded")
    else:
        st.warning("BERTopic model not loaded")

    st.markdown("---")
    st.write("Dashboard modules:")
    st.write("1. Live text analysis")
    st.write("2. Batch CSV analysis")
    st.write("3. System overview")


# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3 = st.tabs(
    [
        "Live Analysis",
        "Batch Analysis",
        "System Overview",
    ]
)

# =========================================================
# Tab 1 - Live Analysis
# =========================================================
with tab1:
    left, right = st.columns([1.15, 1])

    with left:
        st.subheader("Input Financial Text")

        text = st.text_area(
            "Text",
            value="The company faces cybersecurity risk and continuing regulatory uncertainty.",
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

                c1.markdown(
                    f"""
                    <div class="metric-card">
                        <h3>{result["sentiment"]}</h3>
                        <p>Sentiment</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                c2.markdown(
                    f"""
                    <div class="metric-card">
                        <h3>{result["topic"]}</h3>
                        <p>Topic</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                c3.markdown(
                    f"""
                    <div class="metric-card">
                        <h3>{result["risk_signal"]}</h3>
                        <p>Risk Signal</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("### Confidence Overview")

                conf_df = pd.DataFrame(
                    {
                        "Module": ["Sentiment", "Topic"],
                        "Confidence": [
                            result["sentiment_confidence"],
                            result["topic_confidence"],
                        ],
                    }
                )

                plot_bar(
                    conf_df,
                    "Module",
                    "Confidence",
                    "Module Confidence",
                    ylim=(0, 1),
                )

                st.markdown("### Detailed Output")
                st.dataframe(pd.DataFrame([result]), use_container_width=True)

        else:
            st.markdown(
                """
                <div class="section-card">
                    <p class="small-note">
                        Enter a financial sentence and click Run Analysis to see sentiment,
                        topic, and risk signal outputs.
                    </p>
                </div>
                """,
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

        candidate_cols = [
            c
            for c in df.columns
            if c.lower()
            in ["text", "sentence", "content", "paragraph", "headline", "notes"]
        ]

        default_idx = list(df.columns).index(candidate_cols[0]) if candidate_cols else 0

        text_col = st.selectbox(
            "Select text column",
            list(df.columns),
            index=default_idx,
        )

        if st.button("Run Batch Analysis", use_container_width=True):
            result_df = batch_analyze(df, text_col)

            if result_df.empty:
                st.warning("No valid text found in selected column.")
            else:
                c1, c2, c3, c4 = st.columns(4)

                c1.markdown(
                    f"""
                    <div class="metric-card">
                        <h3>{len(result_df)}</h3>
                        <p>Total Records</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                c2.markdown(
                    f"""
                    <div class="metric-card">
                        <h3>{(result_df["sentiment"] == "Negative").sum()}</h3>
                        <p>Negative</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                c3.markdown(
                    f"""
                    <div class="metric-card">
                        <h3>{(result_df["topic_group"] == "Risk").sum()}</h3>
                        <p>Risk Topics</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                c4.markdown(
                    f"""
                    <div class="metric-card">
                        <h3>{result_df["topic"].nunique()}</h3>
                        <p>Unique Topics</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("### Output Table")
                st.dataframe(result_df, use_container_width=True)

                col1, col2 = st.columns(2)

                with col1:
                    sent_dist = result_df["sentiment"].value_counts().reset_index()
                    sent_dist.columns = ["sentiment", "count"]

                    plot_bar(
                        sent_dist,
                        "sentiment",
                        "count",
                        "Sentiment Distribution",
                    )

                with col2:
                    topic_dist = (
                        result_df["topic"]
                        .value_counts()
                        .head(10)
                        .reset_index()
                    )
                    topic_dist.columns = ["topic", "count"]

                    plot_bar(
                        topic_dist,
                        "topic",
                        "count",
                        "Top Topic Distribution",
                    )

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
        ### Current Integration

        - **Sentiment module**: Peishan's fine-tuned FinBERT model from Hugging Face.
        - **Topic module**: Huiwen's BERTopic model loaded from Hugging Face using `BERTopic.load()`.
        - **UI layer**: Streamlit dashboard for live demo and batch analysis.

        ### Pipeline Logic

        1. User enters or uploads financial text.
        2. FinBERT classifies the sentence sentiment as Positive, Neutral, or Negative.
        3. BERTopic identifies the underlying business topic.
        4. The dashboard maps the topic id to a readable business label.
        5. The system combines sentiment and topic group to produce a risk signal.

        ### Business Value

        This dashboard converts unstructured financial text into structured indicators,
        helping analysts quickly identify sentiment polarity, topic category, and potential
        risk signals from financial disclosures.
        """
    )
