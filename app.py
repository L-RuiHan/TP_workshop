import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

# =========================================================
# Page Config
# =========================================================
st.set_page_config(
    page_title="Financial NLP Risk Dashboard",
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
# Hugging Face Dataset Config
# =========================================================
HF_DATASET_REPO_ID = "LRH865/PLP_workshop"

COMPANY_RSI_FILE = "Step4_Company_Level_RSI.csv"
MASTER_DATA_FILE = "Step5_Master_Data_Sentiment_Topic_Sample.csv"
TOPIC_RSI_FILE = "Step6_Topic_Level_RSI.csv"
MARKET_TOPIC_RSI_FILE = "Step7_Macro_Market_Topic_RSI.csv"

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
    .risk-sentence-card {
        background: white;
        padding: 1rem 1.2rem;
        border-left: 6px solid #ef4444;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.07);
        margin-bottom: 1rem;
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
# Cached Model Loaders
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


# =========================================================
# Cached Data Loaders from Hugging Face Dataset
# =========================================================
@st.cache_data
def load_csv_from_hf(filename: str):
    """
    Load CSV files from Hugging Face Dataset.
    Since the dataset is public, no HF_TOKEN is required.
    """
    try:
        from huggingface_hub import hf_hub_download

        file_path = hf_hub_download(
            repo_id=HF_DATASET_REPO_ID,
            filename=filename,
            repo_type="dataset",
        )

        return pd.read_csv(file_path)

    except Exception as e:
        st.warning(
            f"Failed to load {filename} from Hugging Face Dataset: "
            f"{type(e).__name__}: {e}"
        )
        return pd.DataFrame()


@st.cache_data
def load_company_rsi_data():
    return load_csv_from_hf(COMPANY_RSI_FILE)


@st.cache_data
def load_master_sentence_data():
    return load_csv_from_hf(MASTER_DATA_FILE)


@st.cache_data
def load_topic_rsi_data():
    return load_csv_from_hf(TOPIC_RSI_FILE)


@st.cache_data
def load_market_topic_rsi_data():
    return load_csv_from_hf(MARKET_TOPIC_RSI_FILE)


# =========================================================
# Load Models
# =========================================================
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


def clean_company_name(name):
    return str(name).upper().strip()


def safe_round(value, digits=2):
    try:
        return round(float(value), digits)
    except Exception:
        return "N/A"


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


def prepare_company_data(company_rsi, master_df, topic_rsi):
    if not company_rsi.empty and "name" in company_rsi.columns:
        company_rsi = company_rsi.copy()
        company_rsi["name_clean"] = company_rsi["name"].apply(clean_company_name)

    if not master_df.empty and "name" in master_df.columns:
        master_df = master_df.copy()
        master_df["name_clean"] = master_df["name"].apply(clean_company_name)

    if not topic_rsi.empty and "name" in topic_rsi.columns:
        topic_rsi = topic_rsi.copy()
        topic_rsi["name_clean"] = topic_rsi["name"].apply(clean_company_name)

    return company_rsi, master_df, topic_rsi


def get_top_risk_sentences(company_sentences: pd.DataFrame, top_n=3):
    """
    Ranking logic:
    1. Risk group + Negative sentiment
    2. Risk group only
    3. Negative sentiment only
    4. Fallback to all records
    """
    if company_sentences.empty:
        return pd.DataFrame()

    df = company_sentences.copy()

    if "group" not in df.columns:
        df["group"] = ""

    if "sentiment_label" not in df.columns:
        df["sentiment_label"] = ""

    if "sentiment_probability" not in df.columns:
        df["sentiment_probability"] = 0

    df["group_clean"] = df["group"].astype(str).str.lower().str.strip()
    df["sentiment_clean"] = df["sentiment_label"].astype(str).str.lower().str.strip()
    df["sentiment_probability"] = pd.to_numeric(
        df["sentiment_probability"],
        errors="coerce",
    ).fillna(0)

    top_df = df[
        (df["group_clean"] == "risk") &
        (df["sentiment_clean"] == "negative")
    ].copy()

    if top_df.empty:
        top_df = df[df["group_clean"] == "risk"].copy()

    if top_df.empty:
        top_df = df[df["sentiment_clean"] == "negative"].copy()

    if top_df.empty:
        top_df = df.copy()

    return top_df.sort_values(
        by="sentiment_probability",
        ascending=False,
    ).head(top_n)


# =========================================================
# Header
# =========================================================
st.markdown(
    f"""
    <div class="hero">
        <h1 style="margin-bottom:0.4rem;">📊 Financial NLP Risk Dashboard</h1>
        <p class="small-note">
            This dashboard integrates a fine-tuned FinBERT sentiment model, a BERTopic model,
            and company-level RSI outputs to support financial risk monitoring.
        </p>
        <p class="small-note"><b>Sentiment model:</b> {SENTIMENT_MODEL_NAME}</p>
        <p class="small-note"><b>Topic model:</b> {TOPIC_MODEL_REPO_ID}</p>
        <p class="small-note"><b>Dataset:</b> {HF_DATASET_REPO_ID}</p>
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

    st.write("**Dataset Repo**")
    st.code(HF_DATASET_REPO_ID)

    st.markdown("---")
    st.subheader("Model Status")

    if sentiment_pipeline is not None:
        st.success("FinBERT loaded")
    else:
        st.error("FinBERT failed")

    if topic_model is not None:
        st.success("BERTopic model loaded")
    else:
        st.warning("BERTopic model not loaded")

    st.markdown("---")
    st.write("Dashboard modules:")
    st.write("1. Live text analysis")
    st.write("2. Company risk explorer")


# =========================================================
# Tabs
# =========================================================
tab1, tab2 = st.tabs(
    [
        "Live Analysis",
        "Company Risk Explorer",
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
# Tab 2 - Company Risk Explorer
# =========================================================
with tab2:
    st.subheader("Company Risk Explorer")

    st.markdown(
        """
        Search a company to review its firm-level RSI, sentiment distribution,
        topic-level risk ranking, and the top 3 highest-risk disclosure sentences.
        """
    )

    company_rsi = load_company_rsi_data()
    master_df = load_master_sentence_data()
    topic_rsi = load_topic_rsi_data()
    market_topic_rsi = load_market_topic_rsi_data()

    company_rsi, master_df, topic_rsi = prepare_company_data(
        company_rsi,
        master_df,
        topic_rsi,
    )

    if master_df.empty:
        st.error(
            "Master sentence-level data is missing. "
            "Please check the Hugging Face dataset file: "
            f"{MASTER_DATA_FILE}"
        )

    elif "name_clean" not in master_df.columns:
        st.error("The master data must contain a company name column named 'name'.")

    else:
        company_options = sorted(master_df["name_clean"].dropna().unique())

        search_text = st.text_input(
            "Search company name",
            placeholder="Example: ALTO INGREDIENTS, INC.",
        )

        if search_text.strip():
            search_clean = clean_company_name(search_text)
            filtered_options = [
                c for c in company_options if search_clean in c
            ]
        else:
            filtered_options = company_options

        if not filtered_options:
            st.warning("No matching company found. Please try another company name.")

        else:
            selected_company = st.selectbox(
                "Select a company",
                filtered_options,
            )

            company_sentences = master_df[
                master_df["name_clean"] == selected_company
            ].copy()

            company_overview = pd.DataFrame()
            if not company_rsi.empty and "name_clean" in company_rsi.columns:
                company_overview = company_rsi[
                    company_rsi["name_clean"] == selected_company
                ].copy()

            company_topics = pd.DataFrame()
            if not topic_rsi.empty and "name_clean" in topic_rsi.columns:
                company_topics = topic_rsi[
                    topic_rsi["name_clean"] == selected_company
                ].copy()

            # =================================================
            # 1. Company Overview Metrics
            # =================================================
            st.markdown("### 1. Company Overview Metrics")

            if not company_overview.empty:
                row = company_overview.iloc[0]

                c1, c2, c3, c4 = st.columns(4)

                c1.markdown(
                    f"""
                    <div class="metric-card">
                        <h3>{int(row.get("Total_Sentences", 0))}</h3>
                        <p>Total Sentences</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                c2.markdown(
                    f"""
                    <div class="metric-card">
                        <h3>{safe_round(row.get("Sum_Negative_Prob", 0), 2)}</h3>
                        <p>Negative Probability Sum</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                c3.markdown(
                    f"""
                    <div class="metric-card">
                        <h3>{safe_round(row.get("RSI_Raw", 0), 2)}</h3>
                        <p>Raw RSI</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                c4.markdown(
                    f"""
                    <div class="metric-card">
                        <h3>{safe_round(row.get("RSI_Score_100", 0), 2)}</h3>
                        <p>RSI Score / 100</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:
                st.info(
                    "No company-level RSI record found. "
                    "The app will still show sentence-level records."
                )

            # =================================================
            # 2. Sentiment and Topic Summary
            # =================================================
            st.markdown("### 2. Sentiment and Topic Summary")

            if not company_sentences.empty:
                left, right = st.columns(2)

                with left:
                    if "sentiment_label" in company_sentences.columns:
                        sentiment_dist = (
                            company_sentences["sentiment_label"]
                            .fillna("Unknown")
                            .value_counts()
                            .reset_index()
                        )
                        sentiment_dist.columns = ["sentiment", "count"]

                        plot_bar(
                            sentiment_dist,
                            "sentiment",
                            "count",
                            "Sentiment Distribution",
                        )
                    else:
                        st.info("No sentiment_label column found.")

                with right:
                    if "topic_label" in company_sentences.columns:
                        topic_dist = (
                            company_sentences["topic_label"]
                            .fillna("Unknown")
                            .value_counts()
                            .head(10)
                            .reset_index()
                        )
                        topic_dist.columns = ["topic", "count"]

                        plot_bar(
                            topic_dist,
                            "topic",
                            "count",
                            "Top 10 Topic Distribution",
                        )
                    else:
                        st.info("No topic_label column found.")

            # =================================================
            # 3. Top 3 Highest-Risk Disclosure Sentences
            # =================================================
            st.markdown("### 3. Top 3 Highest-Risk Disclosure Sentences")

            top_risk_sentences = get_top_risk_sentences(company_sentences, top_n=3)

            if top_risk_sentences.empty:
                st.info("No related sentence records found for this company.")

            else:
                for idx, (_, row) in enumerate(top_risk_sentences.iterrows(), start=1):
                    sentence = row.get("sentence", "")
                    topic_label = row.get("topic_label", "N/A")
                    topic_group = row.get("group", "N/A")
                    sentiment_label = row.get("sentiment_label", "N/A")
                    sentiment_probability = safe_round(
                        row.get("sentiment_probability", 0),
                        4,
                    )
                    period = row.get("period", "N/A")
                    form = row.get("form", "N/A")
                    tag = row.get("tag", "N/A")

                    st.markdown(
                        f"""
                        <div class="risk-sentence-card">
                            <h4>Risk Sentence {idx}</h4>
                            <p><b>Topic:</b> {topic_label} | <b>Group:</b> {topic_group}</p>
                            <p><b>Sentiment:</b> {sentiment_label} | <b>Probability:</b> {sentiment_probability}</p>
                            <p><b>Period:</b> {period} | <b>Form:</b> {form}</p>
                            <p><b>Tag:</b> {tag}</p>
                            <p style="margin-top:0.8rem;">{sentence}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # =================================================
            # 4. Company Topic-Level Risk Ranking
            # =================================================
            st.markdown("### 4. Company Topic-Level Risk Ranking")

            if company_topics.empty:
                st.info("No topic-level RSI data found for this company.")

            else:
                topic_cols = [
                    "topic",
                    "topic_label",
                    "group",
                    "Topic_Neg_Prob_Sum",
                    "Total_Sentences_Company",
                    "Topic_RSI_Raw",
                    "Company_Internal_Risk_Rank",
                ]

                topic_cols = [c for c in topic_cols if c in company_topics.columns]

                company_topics_display = company_topics.copy()

                if "Company_Internal_Risk_Rank" in company_topics_display.columns:
                    company_topics_display = company_topics_display.sort_values(
                        "Company_Internal_Risk_Rank"
                    )

                st.dataframe(
                    company_topics_display[topic_cols],
                    use_container_width=True,
                )

            # =================================================
            # 5. Market Topic Benchmark
            # =================================================
            st.markdown("### 5. Market Topic Benchmark")

            if (
                not market_topic_rsi.empty
                and not company_topics.empty
                and "topic" in company_topics.columns
                and "topic" in market_topic_rsi.columns
            ):
                benchmark_df = company_topics.merge(
                    market_topic_rsi,
                    on=["topic", "topic_label", "group"],
                    how="left",
                )

                benchmark_cols = [
                    "topic",
                    "topic_label",
                    "group",
                    "Topic_RSI_Raw",
                    "Company_Internal_Risk_Rank",
                    "Market_Topic_RSI",
                    "Market_Risk_Rank",
                ]

                benchmark_cols = [
                    c for c in benchmark_cols if c in benchmark_df.columns
                ]

                st.dataframe(
                    benchmark_df[benchmark_cols],
                    use_container_width=True,
                )

            else:
                st.info(
                    "Market benchmark is unavailable. "
                    "Please check Step7_Macro_Market_Topic_RSI.csv."
                )

            # =================================================
            # 6. Related Sentence-Level Records
            # =================================================
            st.markdown("### 6. Related Sentence-Level Records")

            display_cols = [
                "adsh",
                "name",
                "period",
                "form",
                "tag",
                "sentence",
                "word_count",
                "sentiment_label",
                "sentiment_probability",
                "topic",
                "topic_label",
                "group",
            ]

            display_cols = [c for c in display_cols if c in company_sentences.columns]

            st.dataframe(
                company_sentences[display_cols],
                use_container_width=True,
            )

            csv_download = company_sentences[display_cols].to_csv(
                index=False,
            ).encode("utf-8")

            st.download_button(
                "Download This Company's Records",
                csv_download,
                f"{selected_company}_risk_records.csv",
                "text/csv",
                use_container_width=True,
            )
