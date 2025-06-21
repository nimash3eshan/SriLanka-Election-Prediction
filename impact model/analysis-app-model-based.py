import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import re
from textblob import TextBlob
import emoji

st.set_page_config(page_title="Predict Tweet Impact (XGBoost Model)", layout="wide")
st.title("📊 Predict Social Media Impact of Sri Lankan Politicians")

# --- Helper functions ---
singlish_happy = {
    'patta': 2, 'maru': 2, 'niyamai': 2, 'ela': 2, 'supiri': 2, 'gathi': 2,
    'hodai': 1, 'lassanai': 1, 'shok': 1, 'jayawewa': 2, 'good':1, 'love':2
}
singlish_sad = {
    'apalai': -2, 'chaa': -2, 'anthimai': -2, 'weda na': -2, 'boring': -1,
    'boru': -1, 'harak': -2, 'gon': -2, 'pissu': -2, 'sad':-1,'bad':-1
}

def clean_text(text):
    if text is None:
        return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = emoji.demojize(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

def calculate_custom_polarity(text):
    score = 0
    words = text.split()
    for word in words:
        score += singlish_happy.get(word, 0)
        score += singlish_sad.get(word, 0)
    return score

def get_hybrid_sentiment(text, custom_score):
    textblob_polarity = TextBlob(text).sentiment.polarity
    hybrid_score = (textblob_polarity * 0.4) + (custom_score * 0.6)
    return max(-1.0, min(1.0, hybrid_score))

# --- Load model
@st.cache_resource(show_spinner="🔄 Loading model...")
def load_model():
    with open("xgb_pipeline_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# --- Upload CSV
uploaded_file = st.file_uploader("📁 Upload tweet CSV (raw features: original_text, favorite_count, retweet_count, target_politicians)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    expected_cols = ["original_text", "favorite_count", "retweet_count", "target_politicians"]
    if not all(col in df.columns for col in expected_cols):
        st.error(f"❌ CSV must contain columns: {expected_cols}")
    else:
        st.success("✅ File loaded. Processing...")

        df['clean_text'] = df['original_text'].fillna('').apply(clean_text)
        df['custom_polarity'] = df['clean_text'].apply(calculate_custom_polarity)
        df['textblob_polarity'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
        df['hybrid_polarity'] = 0.4 * df['textblob_polarity'] + 0.6 * df['custom_polarity']

        # Predict
        X = df[['clean_text', 'favorite_count', 'retweet_count', 'hybrid_polarity', 'target_politicians']]
        predictions = model.predict(X)
        df['predicted_weighted_polarity'] = predictions

        st.subheader("📋 Sample Predictions")
        st.dataframe(df[["clean_text", "target_politicians", "predicted_weighted_polarity"]].head(10))

        st.subheader("📊 Aggregated Impact per Politician")
        summary = df.groupby('target_politicians').agg(
            total_predicted_impact=('predicted_weighted_polarity', 'sum'),
            avg_predicted_impact=('predicted_weighted_polarity', 'mean'),
            tweet_count=('predicted_weighted_polarity', 'count')
        ).sort_values('total_predicted_impact', ascending=False)

        st.dataframe(summary)

        st.subheader("📈 Total Predicted Impact by Politician")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            x=summary.index,
            y=summary['total_predicted_impact'],
            palette='viridis', ax=ax
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylabel('Total Predicted Impact')
        ax.set_title('Predicted Social Media Influence')
        st.pyplot(fig)
else:
    st.info("👈 Please upload a CSV file to begin.")
