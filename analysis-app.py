import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import re
from textblob import TextBlob
from deep_translator import GoogleTranslator
import time
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Twitter API Helper ---
def fetch_tweets_via_api(api_key: str, max_pages: int = 10, output_csv: str = 'tweets_download_incremental.csv'):
    API_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"
    SEARCH_QUERY = '("Anura Kumara" OR AKD OR "Sajith" OR "Ranil" OR "Namal Rajapaksa") (lang:en OR lang:si) -filter:retweets'
    COLS = ['id', 'created_at', 'target_politicians', 'source', 'original_text',
            'clean_text', 'sentiment', 'polarity', 'subjectivity', 'lang',
            'favorite_count', 'retweet_count', 'original_author', 'possibly_sensitive',
            'hashtags', 'user_mentions', 'place', 'place_coord_boundaries']

    headers = {'X-API-Key': api_key}
    cursor = ""
    page_count = 0
    all_pages = []

    progress = st.progress(0, text="Fetching tweets from TwitterAPI.io...")

    while page_count < max_pages:
        page_count += 1
        params = {
            'query': SEARCH_QUERY,
            'queryType': 'Latest',
            'cursor': cursor
        }

        try:
            response = requests.get(API_URL, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            tweets_on_page = data.get('tweets', [])
            if not tweets_on_page:
                break

            page_data = []
            for tweet in tweets_on_page:
                author_info = tweet.get('author', {})
                entities = tweet.get('entities', {})
                page_data.append({
                    'id': tweet.get('id'), 'created_at': tweet.get('createdAt'),
                    'source': tweet.get('source'), 'original_text': tweet.get('text'),
                    'lang': tweet.get('lang'), 'favorite_count': tweet.get('likeCount'),
                    'retweet_count': tweet.get('retweetCount'), 'original_author': author_info.get('id'),
                    'possibly_sensitive': author_info.get('possiblySensitive'),
                    'hashtags': [h.get('text') for h in entities.get('hashtags', [])],
                    'user_mentions': [m.get('screen_name') for m in entities.get('user_mentions', [])],
                    'place': author_info.get('location'), 'place_coord_boundaries': None,
                    'target_politicians': None, 'clean_text': None, 'sentiment': None,
                    'polarity': None, 'subjectivity': None
                })

            if page_data:
                df_page = pd.DataFrame(page_data, columns=COLS)
                all_pages.append(df_page)

            if data.get('has_next_page'):
                cursor = data.get('next_cursor')
                progress.progress(page_count / max_pages, text=f"Page {page_count} fetched...")
                time.sleep(1)
            else:
                break

        except requests.exceptions.RequestException:
            break

    progress.empty()
    if all_pages:
        final_df = pd.concat(all_pages, ignore_index=True)
        final_df.to_csv(output_csv, index=False)
        return final_df
    else:
        return pd.DataFrame(columns=COLS)

# --- Dictionaries ---
politician_keywords = {
    'Anura Kumara Dissanayake': ['anura kumara', 'akd', '@anuradissanayake', 'anuradissanayake'],
    'Sajith Premadasa': ['sajith premadasa', 'sajith', '@sajithpremadasa'],
    'Ranil Wickremesinghe': ['ranil wickremesinghe', 'ranil', '@RW_UNP'],
    'Namal Rajapaksa': ['namal rajapaksa', 'namal', '@RajapaksaNamal']
}

emoticons_happy = {':)', ':D', 'XD', '<3', ':-)', '=)', 'xD'}
emoticons_sad = {':(', ':-(', ":'(", ':-/', '>:('}
singlish_happy = {'hondai', 'niyamai', 'supiri', 'patta', 'maru', 'ela', 'jayawewa', 'sira', 'lassanai', 'ow', 'hari'}
singlish_sad = {'narakai', 'boru', 'weradi', 'epaa', 'hora', 'pissu', 'gon', 'kalakanni', 'aiyo', 'ne', 'nathi'}

# --- Helper Functions ---
def get_target_politicians(text):
    if not isinstance(text, str): return []
    mentioned = []
    text_lower = text.lower()
    for pol, keys in politician_keywords.items():
        if any(k in text_lower for k in keys):
            mentioned.append(pol)
    return mentioned

def clean_text(tweet):
    tweet = re.sub(r'https?:\/\/\S+', '', tweet)
    tweet = re.sub(r'@[A-Za-z0-9_]+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    return tweet

def calculate_custom_polarity(text):
    if not isinstance(text, str): return 0.0
    text_lower = text.lower()
    words = text_lower.split()
    happy_count = sum(1 for word in words if word in singlish_happy)
    sad_count = sum(1 for word in words if word in singlish_sad)
    happy_emo_count = sum(1 for emo in emoticons_happy if emo in text_lower)
    sad_emo_count = sum(1 for emo in emoticons_sad if emo in text_lower)
    polarity = 0.1 * (happy_count + happy_emo_count - sad_count - sad_emo_count)
    return max(min(polarity, 1.0), -1.0)

def get_hybrid_sentiment(text, custom_polarity):
    analysis = TextBlob(text)
    hybrid = analysis.sentiment.polarity + custom_polarity
    hybrid = max(min(hybrid, 1.0), -1.0)
    label = 'Positive' if hybrid > 0.05 else 'Negative' if hybrid < -0.05 else 'Neutral'
    return label, hybrid, analysis.sentiment.subjectivity

def translate_text(text, lang='en'):
    try:
        return GoogleTranslator(source='auto', target=lang).translate(text)
    except:
        return text

# --- Streamlit UI ---
st.set_page_config(page_title="Tweet Analysis For Election Results Prediction Sri Lanka", layout="wide")
st.title("🇱🇰 Tweet Analysis For Election Results Prediction - Sri Lanka")

st.sidebar.header("🔄 Data Source")
source_choice = st.sidebar.radio("Select data input method:", ["Upload CSV", "Fetch from TwitterAPI.io"])

df = None

if source_choice == "Upload CSV":
    file = st.file_uploader("📁 Upload the CSV containing tweet data", type="csv")
    if file:
        df = pd.read_csv(file)
elif source_choice == "Fetch from TwitterAPI.io":
    if st.button("🚀 Fetch Tweets Now"):
        api_key = os.getenv("TWITTER_API_IO_KEY")
        if not api_key:
            st.error("❌ API Key not found. Please set TWITTER_API_IO_KEY in .env")
        else:
            df = fetch_tweets_via_api(api_key)

# --- Proceed with analysis if df is loaded ---
if df is not None and not df.empty:
    df['original_text'] = df['original_text'].fillna('')
    df['target_politicians'] = df['original_text'].apply(get_target_politicians)
    df = df[df['target_politicians'].apply(len) > 0].reset_index(drop=True)

    st.info("⚙️ Calculating hybrid sentiment... Please wait.")

    progress_bar = st.progress(0)
    preview_box = st.empty()
    results = []
    latest_logs = []

    for idx, row in df.iterrows():
        text = row['original_text']
        lang = row.get('lang', 'en')
        cleaned = clean_text(text)
        if lang == 'si':
            cleaned = translate_text(cleaned)
        custom_p = calculate_custom_polarity(text)
        sentiment, polarity, subj = get_hybrid_sentiment(cleaned, custom_p)
        results.append((cleaned, sentiment, polarity, subj))

        log = f"📝: {text[:50]}...\n🧼: {cleaned[:50]}...\n⚖️: {polarity:.2f} → **{sentiment}**"
        latest_logs.append(log)
        if len(latest_logs) > 5:
            latest_logs.pop(0)

        preview_box.markdown("**🔍 Latest Processing (last 5 tweets)**\n" + "\n\n".join(latest_logs))
        progress_bar.progress((idx + 1) / len(df))

    progress_bar.empty()

    df[['clean_text', 'sentiment', 'polarity', 'subjectivity']] = pd.DataFrame(results, index=df.index)
    df = df.explode('target_politicians')
    df['engagement'] = df['retweet_count'] + df['favorite_count'] + 1
    df['weighted_polarity'] = df['polarity'] * df['engagement']

    st.success("✅ Sentiment analysis complete!")

    st.subheader("📊 Sentiment Distribution by Politician")
    sentiment_counts = df.groupby(['target_politicians', 'sentiment']).size().unstack(fill_value=0)
    sentiment_counts = sentiment_counts[['Positive', 'Neutral', 'Negative']]
    melted_sentiments = sentiment_counts.reset_index().melt(
        id_vars='target_politicians',
        var_name='Sentiment',
        value_name='Count')
    chart = alt.Chart(melted_sentiments).mark_bar().encode(
        x=alt.X('target_politicians:N', title='Politician'),
        y=alt.Y('Count:Q'),
        color=alt.Color('Sentiment:N', scale=alt.Scale(domain=['Positive', 'Neutral', 'Negative'], range=['#4caf50', '#ff9800', '#f44336'])),
        tooltip=['target_politicians', 'Sentiment', 'Count']
    ).properties(width=1000, height=400)
    st.altair_chart(chart, use_container_width=False)

    st.subheader("🔥 Engagement-Weighted Sentiment")
    weighted = df.groupby('target_politicians')['weighted_polarity'].sum().sort_values(ascending=False)
    color_map = ['#2ca02c' if val > 0 else '#d62728' for val in weighted]
    fig, ax = plt.subplots(figsize=(6, 3), facecolor='none')
    sns.barplot(x=weighted.index, y=weighted.values, palette=color_map, ax=ax)
    ax.tick_params(colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')
    plt.xticks(rotation=45, ha='right')
    plt.title("Total Weighted Sentiment by Politician")
    ax.set_facecolor('none')
    fig.patch.set_alpha(0)
    st.pyplot(fig, transparent=True)

    st.subheader("🔍 View Processed Data")
    st.dataframe(df)

elif df is not None and df.empty:
    st.warning("⚠️ No valid tweets found.")
