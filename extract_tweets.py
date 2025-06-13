# -*- coding: utf-8 -*-
import tweepy
import pandas as pd
import re
from textblob import TextBlob
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# --- 1. SETUP: TWITTER API, KEYWORDS, AND SENTIMENT DICTIONARIES ---

# Load Bearer Token from your .env file
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

# Define the columns for the CSV file
COLS = ['id', 'created_at', 'target_politicians', 'source', 'original_text', 
        'clean_text', 'sentiment', 'polarity', 'subjectivity', 'lang', 
        'favorite_count', 'retweet_count', 'original_author', 'possibly_sensitive', 
        'hashtags', 'user_mentions', 'place', 'place_coord_boundaries']

# Keywords for each politician for precise tagging
politician_keywords = {
    'Anura Kumara Dissanayake': ['anura kumara', 'akd', '@anuradissanayake', 'anuradissanayake'],
    'Sajith Premadasa': ['sajith premadasa', 'sajith', '@sajithpremadasa'],
    'Ranil Wickremesinghe': ['ranil wickremesinghe', 'ranil', '@RW_UNP'],
    'Namal Rajapaksa': ['namal rajapaksa', 'namal', '@RajapaksaNamal']
}

# The broad search query to capture all relevant tweets
SEARCH_QUERY = '("Anura Kumara" OR AKD OR "Sajith" OR "Ranil" OR "Namal Rajapaksa") lang:en OR lang:si -is:retweet'

# Happy Emoticons
emoticons_happy = set([':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}', ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D', '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P', 'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)', '<3'])
# Sad Emoticons
emoticons_sad = set([':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<', ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c', ':c', ':{', '>:\\', ';('])

# --- NEW: Singlish Pragmatics Dictionaries ---
singlish_happy = set(['hondai', 'honday', 'hondaiy', 'hondaii', 'niyamai', 'niyamay', 'supiri', 'supiriyak', 'supiriii', 'patta', 'maru', 'shok', 'shoi', 'ela', 'elakiri', 'elaa', 'jayawewa', 'jaya wewa', 'lassanai', 'lassanay', 'gammak', 'gammac', 'sira', 'siraa', 'ow', 'owu', 'ov', 'hari', 'aththa', 'aththac', 'subapathum', 'suba pathum', 'pissu kora', 'thanks', 'thankz', 'thnx', 'tnx'])

singlish_sad = set(['narakai', 'narakay', 'boru', 'boruwak', 'boruu', 'weradi', 'waradi', 'varadi', 'weradii', 'chater', 'chaater', 'chaa', 'epaa', 'epa', 'hora', 'horu', 'horakam', 'pissu', 'pisso', 'gon', 'gonn', 'gon haraka', 'pal horu', 'kalakanni', 'pala', 'palayan', 'aiyo', 'aiyoo', 'ane', 'apoi', 'ammapa', 'na', 'naa', 'ne', 'naha', 'nathuwa', 'nathi', 'neti', 'nathe'])

# --- 2. HELPER FUNCTIONS (UPDATED FOR HYBRID ANALYSIS) ---

def get_target_politicians(text, keywords_dict):
    """Identifies which politician(s) are mentioned in a tweet."""
    mentioned = []
    text_lower = text.lower()
    for politician, keywords in keywords_dict.items():
        if any(keyword in text_lower for keyword in keywords):
            mentioned.append(politician)
    return mentioned

def clean_text_for_blob(tweet_text):
    """
    Performs minimal cleaning on text before sending to TextBlob.
    Removes URLs, mentions, and hashtag symbols but keeps the text intact.
    """
    tweet_text = re.sub(r'https?:\/\/\S+', '', tweet_text)
    tweet_text = re.sub(r'@[A-Za-z0-9_]+', '', tweet_text)
    tweet_text = re.sub(r'#', '', tweet_text) 
    return tweet_text

def calculate_custom_polarity(tweet_text):
    """
    Calculates a custom polarity score by treating Singlish words and emoticons
    as polarity modifiers, adding/subtracting 0.1 for each item found.
        It starts with a polarity of 0.0
        Each happy item adds +0.1
        Each sad item subtracts -0.1
    """
    polarity = 0.0
    text_lower = tweet_text.lower()
    
    # Check for Singlish words
    for word in text_lower.split():
        if word in singlish_happy:
            polarity += 0.1
        elif word in singlish_sad:
            polarity -= 0.1
            
    # Check for emoticons
    for emoticon in emoticons_happy:
        if emoticon in text_lower:
            polarity += 0.1
            
    for emoticon in emoticons_sad:
        if emoticon in text_lower:
            polarity -= 0.1
            
    # Clamp the custom polarity itself to be within [-1, 1] as a safeguard
    return max(min(polarity, 1.0), -1.0)

def get_hybrid_sentiment(text_for_blob, custom_polarity):
    """
    Combines TextBlob's polarity with our custom polarity to get a hybrid result.
    The final score is clamped to stay within the [-1.0, 1.0] range.
    """
    analysis = TextBlob(text_for_blob)
    textblob_polarity = analysis.sentiment.polarity
    
    # Combine the polarities by simple addition
    hybrid_polarity = textblob_polarity + custom_polarity
    
    # --- CRITICAL STEP: Clamp the final score to the valid range [-1.0, 1.0] ---
    hybrid_polarity = max(min(hybrid_polarity, 1.0), -1.0)
    
    # Classify sentiment based on the more accurate hybrid score
    if hybrid_polarity > 0.05:
        sentiment = 'Positive'
    elif hybrid_polarity < -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
        
    return sentiment, hybrid_polarity, analysis.sentiment.subjectivity

def translate_text(text, target_lang='en'):
    """Translates text to English, returns original text on failure."""
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text) or text
    except Exception as e:
        print(f"Translation failed for text: '{text[:30]}...'. Error: {e}")
        return text

# --- 3. MAIN DATA EXTRACTION SCRIPT ---

def main():
    """Main function to extract, process with hybrid sentiment, and save tweets."""
    client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)
    tweet_data = []
    print("Starting HYBRID sentiment extraction with POLARITY MODIFIERS...")
    
    # Use a high limit to get enough data for analysis. Adjust as needed.
    for i, tweet in enumerate(tweepy.Paginator(client.search_recent_tweets, 
                                     query=SEARCH_QUERY,
                                     tweet_fields=["id", "text", "created_at", "source", "lang", "public_metrics", "possibly_sensitive", "author_id", "entities", "geo"],
                                     user_fields=["username"],
                                     expansions=["author_id", "geo.place_id"],
                                     max_results=100).flatten(limit=1500)):
        
        if i % 100 == 0 and i > 0:
            print(f"{i} tweets processed...")
            
        original_text = tweet.text
        
        # Identify the target politician(s) for attribution
        target_politicians = get_target_politicians(original_text, politician_keywords)
        if not target_politicians:
            continue
        
        # --- HYBRID SENTIMENT LOGIC ---
        # 1. Calculate our custom polarity modifier from the raw text
        custom_polarity = calculate_custom_polarity(original_text)
        
        # 2. Prepare text for TextBlob (minimal cleaning)
        cleaned_for_blob = clean_text_for_blob(original_text)
        
        # 3. Translate if needed for TextBlob analysis
        text_for_analysis = cleaned_for_blob
        if tweet.lang == 'si':
            text_for_analysis = translate_text(cleaned_for_blob)

        # 4. Get the final hybrid sentiment and clamped polarity
        sentiment, final_polarity, subjectivity = get_hybrid_sentiment(text_for_analysis, custom_polarity)

        # Extract other useful data from the tweet object
        hashtags = [tag['tag'] for tag in tweet.entities.get('hashtags', [])] if tweet.entities else []
        user_mentions = [mention['username'] for mention in tweet.entities.get('mentions', [])] if tweet.entities else []
        place_info = tweet.geo.get('full_name', None) if tweet.geo else None
        coords = tweet.geo.get('bbox', None) if tweet.geo else None
        
        # Append all data to our list
        tweet_data.append({
            'id': tweet.id, 'created_at': tweet.created_at, 
            'target_politicians': target_politicians,
            'source': tweet.source, 
            'original_text': original_text, 
            'clean_text': cleaned_for_blob,
            'sentiment': sentiment, 
            'polarity': final_polarity, 
            'subjectivity': subjectivity,
            'lang': tweet.lang, 
            'favorite_count': tweet.public_metrics['like_count'],
            'retweet_count': tweet.public_metrics['retweet_count'], 
            'original_author': tweet.author_id,
            'possibly_sensitive': tweet.possibly_sensitive, 
            'hashtags': hashtags,
            'user_mentions': user_mentions, 
            'place': place_info, 
            'place_coord_boundaries': coords
        })

    df = pd.DataFrame(tweet_data, columns=COLS)
    output_filename = 'sri_lanka_election_tweets_final.csv'
    df.to_csv(output_filename, index=False, encoding='utf-8')

    print("-" * 30)
    print(f"Extraction complete! Data saved to {output_filename}")
    print(f"Total targeted tweets collected: {len(df)}")
    print("Example rows showing final clamped polarity:")
    print(df[['target_politicians', 'sentiment', 'polarity', 'original_text']].head(10))

if __name__ == '__main__':
    main()