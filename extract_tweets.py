# -*- coding: utf-8 -*-
import requests
import pandas as pd
from dotenv import load_dotenv
import os
import time

# Load environment variables from .env file
load_dotenv()

# --- 1. SETUP: TWITTERAPI.IO & CONSTANTS ---

# Load your API Key from the .env file
API_KEY = os.getenv("TWITTER_API_IO_KEY")
if not API_KEY:
    raise ValueError("API Key not found. Please set TWITTER_API_IO_KEY in your .env file.")

# The API endpoint for advanced search
API_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"

# Define the columns for the CSV file
COLS = ['id', 'created_at', 'target_politicians', 'source', 'original_text',
        'clean_text', 'sentiment', 'polarity', 'subjectivity', 'lang',
        'favorite_count', 'retweet_count', 'original_author', 'possibly_sensitive',
        'hashtags', 'user_mentions', 'place', 'place_coord_boundaries']

# The search query, using syntax compatible with the API
SEARCH_QUERY = '("Anura Kumara" OR AKD OR "Sajith" OR "Ranil" OR "Namal Rajapaksa") (lang:en OR lang:si) -filter:retweets'

# Define the output filename
OUTPUT_FILENAME = 'tweets_download_incremental.csv'

# --- NEW: ADD A MANUAL STOP CONDITION ---
# Set a limit on the number of pages to fetch to ensure the script stops.
# You can increase this value if you need more data.
MAX_PAGES = 1500

# --- 2. MAIN DATA EXTRACTION SCRIPT (WITH INCREMENTAL SAVING) ---

def main():
    """Main function to download tweets and save them incrementally to a CSV."""
    print(f"Starting tweet download for query: {SEARCH_QUERY}")
    print(f"Data will be saved incrementally to {OUTPUT_FILENAME}")
    print(f"The script will stop after fetching a maximum of {MAX_PAGES} pages.")
    
    headers = {'X-API-Key': API_KEY}
    cursor = ""
    page_count = 0
    total_tweets_saved = 0

    while page_count < MAX_PAGES:
        page_count += 1
        print("-" * 30)
        print(f"Fetching page {page_count}...")
        
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
                print("No more tweets found on this page. Stopping.")
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
            
            # --- INCREMENTAL SAVE LOGIC ---
            if page_data:
                df_page = pd.DataFrame(page_data, columns=COLS)
                # If it's the first page, write the file with a header.
                # Otherwise, append to the existing file without the header.
                write_header = not os.path.exists(OUTPUT_FILENAME)
                if page_count == 1 and not write_header: # Overwrite old file on new run
                     df_page.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8', mode='w', header=True)
                else:
                    df_page.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8', mode='a', header=write_header)

                num_saved = len(df_page)
                total_tweets_saved += num_saved
                print(f"Success! Saved {num_saved} new tweets.")
                print(f"Total tweets saved so far: {total_tweets_saved}")
            
            # Check for pagination and update cursor
            if data.get('has_next_page'):
                cursor = data.get('next_cursor')
                time.sleep(1) # Be respectful to the API
            else:
                print("API indicates no more pages available. Stopping.")
                break
                
        except requests.exceptions.RequestException as e:
            print(f"An error occurred with the API request: {e}")
            break
    
    print("-" * 30)
    if page_count >= MAX_PAGES:
        print(f"Reached the maximum limit of {MAX_PAGES} pages.")
    print("Download process finished.")
    print(f"All data is saved in {OUTPUT_FILENAME}")
    print(f"Total tweets collected in this run: {total_tweets_saved}")

if __name__ == '__main__':
    main()