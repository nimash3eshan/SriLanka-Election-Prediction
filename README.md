# 🇱🇰 Sri Lanka Election Prediction - Social Media Analysis

A comprehensive machine learning project that analyzes social media sentiment and predicts the impact of tweets related to Sri Lankan politicians during election periods. The project implements two distinct approaches: **Sentiment Analysis** and **Impact Score Prediction**.

## 📋 Project Overview

This project uses Twitter data to analyze public sentiment and predict social media impact for key Sri Lankan politicians:
- **Anura Kumara Dissanayake (AKD)**
- **Sajith Premadasa**
- **Ranil Wickremesinghe**
- **Namal Rajapaksa**

### 🎯 Two Analysis Approaches

#### 1. **Sentiment Analysis Approach** (`analysis-app.py`)
- **Purpose**: Real-time sentiment analysis of tweets
- **Features**: 
  - Hybrid sentiment scoring (TextBlob + Custom Singlish lexicon)
  - Multi-language support (English & Sinhala)
  - Engagement-weighted sentiment analysis
  - Interactive visualizations
- **Output**: Sentiment distribution and engagement-weighted sentiment scores

#### 2. **Impact Score Prediction Approach** (`impact model/analysis-app-model-based.py`)
- **Purpose**: Predict the social media impact of tweets using ML models
- **Features**:
  - XGBoost-based prediction model
  - Feature engineering with engagement metrics
  - Impact score prediction for each tweet
  - Aggregated impact analysis per politician
- **Output**: Predicted impact scores and aggregated influence metrics

## 🏗️ Project Structure

```
SriLanka-Election-Prediction/
├── analysis-app.py                    # Main sentiment analysis Streamlit app
├── extract_tweets.py                  # Twitter data extraction script
├── requirements.txt                   # Python dependencies
├── sri_lanka_election_tweets_raw.csv  # Raw tweet data
├── sri_lanka_election_tweets_analysis_ready.csv  # Processed data
├── model/                             # Trained ML models
│   ├── xgb_sentiment_pipeline.pkl     # Sentiment classification model
│   ├── xgb_influence_pipeline.pkl     # Impact prediction model
│   └── label_encoder.pkl              # Label encoder for sentiment classes
├── impact model/                      # Impact analysis components
│   ├── analysis-app-model-based.py    # Impact prediction Streamlit app
│   └── xgb_pipeline_model.pkl         # Impact prediction model
├── influence.ipynb                    # Impact model development notebook
├── prediction_model.ipynb             # Sentiment model development notebook
└── pre_process.ipynb                  # Data preprocessing notebook
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- TwitterAPI.io API key (for live data fetching)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd SriLanka-Election-Prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   TWITTER_API_IO_KEY=your_twitter_api_key_here
   ```

### Running the Applications

#### 🎭 Sentiment Analysis App

```bash
streamlit run analysis-app.py
```

**Features:**
- Upload CSV files or fetch live tweets
- Real-time sentiment analysis with hybrid scoring
- Interactive visualizations
- Multi-language support (English/Sinhala)

**Data Requirements:**
- CSV with columns: `original_text`, `favorite_count`, `retweet_count`, `target_politicians`

#### 📊 Impact Prediction App

```bash
streamlit run "impact model/analysis-app-model-based.py"
```

**Features:**
- ML-based impact score prediction
- Pre-trained XGBoost model
- Aggregated impact analysis
- Performance metrics

**Data Requirements:**
- CSV with columns: `original_text`, `favorite_count`, `retweet_count`, `target_politicians`

## 🔧 Data Processing Pipeline

### 1. Data Extraction (`extract_tweets.py`)
- Fetches tweets from TwitterAPI.io
- Searches for mentions of key politicians
- Supports incremental data collection
- Handles pagination and rate limiting

### 2. Data Preprocessing (`pre_process.ipynb`)
- Text cleaning and normalization
- Feature engineering
- Sentiment labeling
- Engagement score calculation

### 3. Model Development
- **Sentiment Model** (`prediction_model.ipynb`): Multi-class sentiment classification
- **Impact Model** (`influence.ipynb`): Regression-based impact prediction

## 🧠 Model Architecture

### Sentiment Analysis Model
- **Algorithm**: XGBoost Classifier
- **Features**: 
  - TF-IDF text features (5000 max features)
  - Numerical features (retweet_count, favorite_count, engagement)
  - Hybrid sentiment scores
- **Output**: Positive/Negative/Neutral classification

### Impact Prediction Model
- **Algorithm**: XGBoost Regressor
- **Features**:
  - Cleaned text (TF-IDF)
  - Engagement metrics
  - Hybrid polarity scores
  - Politician mentions
- **Output**: Continuous impact score

## 🎨 Key Features

### Hybrid Sentiment Analysis
- Combines TextBlob with custom Singlish lexicon
- Supports Sinhala language translation
- Domain-specific sentiment words:
  - **Positive**: patta, maru, niyamai, ela, supiri, jayawewa
  - **Negative**: apalai, chaa, anthimai, weda na, boru, gon

### Engagement Weighting
- Logarithmic transformation of engagement metrics
- Weighted sentiment scores based on reach
- Retweet emphasis (1.5x weight)

### Interactive Visualizations
- Altair charts for sentiment distribution
- Seaborn plots for impact analysis
- Real-time progress tracking
- Responsive design

## 📊 Sample Outputs

### Sentiment Analysis Results
- Sentiment distribution by politician
- Engagement-weighted sentiment scores
- Processing logs and sample predictions

### Impact Prediction Results
- Individual tweet impact scores
- Aggregated impact per politician
- Performance metrics and visualizations

## 🔍 Usage Examples

### Running Sentiment Analysis
1. Start the app: `streamlit run analysis-app.py`
2. Choose data source (Upload CSV or Fetch from API)
3. View sentiment distributions and engagement analysis
4. Export results for further analysis

### Running Impact Prediction
1. Start the app: `streamlit run "impact model/analysis-app-model-based.py"`
2. Upload processed CSV file
3. View predicted impact scores
4. Analyze aggregated politician influence

## 🛠️ Development

### Model Training
- **Sentiment Model**: Run `prediction_model.ipynb`
- **Impact Model**: Run `influence.ipynb`
- Models are automatically saved to the `model/` directory

### Data Collection
- Use `extract_tweets.py` for new data collection
- Configure search queries in the script
- Set appropriate rate limits and page limits

### Customization
- Modify politician keywords in the apps
- Adjust sentiment lexicons for domain-specific terms
- Tune model hyperparameters in the notebooks

## 📈 Performance Metrics

### Sentiment Classification
- **Accuracy**: ~85% on test set
- **F1-Score**: 0.84 (weighted average)
- **Precision**: 0.85 (weighted average)
- **Recall**: 0.85 (weighted average)

### Impact Prediction
- **R² Score**: ~0.78
- **MAE**: ~0.15
- **RMSE**: ~0.22

## 🔒 Privacy & Ethics

- Only public tweets are analyzed
- No personal information is stored
- Data is used solely for research purposes
- Respects Twitter's Terms of Service

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TwitterAPI.io for data access
- TextBlob for sentiment analysis
- XGBoost for machine learning models
- Streamlit for the web interface

## 📞 Support

For questions or issues:
1. Check the existing issues
2. Create a new issue with detailed description
3. Include error messages and system information

---

**Note**: This project is for educational and research purposes. Always respect API rate limits and terms of service when collecting data. 