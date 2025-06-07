# preprocess.py
import pandas as pd
import re
import nltk
import joblib
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logging.info("🚀 Starting preprocessing...")

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load and sample dataset
try:
    df = pd.read_csv("spotify_millsongdata.csv").sample(10000)
    logging.info("✅ Dataset loaded and sampled: %d rows", len(df))
except Exception as e:
    logging.error("❌ Failed to load dataset: %s", str(e))
    raise e

# Drop link column and preprocess
df = df.drop(columns=['link'], errors='ignore').reset_index(drop=True)

# Define stopwords
stop_words = set(stopwords.words('english'))

# Clean each song lyric
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Apply cleaning
logging.info("🧹 Cleaning text...")
df['cleaned_text'] = df['text'].apply(preprocess_text)
logging.info("✅ Text cleaned.")

# TF-IDF vectorization
logging.info("🔠 Vectorizing using TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
logging.info("✅ TF-IDF matrix shape: %s", tfidf_matrix.shape)

# Cosine similarity matrix
logging.info("📐 Calculating cosine similarity...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
logging.info("✅ Cosine similarity matrix generated.")

# Save the results
joblib.dump(df, 'data/df_cleaned.pkl')
joblib.dump(tfidf_matrix, 'data/tfidf_matrix.pkl')
joblib.dump(cosine_sim, 'data/cosine_sim.pkl')
logging.info("💾 All files saved to 'data/' folder.")

logging.info("✅ Preprocessing complete.")
py