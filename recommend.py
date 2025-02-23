import re
import ast
import pandas as pd
import nltk
import ace_tools_open as tools
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.read_csv("./data/titles.csv")

# Filter only movies and select relevant columns
movies = data[data["type"] == "MOVIE"][["title", "description", "genres"]].dropna()

# Download stopwords & sentiment model
nltk.download("stopwords")
nltk.download("vader_lexicon")
stop_words = set(stopwords.words("english"))
sia = SentimentIntensityAnalyzer()

# Function to handle negation in text
def handle_negation(text):
    """ Converts 'not action' -> 'not_action' to retain negation meaning """
    words = text.split()
    new_words = []
    negation = False

    for word in words:
        if word in ["not", "no", "don't", "doesn't", "isn't", "wasn't", "aren't", "weren't", "hasn't", "haven't"]:
            negation = True
        elif negation:
            new_words.append("not_" + word)  # Merge negation with next word
            negation = False
        else:
            new_words.append(word)

    return " ".join(new_words)

# Function to clean text with negation handling
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\W", " ", text)  # Remove special characters
    text = handle_negation(text)  # Apply negation handling
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Apply text preprocessing
movies["clean_description"] = movies["description"].apply(clean_text)

# Convert genres from list format safely
movies["genres"] = movies["genres"].apply(lambda x: " ".join(ast.literal_eval(x)) if isinstance(x, str) else "")

# Combine genres and descriptions
movies["combined_text"] = movies["clean_description"] + " " + movies["genres"]

# --- VECTORIZATION METHODS (USING GENRES ONLY) ---

# 1. Binary Feature Matrix (Presence/Absence)
binary_vectorizer = CountVectorizer(binary=True)
binary_matrix = binary_vectorizer.fit_transform(movies["genres"])

# 2. Bag of Words (BoW) (Word Counts)
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(movies["genres"])

# 3. TF-IDF (Term Frequency-Inverse Document Frequency)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(movies["genres"])

# --- SENTENCE TRANSFORMER MODEL (USING COMBINED TEXT) ---
model = SentenceTransformer("all-mpnet-base-v2")
movies["combined_text_embedding"] = movies["combined_text"].apply(lambda x: model.encode(x, convert_to_tensor=True))

# Function to detect sentiment (positive or negative)
def detect_sentiment(user_input):
    """ Returns 'positive' or 'negative' based on sentiment analysis """
    sentiment_score = sia.polarity_scores(user_input)["compound"]
    return "positive" if sentiment_score >= 0.65 else "negative"

# Function to extract negated genres
def extract_negated_genres(user_input):
    """ Identifies words following 'not', 'don't', 'no', etc. """
    words = user_input.lower().split()
    negated_terms = []
    negation = False

    for word in words:
        if word in ["not", "no", "don't", "doesn't", "isn't", "wasn't", "aren't", "weren't", "hasn't", "haven't"]:
            negation = True
        elif negation:
            negated_terms.append(word)  # Capture negated genre
            negation = False

    return negated_terms

# Function to recommend movies using genre-based vectorization
def recommend_movies_genre(user_input, vectorizer_type="tfidf", top_n=5):
    sentiment = detect_sentiment(user_input)  # Detect sentiment

    # Select the appropriate vectorizer and matrix
    if vectorizer_type == "binary":
        vectorizer = binary_vectorizer
        feature_matrix = binary_matrix
    elif vectorizer_type == "bow":
        vectorizer = bow_vectorizer
        feature_matrix = bow_matrix
    elif vectorizer_type == "tfidf":
        vectorizer = tfidf_vectorizer
        feature_matrix = tfidf_matrix
    else:
        raise ValueError("Invalid vectorizer_type. Choose from 'binary', 'bow', or 'tfidf'.")

    # Transform user input using vectorizer
    user_vector = vectorizer.transform([user_input])

    # Compute cosine similarity
    similarity_scores = cosine_similarity(user_vector, feature_matrix).flatten()

    # If sentiment is positive, return **most similar** movies instead
    if sentiment == "positive":
        top_indices = similarity_scores.argsort()[::-1][:top_n]
    else:
        top_indices = similarity_scores.argsort()[:top_n]

    # Create a dataframe with recommendations and similarity scores
    recommendations = movies.iloc[top_indices][["title", "genres", "description"]].copy()
    recommendations["similarity_score"] = similarity_scores[top_indices]

    return recommendations

# Function to recommend movies using SentenceTransformer embeddings
def recommend_movies_embeddings(user_input, top_n=5):
    sentiment = detect_sentiment(user_input)  # Detect sentiment
    negated_genres = extract_negated_genres(user_input)  # Extract negated genres

    # Remove movies that match the negated genres
    filtered_movies = movies[~movies["genres"].apply(lambda x: any(neg in x for neg in negated_genres))]

    # If all movies are removed due to negation, return an empty DataFrame
    if filtered_movies.empty:
        return pd.DataFrame({"title": [], "genres": [], "description": [], "similarity_score": []})

    # Encode the user query into an embedding
    user_embedding = model.encode(user_input, convert_to_tensor=True)

    # Compute similarity scores for remaining movies
    similarity_scores = [util.pytorch_cos_sim(user_embedding, genre_emb).item() for genre_emb in filtered_movies["combined_text_embedding"]]

    # If sentiment is positive, return **most similar** movies
    if sentiment == "positive":
        top_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:top_n]
    else:
        top_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i])[:top_n]

    # Create recommendations dataframe
    recommendations = filtered_movies.iloc[top_indices][["title", "genres", "description"]].copy()
    recommendations["similarity_score"] = [similarity_scores[i] for i in top_indices]

    return recommendations

# --- USER INTERACTION ---
if __name__ == "__main__":
    # Ask user for input query
    user_query = input("\nEnter your movie preference: ")

    # Ask user to choose method
    method_type = input("\nChoose a recommendation method (vectorization or transformer): ").strip().lower()
    while method_type not in ["vectorization", "transformer"]:
        method_type = input("Invalid choice. Choose from 'vectorization' or 'transformer': ").strip().lower()

    # Ask user for vectorizer type (only for genre-based methods)
    vectorizer_type = None
    if method_type == "vectorization":
        vectorizer_type = input("\nChoose a vectorizer type (binary, bow, tfidf): ").strip().lower()
        while vectorizer_type not in ["binary", "bow", "tfidf"]:
            vectorizer_type = input("Invalid choice. Choose from 'binary', 'bow', or 'tfidf': ").strip().lower()

    # Ask user for number of recommendations
    top_n = int(input("\nEnter the number of recommendations you want (e.g., 5): "))

    # Get recommendations
    if method_type == "vectorization":
        recommendations = recommend_movies_genre(user_query, vectorizer_type=vectorizer_type, top_n=top_n)
    else:
        recommendations = recommend_movies_embeddings(user_query, top_n=top_n)

    # Display results using tools
    tools.display_dataframe_to_user(name=f"Movie Recommendations ({method_type.upper()})", dataframe=recommendations)
