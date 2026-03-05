import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def clean_text(text):
    """
    Basic text cleaning:
    - Lowercasing
    - Removing special characters, numbers, and extra spaces
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)   # keep only letters and spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_and_preprocess_data(filepath, test_size=0.2, random_state=42):
    """
    Load dataset, clean text, split into train/test, and apply TF-IDF vectorization.
    Returns: X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer
    """
    # Load dataset
    df = pd.read_csv(filepath)

    # Assuming dataset has columns: 'headline' and 'label'
    df["headline"] = df["headline"].apply(clean_text)

    X = df["headline"]
    y = df["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer
