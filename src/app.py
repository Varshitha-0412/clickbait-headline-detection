import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load(r"C:\Users\MEGHANA\OneDrive\Desktop\clickbait-detection\notebooks\best_model.pkl")   # replace with your actual saved file
vectorizer = joblib.load(r"C:\Users\MEGHANA\OneDrive\Desktop\clickbait-detection\notebooks\tfidf_vectorizer.pkl")  # replace with your actual saved file

# Streamlit UI
st.title("📰 Clickbait Headline Detector")
st.write("Enter a headline below to check if it's clickbait or not.")

# User input
headline = st.text_input("Headline:")

if st.button("Predict"):
    if headline.strip() != "":
        # Transform input using TF-IDF
        X = vectorizer.transform([headline])
        prediction = model.predict(X)[0]

        # Display result
        if prediction == 1:
            st.error("⚠️ This looks like **Clickbait**!")
        else:
            st.success("✅ This looks like **Not Clickbait**.")
    else:
        st.warning("Please enter a headline before predicting.")
