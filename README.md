# 📰 Clickbait Headline Detection

A machine learning project to classify news headlines as **Clickbait** or **Not Clickbait** using multiple models (Logistic Regression, Random Forest, SVM, XGBoost, Naive Bayes, KNN, Gradient Boosting).  
The best-performing model is automatically saved and deployed via a **Streamlit app**.

---

## 🚀 Features
- Preprocessing pipeline (`preprocessing.py`) for cleaning text and applying TF-IDF vectorization.
- Model training and evaluation (`model.py`) across 7 ML algorithms.
- Automatic selection and saving of the best model (`best_model.pkl`) and vectorizer (`tfidf_vectorizer.pkl`).
- Interactive web app (`app.py`) built with Streamlit for real-time headline classification.

---

## 📂 Project Structure
├── preprocessing.py
# Handles text cleaning, splitting, and TF-IDF vectorization 
├── model.py           
# Trains multiple models, evaluates, saves the best one 
├── app.py             
# Streamlit app for headline classification 
├── headlines.csv      
# Dataset (headline + label) 
├── requirements.txt   
# Dependencies 
├── README.md          
# Project documentation

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/clickbait-detector.git
   cd clickbait-detector
2.Install dependencies:

  pip install -r requirements.txt
# Usage
1. Preprocess & Train
Run the training pipeline:
python train.py
This will:
- Load and preprocess the dataset (headlines.csv).
- Train all models.
- Save the best model as best_model.pkl and the TF-IDF vectorizer as tfidf_vectorizer.pkl.
2. Run the App
Start the Streamlit app:
streamlit run app.py
📈 Models Compared
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
- Naive Bayes
- K-Nearest Neighbors (KNN)
- Gradient Boosting
The best model is selected based on F1-Score.

# Example Prediction
Input:
"You won’t believe what happened next!"
Output:
⚠️ This looks like Clickbait!
# Requirements
- Python 3.8+
- scikit-learn
- xgboost
- pandas
- numpy
- seaborn
- matplotlib
- streamlit
- joblib







