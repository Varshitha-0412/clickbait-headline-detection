import joblib
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

def train_and_save_models(X_train_tfidf, y_train, X_test_tfidf, y_test, vectorizer):
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="linear", random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "Naive Bayes": MultinomialNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    reports = {}
    trained_models = {}

    # Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)
        report = classification_report(y_test, preds, output_dict=True)
        reports[name] = report
        trained_models[name] = model
        print(f"\n{name} Classification Report:\n")
        print(classification_report(y_test, preds))

    # Pick best model by F1-Score
    best_model_name = max(reports, key=lambda m: reports[m]["weighted avg"]["f1-score"])
    best_model = trained_models[best_model_name]

    print(f"\nBest Model Selected: {best_model_name}")

   # Save best model and vectorizer with fixed names
joblib.dump(best_model, "best_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("best_model.pkl and tfidf_vectorizer.pkl saved successfully!")
