# ========================
# 📦 Imports
# ========================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import streamlit as st

from sklearn.metrics import classification_report, confusion_matrix

# ========================
# 🔄 Load Model & Vectorizer (Cache for speed)
# ========================

df = pd.read_csv("./data/cleaned_reviews.csv")

from utils.clean_text import clean_text

@st.cache_resource
def load_model_data():
    X_train = joblib.load("./model/X_train.pkl")
    X_test = joblib.load("./model/X_test.pkl")
    y_train = joblib.load("./model/y_train.pkl")
    y_test = joblib.load("./model/y_test.pkl")
    vectorizer = joblib.load("./model/tfidf.pkl")
    model = joblib.load("./model/model.pkl")
    return X_train, X_test, y_train, y_test, vectorizer, model

X_train, X_test, y_train, y_test, vectorizer, model = load_model_data()

feature_names = vectorizer.get_feature_names_out()

# ========================
# ⚙️ Page Controls
# ========================

st.sidebar.header("⚙️ Page Controls")

page_mode = st.sidebar.radio(
    "Choose Section",
    ["Predict", "Evaluation Dashboard", "Text Insights"]
)

# ========================
# 🧾 A. Input + Prediction
# ========================
if page_mode == "Predict":
    
    st.title("🧾 Fake Review Detector")
    review_text = st.text_area("Enter a product review")

    if st.button("Predict"):
        if not review_text.strip():
            st.warning("Please enter a review.")
        else:
            # Preprocess and transform input
            cleaned = clean_text(review_text)
            vector = vectorizer.transform([cleaned])
            # Predict
            pred = model.predict(vector)[0]
            # Display result
            label = "🟢 Real Review (OR)" if pred == 0 else "🔴 Fake Review (CG)"
            st.success(f"Prediction: {label}")

# ========================
# 📊 B. Evaluation Dashboard
# ========================
if page_mode == "Evaluation Dashboard":
    
    st.header("📊 Model Evaluation")
    
    # 1. model evaluation
    st.subheader("1) Model Evaluation")

    # Confusion Matrix
    cm = confusion_matrix(y_test, model.predict(X_test))

    # Show metrics
    report = classification_report(y_test, model.predict(X_test), output_dict=True)
    st.write(pd.DataFrame(report).transpose())

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)


    # 2. model comparison
    st.subheader("2) Model Comparison")

    model_scores = {
        "Logistic Regression": 0.8726,
        "Linear SVC": 0.8760,
        "Naive Bayes": 0.8583,
        "Random Forest": 0.8466,
    }

    score_df = pd.DataFrame(model_scores.items(), columns=['Model', 'F1 Score'])

    fig2 = plt.figure(figsize=(6, 3))
    sns.barplot(data=score_df, x='Model', y='F1 Score')
    plt.xticks(rotation=45)
    st.pyplot(fig2)


    # 3. prediction distribution
    st.subheader("3) Prediction Distribution")

    preds = model.predict(X_test)
    pred_df = pd.Series(preds).map({0: "Real", 1: "Fake"}).value_counts()

    fig3, ax3 = plt.subplots()
    ax3.pie(pred_df, labels=pred_df.index, autopct='%1.1f%%', startangle=90)
    ax3.axis('equal')
    st.pyplot(fig3)

# ========================
# 🧠 C. Text Insights
# ========================
if page_mode == "Text Insights":
    
    st.header("🧠 Text Insights")

    # 1. Top 10 Indicative Words
    st.subheader("1) Top Words Indicative of Fake Reviews")

    # Assuming you're using LogisticRegression or LinearSVC
    coefs = model.coef_[0]
    top_fake = np.argsort(coefs)[-10:]
    top_real = np.argsort(coefs)[:10]

    top_words_fake = [(feature_names[i], coefs[i]) for i in top_fake]
    top_words_real = [(feature_names[i], coefs[i]) for i in top_real]

    st.write("🔴 Fake Review Indicators:")
    st.write(pd.DataFrame(top_words_fake, columns=["Word", "Weight"]).sort_values(by="Weight", ascending=False))

    st.write("🟢 Real Review Indicators:")
    st.write(pd.DataFrame(top_words_real, columns=["Word", "Weight"]).sort_values(by="Weight"))


    # 2. Example Reviews
    st.subheader("2) 🔍 Example Reviews")

    fake_examples = df[df['label'] == 'CG'].sample(2)
    real_examples = df[df['label'] == 'OR'].sample(2)

    st.markdown("**🟢 Real Reviews:**")
    for text in real_examples['text_']:
        st.success(text)

    st.markdown("**🔴 Fake Reviews:**")
    for text in fake_examples['text_']:
        st.error(text)