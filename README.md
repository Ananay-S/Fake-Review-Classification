## 🧾 Fake Product Review Detection using Machine Learning

Detecting **Computer-Generated (CG)** fake product reviews versus **Original (OR)** human-written ones using machine learning models trained on 40,000 labeled reviews.

---

### 🎯 Objective

The goal of this project is to build a robust machine learning pipeline that can:

* Accurately distinguish fake (CG) reviews from real (OR) reviews
* Provide real-time predictions through a Streamlit web app
* Offer interpretable insights into what makes a review appear "fake"

---

### ⚙️ Methodology

#### 🔄 Preprocessing

* Lowercasing, 
* punctuation & digit removal
* Stopword removal
* **TF-IDF Vectorization** 

#### 🤖 Models Trained

* Logistic Regression
* Linear SVM
* Multinomial Naive Bayes
* Random Forest Classifier

#### 🧪 Evaluation

* 5-fold cross-validation
* Hyperparameter tuning via `GridSearchCV`
* Final testing on a held-out test set of 8,087 reviews

---

### 📊 Model Performance (on Test Set)

| Model               | Accuracy | Precision | Recall | F1-Score | Best Hyperparameters                     |
| ------------------- | -------- | --------- | ------ | -------- | ---------------------------------------- |
| Logistic Regression | 0.88     | 0.88      | 0.88   | 0.88     | `solver=liblinear`, `C=10`, `penalty=l2` |
| Linear SVM          | 0.88     | 0.88      | 0.88   | 0.88     | `loss=hinge`, `C=1`, `max_iter=1000`     |
| Naive Bayes         | 0.85     | 0.85      | 0.85   | 0.85     | default                                  |
| Random Forest       | 0.86     | 0.86      | 0.86   | 0.86     | (grid search results not shown)          |

> ✅ **Top Performers**: Logistic Regression and Linear SVM (F1 = 0.88)

---

### 🧠 Text Feature Insights

#### 🔴 Top Fake Review Indicators:

`developed`, `wide`, `couple`, `materials`, `onei`, `downside`, `admit`, `problem`, `reason`, `iti`

> These words are often vague, technical, or oddly placed — suggesting computer generation.

#### 🟢 Top Real Review Indicators:

`though`, `even`, `without`, `instead`, `connects`, `ask`, `coming`, `bc`, `fitting`, `due`

> These are more conversational and natural, typical of genuine user experiences.

---

### 🖥️ Streamlit Web App

An interactive web application is included to:

* 🧾 **Predict** whether a new review is Fake or Real
* 📊 **Visualize model performance** (confusion matrix, classification report, comparison bar and pie chart)
* 🧠 **Explore linguistic patterns** with top indicative words and sample reviews

#### 🛠️ App Structure

```
📂 project/
├── app/
│   └── streamlit_app.py                # Streamlit app
├── data/
│   ├── fake_reviews.csv                # Original data
│   ├── cleaned_reviews.csv             # Processed data
│   └── stopwords_en.json               # stop words
├── model/
│   ├── X_train.pkl                     # Cached datasets
│   ├── X_test.pkl                      # Cached datasets
│   ├── y_train.pkl                     # Cached datasets
│   ├── y_test.pkl                      # Cached datasets
│   ├── tfidf.pkl                       # TF-IDF vectorizer
│   └── model.pkl                       # Trained model
├── notebooks/
│   ├── fake_reviews.ipynb
│   └── logistic_regression.ipynb
├── utils/
│   ├── __init__.py
│   └── clean_text.py                   # Text preprocessing functions
├── requirements.txt
├── README.md
```

#### 🔗 How to Run

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

---

### 🧪 Training Pipeline

1. Preprocess & vectorize text
2. Train 4 ML models
3. Evaluate using precision, recall, F1-score
4. Fine-tune with GridSearchCV
5. Save best model and vectorizer with `joblib`
6. Deploy via Streamlit app

---

### 📦 Requirements

* Python 3.8+
- ipykernel
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- nltk
- joblib
- streamlit

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

### 📌 Conclusion

* Both **Logistic Regression** and **Linear SVM** achieved the best F1-score of **0.88**
* The models effectively learn subtle linguistic patterns between CG and OR reviews
* The web app enables real-time predictions and explainability for end users

---

### 📁 Repository Overview

```
📂 project-root/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── fake_reviews.csv
│   ├── cleaned_reviews.csv
│   └── stopwords_en.json
├── model/
│   ├── X_train.pkl
│   ├── X_test.pkl
│   ├── y_train.pkl
│   ├── y_test.pkl
│   ├── tfidf.pkl
│   └── model.pkl
├── notebooks/
│   ├── fake_reviews.ipynb
│   └── logistic_regression.ipynb
├── utils/
│   ├── __init__.py
│   └── clean_text.py
├── requirements.txt
├── README.md
```

---

