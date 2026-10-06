# 🧠 Sentiment Analysis of Tweets

> An end-to-end Natural Language Processing (NLP) web application that analyzes text and classifies it as **Positive, Negative, or Neutral** using machine learning.

[![Live Demo](https://img.shields.io/badge/Live-Demo-success)](https://sentiment-analysis-tweets-b3ht.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black)](https://flask.palletsprojects.com/)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-ML-orange)](https://scikit-learn.org/)

### 🚀 Live Demo

**Try the application:**
https://sentiment-analysis-tweets-b3ht.onrender.com

---

## 📌 Project Overview

Social media generates a massive amount of unstructured text every day. Understanding whether users express positive, negative, or neutral opinions can help organizations monitor customer feedback, identify problems, and understand public sentiment.

This project implements an end-to-end **NLP sentiment classification system** that accepts user-generated text through a web interface and predicts its sentiment using a trained machine learning model.

The application combines **word-level and character-level TF-IDF features** with **Logistic Regression** to handle normal text as well as variations such as informal language, repeated characters, slang, and spelling variations.

The system also includes a rule-based layer for handling important linguistic patterns such as **negation and "but" constructions**, improving the practical behavior of the classifier on difficult sentences.

---

## ✨ Key Features

* 🧠 **3-Class Sentiment Classification**

  * Positive
  * Negative
  * Neutral

* 🔤 **NLP Text Processing**

  * Text cleaning
  * Normalization
  * Word-level feature extraction
  * Character-level feature extraction

* 📊 **Hybrid TF-IDF Representation**

  * Word n-grams
  * Character n-grams
  * Combined sparse feature representation

* 🤖 **Machine Learning Classification**

  * Logistic Regression
  * Stratified training and testing
  * Class-balanced training

* 🎯 **Confidence Score**

  * Displays the model's prediction confidence

* 🧩 **Practical Language Handling**

  * Negation
  * Slang
  * Profanity
  * Repeated characters
  * Informal expressions
  * "But" sentence structures

* 🌐 **Web Application**

  * Built with Flask
  * Interactive sentiment analyzer
  * Admin dashboard
  * SQLite database for storing analyzed text

* 🚀 **Cloud Deployment**

  * Deployed as a live web application using Render

---

## 🏗️ System Architecture

```text
                    USER TEXT
                       │
                       ▼
              ┌─────────────────┐
              │   Flask Web UI  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Text Cleaning  │
              │ & Normalization │
              └────────┬────────┘
                       │
                       ▼
          ┌──────────────────────────┐
          │    Feature Extraction   │
          │                          │
          │  Word TF-IDF + Char TF-IDF
          └────────────┬─────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Logistic        │
              │ Regression      │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Rule-Based      │
              │ Language Layer  │
              └────────┬────────┘
                       │
                       ▼
             ┌────────────────────┐
             │ Sentiment +        │
             │ Confidence Score   │
             └────────────────────┘
```

---

## 🧠 Machine Learning Approach

### 1. Text Preprocessing

Input text is cleaned and normalized before being passed to the model.

The preprocessing pipeline is designed to preserve useful sentiment information while reducing unnecessary textual variation.

### 2. Word-Level TF-IDF

Word-level TF-IDF features capture important words and word combinations associated with different sentiments.

For example:

```text
"excellent service"
"very disappointed"
"really good"
"terrible experience"
```

### 3. Character-Level TF-IDF

Character n-grams provide additional robustness against informal text and spelling variations.

Examples:

```text
good
goooood
amazing
amaaazing
worst
worsttttt
```

Character features can capture similarities even when the exact word is not present in the training vocabulary.

### 4. Feature Combination

Word and character TF-IDF representations are combined into a single sparse feature matrix.

```text
Word TF-IDF
     +
Character TF-IDF
     ↓
Combined Feature Matrix
     ↓
Logistic Regression
```

### 5. Sentiment Prediction

The trained classifier predicts one of three classes:

```text
Positive
Negative
Neutral
```

The application also calculates a confidence score from the model's predicted probabilities.

---

## 🧩 Handling Difficult Language

A practical sentiment system cannot rely only on individual keywords.

The application includes additional handling for linguistic patterns such as:

### Negation

```text
"I don't hate it"
"I don't like this"
"This is not good"
"This is not bad"
```

### Contrast / "But" Statements

The sentiment after a contrast can be more important than the first part of the sentence.

Examples:

```text
"I love the design but the app keeps crashing"
→ Negative

"The service is terrible but I really like the new update"
→ Positive
```

### Informal Text

```text
"goooood app!!!"
"amaaazing experience"
"worsttttt app ever"
"bruh this is actually good"
"wtf is this garbage"
```

This combination of machine learning and practical text rules makes the application more robust for real-world user-generated text.

---

## 📊 Dataset

The model was trained using a large Twitter sentiment dataset and additional custom examples focused on difficult sentiment patterns.

The final training dataset contains approximately **30,000 balanced examples**, with approximately 10,000 examples for each sentiment class.

| Sentiment | Approx. Samples |
| --------- | --------------: |
| Positive  |         10,000+ |
| Negative  |         10,000+ |
| Neutral   |         10,000+ |

The dataset was cleaned, deduplicated, balanced, and supplemented with custom examples.

---

## 🛠️ Technology Stack

| Technology          | Purpose                  |
| ------------------- | ------------------------ |
| Python              | Core development         |
| Flask               | Web application          |
| Scikit-learn        | Machine learning         |
| Pandas              | Data processing          |
| NumPy               | Numerical operations     |
| SciPy               | Sparse feature matrices  |
| TF-IDF              | Text feature extraction  |
| Logistic Regression | Sentiment classification |
| SQLite              | Local data storage       |
| HTML/CSS/JavaScript | Frontend                 |
| Render              | Cloud deployment         |
| Git/GitHub          | Version control          |

---

## 📁 Project Structure

```text
sentiment-analysis-tweets/
│
├── templates/
│   ├── index.html
│   └── admin.html
│
├── app.py
├── train.py
├── model.pkl
├── word_vectorizer.pkl
├── char_vectorizer.pkl
├── requirements.txt
├── data.csv
├── .gitignore
└── README.md
```

### Important Files

**`app.py`**
Flask application, prediction pipeline, database operations, and web routes.

**`train.py`**
Training pipeline for the sentiment classification model.

**`model.pkl`**
Serialized trained Logistic Regression model.

**`word_vectorizer.pkl`**
Trained word-level TF-IDF vectorizer.

**`char_vectorizer.pkl`**
Trained character-level TF-IDF vectorizer.

**`templates/index.html`**
Main sentiment analysis interface.

**`templates/admin.html`**
Admin dashboard for reviewing analyzed text and predictions.

---

## ⚙️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/Lucifer666-ms/sentiment-analysis-tweets.git
cd sentiment-analysis-tweets
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## 🔬 Example Predictions

| Input                                                      | Prediction |
| ---------------------------------------------------------- | ---------- |
| `I absolutely love this app!`                              | Positive   |
| `This app is amazing and works perfectly`                  | Positive   |
| `I hate this stupid app`                                   | Negative   |
| `Worst experience ever`                                    | Negative   |
| `The app is okay`                                          | Neutral    |
| `Nothing special about it`                                 | Neutral    |
| `I love the design but the app keeps crashing`             | Negative   |
| `The service is terrible but I really like the new update` | Positive   |

---

## 📈 Model Evaluation

The machine learning pipeline uses a stratified train/test split to maintain balanced sentiment classes across training and evaluation data.

The current hybrid word + character TF-IDF model achieved approximately **68% test accuracy** on the prepared dataset.

Rather than presenting accuracy alone, the project focuses on improving practical behavior on difficult user-generated text through:

* Character-level features
* Negation handling
* Contrast-aware rules
* Informal language handling
* Confidence estimation

> **Note:** Sentiment classification is inherently difficult for sarcasm, ambiguity, context-dependent language, and mixed opinions. The project is designed as a practical NLP application rather than a perfect sentiment detector.

---

## 🚀 Future Improvements

Potential future improvements include:

* Transformer-based models such as BERT
* Fine-tuning a pretrained language model
* Improved sarcasm detection
* Multilingual sentiment analysis
* Real-time social media data ingestion
* PostgreSQL for production database persistence
* REST API for external applications
* Automated model evaluation and monitoring
* CI/CD pipeline
* Docker containerization
* Model versioning and experiment tracking

---

## 🎯 Learning Outcomes

Through this project, I worked with:

* Natural Language Processing
* Text classification
* Feature engineering
* TF-IDF
* Character n-grams
* Logistic Regression
* Model evaluation
* Flask application development
* SQLite database integration
* Git/GitHub version control
* Cloud deployment
* Handling real-world noisy text

---

## 👨‍💻 Author

**Keyur Chaudhary**

B.Tech CSE (AI & ML)

Interested in:

* Machine Learning
* Natural Language Processing
* Artificial Intelligence
* Python
* Data Science

---

## ⭐ Project

If you find this project useful or interesting, consider giving the repository a ⭐.

**Live Demo:**
https://sentiment-analysis-tweets-b3ht.onrender.com

**GitHub Repository:**
https://github.com/Lucifer666-ms/sentiment-analysis-tweets
