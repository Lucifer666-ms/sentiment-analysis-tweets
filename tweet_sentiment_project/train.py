import pandas as pd
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv('data.csv')

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

df['text'] = df['text'].apply(clean_text)

# Features & Labels
X = df['text']
y = df['sentiment']

# 🔥 Improved TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)

X_vec = vectorizer.fit_transform(X)

# 🔥 Improved Model
model = LogisticRegression(max_iter=200)

# Train
model.fit(X_vec, y)

# Save new model
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("✅ Model retrained successfully!")