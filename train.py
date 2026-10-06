import pandas as pd
import re
import pickle

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ==============================
# LOAD FINAL DATASET
# ==============================

df = pd.read_csv("data_final.csv")

print("Dataset loaded successfully!")
print(f"Total rows: {len(df)}")
print(df["sentiment"].value_counts())
print()


# ==============================
# TEXT CLEANING
# ==============================

def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove mentions
    text = re.sub(r"@\w+", " ", text)

    # Keep hashtag word
    text = re.sub(r"#(\w+)", r" \1 ", text)

    # Handle common contractions
    contractions = {
        "can't": "can not",
        "won't": "will not",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "couldn't": "could not",
        "wouldn't": "would not",
        "shouldn't": "should not",
        "i'm": "i am",
        "you're": "you are",
        "we're": "we are",
        "they're": "they are",
        "it's": "it is",
        "that's": "that is"
    }

    for contraction, replacement in contractions.items():
        text = text.replace(contraction, replacement)

    # Normalize extremely repeated letters
    # goooood -> goood
    # worsttttt -> worsttt
    text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)

    # Keep letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["text"] = df["text"].apply(clean_text)

df = df[df["text"].str.len() > 2]

print(f"Rows after cleaning: {len(df)}")
print()


# ==============================
# FEATURES & LABELS
# ==============================

X = df["text"]
y = df["sentiment"]


# ==============================
# TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==============================
# WORD TF-IDF
# ==============================

word_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    max_features=40000
)

X_train_word = word_vectorizer.fit_transform(X_train)
X_test_word = word_vectorizer.transform(X_test)

print("Word TF-IDF created!")
print(
    f"Word features: "
    f"{len(word_vectorizer.get_feature_names_out())}"
)


# ==============================
# CHARACTER TF-IDF
# ==============================

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    sublinear_tf=True,
    max_features=30000
)

X_train_char = char_vectorizer.fit_transform(X_train)
X_test_char = char_vectorizer.transform(X_test)

print("Character TF-IDF created!")
print(
    f"Character features: "
    f"{len(char_vectorizer.get_feature_names_out())}"
)

print()


# ==============================
# COMBINE FEATURES
# ==============================

X_train_combined = hstack([
    X_train_word,
    X_train_char
])

X_test_combined = hstack([
    X_test_word,
    X_test_char
])

print("Word + Character features combined!")
print()


# ==============================
# TRAIN MODEL
# ==============================

model = LogisticRegression(
    max_iter=500,
    C=2.0,
    class_weight="balanced"
)

print("Training model...")

model.fit(
    X_train_combined,
    y_train
)

print("Training completed!")
print()


# ==============================
# EVALUATION
# ==============================

predictions = model.predict(X_test_combined)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("==============================")
print("MODEL PERFORMANCE")
print("==============================")
print(
    f"Accuracy: {accuracy * 100:.2f}%"
)
print()

print("Classification Report:")

print(
    classification_report(
        y_test,
        predictions
    )
)


# ==============================
# SAVE MODEL
# ==============================

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("word_vectorizer.pkl", "wb") as file:
    pickle.dump(
        word_vectorizer,
        file
    )

with open("char_vectorizer.pkl", "wb") as file:
    pickle.dump(
        char_vectorizer,
        file
    )


print("==============================")
print("MODEL SAVED SUCCESSFULLY")
print("==============================")
print("model.pkl")
print("word_vectorizer.pkl")
print("char_vectorizer.pkl")
print("==============================")