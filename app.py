from flask import Flask, render_template, request
import pickle
import sqlite3
import re
from scipy.sparse import hstack

app = Flask(__name__)

# =========================================================
# LOAD MODEL AND VECTORIZERS
# =========================================================

model = pickle.load(open("model.pkl", "rb"))
word_vectorizer = pickle.load(open("word_vectorizer.pkl", "rb"))
char_vectorizer = pickle.load(open("char_vectorizer.pkl", "rb"))


# =========================================================
# DATABASE
# =========================================================

def create_database():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            sentiment TEXT,
            confidence REAL
        )
    """)

    # Check whether confidence column already exists
    c.execute("PRAGMA table_info(reviews)")
    columns = [column[1] for column in c.fetchall()]

    if "confidence" not in columns:
        c.execute("""
            ALTER TABLE reviews
            ADD COLUMN confidence REAL
        """)

    conn.commit()
    conn.close()


create_database()


def insert_data(text, result, confidence):
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("""
        INSERT INTO reviews
        (text, sentiment, confidence)
        VALUES (?, ?, ?)
    """, (
        text,
        result,
        confidence
    ))

    conn.commit()
    conn.close()


# =========================================================
# TEXT PREPROCESSING
# =========================================================

def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove mentions
    text = re.sub(r"@\w+", " ", text)

    # Keep hashtag word
    text = re.sub(r"#(\w+)", r" \1 ", text)

    # Reduce repeated characters
    # Example:
    # goooooood -> gooood
    # worsttttt -> worsttt
    text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)

    # Keep only English letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================================================
# SENTIMENT PREDICTION
# =========================================================

def predict_sentiment(text):

    text_lower = str(text).lower().strip()

    # -----------------------------------------------------
    # ML PREDICTION
    # -----------------------------------------------------

    cleaned = clean_text(text)

    word_features = word_vectorizer.transform([cleaned])
    char_features = char_vectorizer.transform([cleaned])

    combined_features = hstack([
        word_features,
        char_features
    ])

    probabilities = model.predict_proba(combined_features)[0]

    result = model.predict(combined_features)[0]

    confidence = max(probabilities) * 100

    # -----------------------------------------------------
    # FORCE VALID SENTIMENT
    # -----------------------------------------------------

    if result not in ["positive", "negative", "neutral"]:
        result = "neutral"

    # =====================================================
    # STRONG NEGATIVE PHRASES
    # =====================================================

    strong_negative = [
        "worst",
        "worst experience",
        "worst ever",
        "terrible",
        "horrible",
        "awful",
        "garbage",
        "useless",
        "pathetic",
        "disgusting",
        "hate",
        "hated",
        "fucking worst",
        "fuck this",
        "wtf",
        "waste of time",
        "waste of money",
        "never again",
        "extremely disappointed",
        "very disappointed",
        "so bad",
        "really bad",
        "too bad",
        "bad experience",
        "bad service",
        "poor service",
        "poor experience",
        "not working",
        "does not work",
        "doesn't work",
        "keeps crashing",
        "keep crashing"
    ]

    # =====================================================
    # STRONG POSITIVE PHRASES
    # =====================================================

    strong_positive = [
        "amazing",
        "awesome",
        "excellent",
        "fantastic",
        "perfect",
        "love",
        "loved",
        "brilliant",
        "wonderful",
        "best",
        "fucking amazing",
        "really good",
        "very good",
        "so good",
        "great",
        "really great",
        "works perfectly",
        "highly recommend",
        "good experience",
        "great experience"
    ]

    # =====================================================
    # NEGATION
    # =====================================================

    positive_after_negation = [
        "not bad",
        "not terrible",
        "not awful",
        "not horrible",
        "not bad actually",
        "don't hate",
        "do not hate",
        "doesn't hate",
        "does not hate",
        "never hated"
    ]

    negative_after_negation = [
        "not good",
        "not great",
        "not amazing",
        "not excellent",
        "don't like",
        "do not like",
        "doesn't like",
        "does not like",
        "never liked"
    ]

    # =====================================================
    # NEUTRAL PHRASES
    # =====================================================

    neutral_phrases = [
        "okay",
        "ok",
        "average",
        "nothing special",
        "nothing great",
        "nothing amazing",
        "could be better",
        "meh",
        "fine i guess",
        "not bad not good",
        "neither good nor bad",
        "nothing much",
        "just okay",
        "just ok"
    ]

      # =====================================================
    # BUT LOGIC
    # The opinion AFTER "but" gets priority
    # =====================================================

    if re.search(r"\bbut\b", text_lower):

        parts = re.split(
            r"\s+\bbut\b\s+",
            text_lower,
            maxsplit=1
        )

        if len(parts) == 2:

            before_but = parts[0].strip()
            after_but = parts[1].strip()

            # ---------------------------------------------
            # POSITIVE EXPRESSIONS AFTER BUT
            # ---------------------------------------------

            positive_after_but = [
                "i like",
                "i liked",
                "i love",
                "i loved",
                "i enjoy",
                "i enjoyed",
                "i really like",
                "i really liked",
                "i really love",
                "i really loved",
                "i actually like",
                "i actually liked",
                "i actually love",
                "i actually loved",
                "i would recommend",
                "i recommend",
                "looks good",
                "look good",
                "feels good",
                "works well",
                "works perfectly",
                "is good",
                "is great",
                "is amazing",
                "is excellent",
                "is awesome",
                "is fantastic",
                "is perfect",
                "was good",
                "was great",
                "was amazing",
                "was excellent",
                "was awesome",
                "was fantastic",
                "was perfect",
                "very good",
                "really good",
                "so good",
                "quite good",
                "pretty good",
                "actually good",
                "actually great",
                "actually amazing"
            ]

            # ---------------------------------------------
            # NEGATIVE EXPRESSIONS AFTER BUT
            # ---------------------------------------------

            negative_after_but = [
                "i hate",
                "i hated",
                "i dislike",
                "i disliked",
                "i don't like",
                "i do not like",
                "i don't love",
                "i do not love",
                "is bad",
                "is terrible",
                "is horrible",
                "is awful",
                "is garbage",
                "is useless",
                "is pathetic",
                "was bad",
                "was terrible",
                "was horrible",
                "was awful",
                "was garbage",
                "was useless",
                "was pathetic",
                "keeps crashing",
                "keep crashing",
                "doesn't work",
                "does not work",
                "not good",
                "not great",
                "not amazing",
                "very bad",
                "really bad",
                "so bad"
            ]

            # ---------------------------------------------
            # FIRST CHECK POSITIVE AFTER BUT
            # ---------------------------------------------

            if any(
                phrase in after_but
                for phrase in positive_after_but
            ):
                result = "positive"
                confidence = max(confidence, 90.0)

            # ---------------------------------------------
            # THEN CHECK NEGATIVE AFTER BUT
            # ---------------------------------------------

            elif any(
                phrase in after_but
                for phrase in negative_after_but
            ):
                result = "negative"
                confidence = max(confidence, 90.0)

            # ---------------------------------------------
            # OTHERWISE USE ML RESULT FOR AFTER-BUT TEXT
            # ---------------------------------------------

            else:
                after_word_features = word_vectorizer.transform(
                    [clean_text(after_but)]
                )

                after_char_features = char_vectorizer.transform(
                    [clean_text(after_but)]
                )

                after_features = hstack([
                    after_word_features,
                    after_char_features
                ])

                after_prediction = model.predict(after_features)[0]

                after_probabilities = model.predict_proba(
                    after_features
                )[0]

                after_confidence = max(after_probabilities) * 100

                if after_prediction in [
                    "positive",
                    "negative",
                    "neutral"
                ]:
                    result = after_prediction
                    confidence = max(
                        confidence,
                        after_confidence
                    )

    # =====================================================
    # NORMAL NEGATION LOGIC
    # =====================================================

    elif any(
        phrase in text_lower
        for phrase in positive_after_negation
    ):
        result = "positive"
        confidence = max(confidence, 90.0)

    elif any(
        phrase in text_lower
        for phrase in negative_after_negation
    ):
        result = "negative"
        confidence = max(confidence, 90.0)

    # =====================================================
    # STRONG NEGATIVE
    # =====================================================

    elif any(
        phrase in text_lower
        for phrase in strong_negative
    ):
        result = "negative"
        confidence = max(confidence, 85.0)

    # =====================================================
    # STRONG POSITIVE
    # =====================================================

    elif any(
        phrase in text_lower
        for phrase in strong_positive
    ):
        result = "positive"
        confidence = max(confidence, 85.0)

    # =====================================================
    # NEUTRAL
    # =====================================================

    elif any(
        phrase in text_lower
        for phrase in neutral_phrases
    ):
        result = "neutral"
        confidence = max(confidence, 75.0)

    # =====================================================
    # FINAL VALIDATION
    # =====================================================

    if result not in ["positive", "negative", "neutral"]:
        result = "neutral"

    return result, round(confidence, 2)


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# PREDICT
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    text = request.form.get("tweet", "").strip()

    if not text:
        return render_template(
            "index.html",
            result="neutral",
            confidence=0,
            text="",
            error="Please enter some text."
        )

    result, confidence = predict_sentiment(text)

    insert_data(
        text,
        result,
        confidence
    )

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        text=text
    )


# =========================================================
# ADMIN PANEL
# =========================================================

@app.route("/admin")
def admin():

    create_database()

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("""
        SELECT
            id,
            text,
            sentiment,
            confidence
        FROM reviews
        ORDER BY id DESC
    """)

    data = c.fetchall()

    conn.close()

    return render_template(
        "admin.html",
        data=data
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)
