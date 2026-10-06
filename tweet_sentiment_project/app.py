from flask import Flask, render_template, request
import pickle
import sqlite3

app = Flask(__name__)

# Load trained files
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# ------------------ DATABASE FUNCTION ------------------
def insert_data(text, result):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS reviews (text TEXT, sentiment TEXT)")
    c.execute("INSERT INTO reviews VALUES (?, ?)", (text, result))
    conn.commit()
    conn.close()

# ------------------ HOME ------------------
@app.route('/')
def home():
    return render_template('index.html')

# ------------------ PREDICT ------------------
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['tweet']
    text_lower = text.lower()

    # ML Prediction
    data = vectorizer.transform([text])
    result = model.predict(data)[0]

    # 🔥 RULE-BASED IMPROVEMENTS

    # Strong negative words
    negative_keywords = ["worst", "hate", "terrible", "bad", "useless", "fucking worst"]

    # Strong positive words
    positive_keywords = ["good", "great", "amazing", "awesome", "excellent"]

    # Apply rules
    if any(word in text_lower for word in negative_keywords):
        result = "negative"
    elif any(word in text_lower for word in positive_keywords):
        result = "positive"

    # Negation handling
    if "not good" in text_lower:
        result = "negative"

    # Mixed sentiment
    if "but" in text_lower:
        result = "neutral"

    # Save to database
    insert_data(text, result)

    return render_template('index.html', result=result, text=text)

# ------------------ ADMIN PANEL ------------------
@app.route('/admin')
def admin():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM reviews")
    data = c.fetchall()
    conn.close()
    return render_template('admin.html', data=data)

# ------------------ RUN ------------------
if __name__ == "__main__":
    app.run(debug=True)