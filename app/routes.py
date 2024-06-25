from flask import request, jsonify, render_template
from app import app
import joblib
import os
import numpy as np

# Define the paths to the model and vectorizer
model_path = os.path.join(os.path.dirname(__file__), '..', 'ML_Model', 'fake_news_detector.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'ML_Model', 'vectorizer.pkl')

# Load the trained model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def preprocess(text):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data['text']
        clean_text = preprocess(text)
        vectorized_text = vectorizer.transform([clean_text])
        prediction = model.predict(vectorized_text)[0]
        # Convert numpy.int64 to Python int
        prediction = int(prediction)
        return jsonify({'prediction': prediction})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500
