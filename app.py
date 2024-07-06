from flask import Flask, request, jsonify, render_template_string
from sentence_transformers import SentenceTransformer
#form preprocess.py
from preprocess import preprocess_text 
from lxml.html.clean import Cleaner
import pickle
import numpy as np
import pandas as pd
from newspaper import Article


# Load the model
with open('article_type_classifier_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load class names
with open('class_names.pkl', 'rb') as file:
    class_names = pickle.load(file)

# Load pre-trained SentenceBERT model
st_model = SentenceTransformer('bert-base-nli-mean-tokens')

app = Flask(__name__)

# HTML template for the form
html_form_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>Article Type Classifier</title>
</head>
<body>
    <h1>Article Type Classifier</h1>
    <form action="/predict" method="post" id="predictionForm">
        <p><input type="radio" id="inputText" name="inputType" value="text" checked>
        <label for="inputText">Enter Text:</label><br>
        <label for="heading">Heading:</label><br>
        <input type="text" id="heading" name="heading" ><br>
        <label for="full_article">Full Article:</label><br>
        <textarea id="full_article" name="full_article" rows="10" cols="50" ></textarea></p>

        <p><input type="radio" id="inputUrl" name="inputType" value="url">
        <label for="inputUrl">Enter URL:</label><br>
        <label for="url">URL:</label><br>
        <input type="text" id="url" name="url"></p>

        <input type="submit" value="Classify">
    </form>
</body>
</html>
'''

# HTML template for displaying prediction
html_prediction_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>Article Type Classifier - Prediction</title>
</head>
<body>
    <h1>Article Type Classifier - Prediction</h1>
    <h2>Prediction: {{ predicted_label }}</h2>
    <br>
    <form action="/" method="get">
        <button type="submit">Go Back</button>
    </form>
</body>
</html>
'''

def preprocess_and_predict(heading, full_article):
    # Preprocess the text
    pre_processed_text = [preprocess_text([heading])[0], preprocess_text([full_article])[0]]

    # Vectorize the text columns
    head_emd = st_model.encode(pre_processed_text[0])
    art_emd = st_model.encode(pre_processed_text[1])

    X_new = np.hstack((head_emd, art_emd))

    # Make predictions
    y_pred = model.predict([X_new])
    predicted_label = class_names[y_pred[0]]

    return predicted_label

def extract_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()

        heading = article.title
        full_article = article.text

        return heading, full_article
    except Exception as e:
        print(f"Error extracting {url}: {e}")
        return None, None

@app.route('/', methods=['GET'])
def form():
    return render_template_string(html_form_template)

@app.route('/predict', methods=['POST'])
def predict():
    input_type = request.form.get('inputType')

    if input_type == 'text':
        heading = request.form.get('heading')
        full_article = request.form.get('full_article')
        url = None
    elif input_type == 'url':
        url = request.form.get('url')
        heading, full_article = extract_article_content(url)
        if heading is None or full_article is None:
            return jsonify({'error': 'Failed to extract article content from URL'}), 400

    if not heading or not full_article:
        return jsonify({'error': 'Heading or Full Article not provided'}), 400

    predicted_label = preprocess_and_predict(heading, full_article)
    
    return render_template_string(html_prediction_template, predicted_label=predicted_label)

if __name__ == '__main__':
    app.run()
