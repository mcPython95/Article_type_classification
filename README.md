# Smart Article Type Text Classification with Machine Learning : Leveraging SentenceBERT and Linear SVC for Accurate Category Prediction and Serving the trained model thorugh an API Endpoint 

This repository contains a machine learning project for classifying articles based on the text data into various categories such as 'Commercial', 'Military', 'Executives', 'Others', 'Support & Services', 'Financing', and 'Training'. The project utilizes text data and employs natural language processing (NLP) techniques and machine learning models to perform the classification.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

The goal of this project is to build a text classification model that can accurately predict the type of an article based on its content. The project includes data preprocessing, model training, and deployment using Flask.

## Project Structure

```
Article_type_classification/
│
├── Article_type_classification_final.ipynb  # Notebook for final model training and evaluation
├── README.md                                # This README file
├── app.py                                   # Flask application for serving the model
├── article_type_classifier_model.pkl        # Trained model
├── articles.csv                             # Dataset containing articles
├── class_names.pkl                          # Pickle file containing class names
├── preprocess.py                            # Script for data preprocessing
├── requirements.txt                         # Dependencies file
├── unknown_articles.csv                     # Dataset with unknown articles URLs for prediction
└── url_article_type_prediction.ipynb        # Notebook for URL based predictions
```

## Dependencies

To install the required dependencies, use the following command:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Installation

1. Clone the repository (in Colab):

```bash
git clone https://github.com/mcPython95/Article_type_classification.git
cd Article_type_classification
```

### Training the Model

To train the model, use the `Article_type_classification_final.ipynb` notebook. This notebook contains the complete workflow for data preprocessing, model training, and evaluation.


### Download files 

Download the data files in the project directory:

1. `articles.csv` - This file contains the dataset of known articles used for training and evaluating the classification model. It includes features and labels necessary for model development.

2. `unknown_articles.csv` - This file contains a list of links to articles with unknown categories. These links will need to be accessed and the content extracted for classification by the trained model.


### Save the model 

1. `article_type_classifier_model.pkl` - This file contains the trained model for classifying article types. It is a pickled object of your model and can be loaded using `pickle.load()`.

2. `class_names.pkl` - This file contains the list of class names used by the model. It is a pickled object that maps the class indices to human-readable names. It can be loaded using `pickle.load()`.

### Predicting Article Types

You can use the `url_article_type_prediction.ipynb` notebook to predict the types of articles from a list of URLs. Simply open the notebook and follow the instructions.


## Usage

### Running the Flask App (Use VScode or any code editor)

1. Download and place the necessary models and data files in the project directory:

   - `article_type_classifier_model.pkl`
   - `class_names.pkl`
     
2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the virtual environment:

   - **On Windows:**

     ```bash
     venv\Scripts\activate
     ```

   - **On macOS/Linux:**

     ```bash
     source venv/bin/activate
     ```

4. Install the dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```     

5. Start the Flask app:

```bash
python app.py
```



## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
