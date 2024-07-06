# Article_type_classification

Creating a README file for your GitHub repository involves detailing the purpose of the project, the dependencies, installation steps, and usage instructions. Below is a sample README file for your article type classification project:

---

# Article Type Classification

This repository contains a machine learning project for classifying articles into various categories such as 'Commercial', 'Military', 'Executives', 'Others', 'Support & Services', 'Financing', and 'Training'. The project utilizes text data and employs natural language processing (NLP) techniques and machine learning models to perform the classification.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

The goal of this project is to build a text classification model that can accurately predict the type of an article based on its content. The project includes data preprocessing, model training, and deployment using Flask and ngrok.

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
├── unknown_articles.csv                     # Dataset with unknown articles for prediction
└── url_article_type_prediction.ipynb        # Notebook for URL based predictions
```

## Dependencies

To install the required dependencies, use the following command:

```bash
pip install -r requirements.txt
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/mcPython95/Article_type_classification.git
cd Article_type_classification
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Download and place the necessary models and data files in the project directory:

- `article_type_classifier_model.pkl`
- `class_names.pkl`
- `articles.csv`
- `unknown_articles.csv`

## Usage

### Running the Flask App

1. Start the Flask app:

```bash
python app.py
```

2. Expose the Flask app to the internet using ngrok:

```bash
ngrok http 5000
```

3. Access the app through the provided ngrok URL.

### Predicting Article Types

You can use the `url_article_type_prediction.ipynb` notebook to predict the types of articles from a list of URLs. Simply open the notebook and follow the instructions.

### Training the Model

To train the model, use the `Article_type_classification_final.ipynb` notebook. This notebook contains the complete workflow for data preprocessing, model training, and evaluation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

---

Feel free to customize this README file as per your specific project details and requirements.
