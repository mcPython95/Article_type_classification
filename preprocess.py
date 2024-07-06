# preprocess_pipeline.py
import re
import spacy # type: ignore
from spacy.lang.en.stop_words import STOP_WORDS # type: ignore
import html
# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def preprocess_text(texts, lemmatize=True, handle_html=True):
    preprocessed_texts = []

    for text in texts:
        # Handle HTML tags and entities
        if handle_html:
            text = html.unescape(text)
            text = re.sub(r'<[^>]+>', '', text)

        # Convert to lowercase (if not handled by spacy model)
        text = text.lower()

        # Replace non-standard apostrophes and other special characters
        text = re.sub(r'[^\x00-\x7F]+', '', text)

        # Tokenize using spacy
        doc = nlp(text)

        # Lemmatize and remove stopwords and punctuations
        words = []
        for token in doc:
            if token.is_stop or token.is_punct:
                continue
            if lemmatize:
                words.append(token.lemma_)
            else:
                words.append(token.text)

        # Join the words back into a single string
        preprocessed_text = ' '.join(words)

        # Additional check to remove leading and trailing spaces
        preprocessed_text = preprocessed_text.strip()

        preprocessed_texts.append(preprocessed_text)

    return preprocessed_texts
