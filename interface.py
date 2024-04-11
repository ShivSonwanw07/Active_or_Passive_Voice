
import streamlit as st
import pandas as pd 
import pickle
import numpy as np
import string 
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

st.title('Text Classification Model to Identify Active or Passive Voice Statement')
labels = {'Active': 0, 'Passive': 1}
nlp = spacy.load("en_core_web_sm")

# Load the TfidfVectorizer used during training
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

def preprocess_text(text):
    return " ".join([token.lemma_ for token in nlp(text)])

punc = string.punctuation
model = pickle.load(open('Active_Passive.pkl', 'rb'))

def predict(statement):
    statement = statement.lower()
    statement = "".join(i for i in statement if i not in punc)
    processed_statement = preprocess_text(statement)

    tfidf_features = tfidf_vectorizer.transform([processed_statement])
    logi_pred = model.predict(tfidf_features)
    for key, value in labels.items():
        if value == logi_pred[0]:
            return key
        
statement = st.text_input('Enter the Statement:-')

if st.button('Identify!'):
    a = predict(statement)
    st.header(f"The given Statement is in {a} Voice")
