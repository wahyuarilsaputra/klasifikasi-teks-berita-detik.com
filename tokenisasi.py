import nltk
from nltk.tokenize import word_tokenize

nltk.download('popular')

def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens
