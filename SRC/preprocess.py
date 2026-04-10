import re

def preprocess(text):
    text = text.lower()
    
    # remove punctuation + numbers
    text = re.sub(r'[^a-z\s]', '', text)

    words = text.split()

    return words