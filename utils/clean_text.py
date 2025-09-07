# utils/clean_text.py

import json
import re
import string

# # save stopwords
# import nltk
# from nltk.corpus import stopwords

# nltk.download('stopwords', quiet=True)
# stop_words = list(stopwords.words("english"))

# with open("stopwords_en.json", "w") as f:
#     json.dump(stop_words, f)

with open("stopwords_en.json") as f:
    stop_words = set(json.load(f))

def clean_text(text):
    # lowercase
    text = text.lower()
    # remove punctuation
    text = re.sub(f"[{string.punctuation}]", "", text)  
    # remove digits
    text = re.sub(r'\d+', '', text)  
    # remove stop words
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)