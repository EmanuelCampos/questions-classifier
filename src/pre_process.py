import pandas as pd
import re
import nltk

nltk.download('punkt')

from nltk.tokenize import word_tokenize

# ================================================
lines = 10000

# ================================================

df = pd.read_csv('./src/assets/questions.csv')

df = df.head(lines)

def clean_text(text):
    text = re.sub('<[^>]*>', '', text) # remove tags HTML
    text = re.sub('[^a-zA-Z]', ' ', text) # remove caracteres especiais e números
    text = text.lower() # transforma todas as letras em minúsculas
    return text

df['Body'] = df['Body'].apply(clean_text)
df['Body'] = df['Body'].apply(word_tokenize)

word2idx = {}

for sentence in df['Body']:
    for word in sentence:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

df['Body'] = df['Body'].apply(lambda sentence: [word2idx[word] for word in sentence])

df.to_csv('./src/csv/file.csv', index=False)