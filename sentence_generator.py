import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from database import pos_extractor
from sklearn.model_selection import train_test_split

# training pos structure from dialogue 1 then 2. input: pos person 1, output: pos person 2
# sentence subject. input: sentence 1 subject reply, output: sentence 2 subject reply
# train the inverse to be able to classify reply vs answer
# markov sentence generator to generate x length setences and appropriate length

# cut setence length then classify, use markov chains to generate the text again


def encoder(data):
    pos_tags_spacy_glossary = (
        "ADJ",
        "ADP",
        "ADV",
        "AUX",
        "CONJ",
        "CCONJ",
        "DET",
        "INTJ",
        "NOUN",
        "NUM",
        "PART",
        "PRON",
        "PROPN",
        "PUNCT",
        "SCONJ",
        "SYM",
        "VERB",
        "X",
        "EOL",
        "SPACE",
    )
    return pos_tags_spacy_glossary.index(data.upper())


def sequence_encoder(data):
    return [encoder(x) for x in pos_extractor(data)]


data = (
    pd.read_csv("casual_data_windows.csv", index_col=0, nrows=1000, dtype="string")
    .iloc[:, 0:2]
    .astype("string")
)

conversational_data = data.applymap(
    func=lambda x: sequence_encoder(x)[0:6] if len(x.split()) >= 6 else np.nan
).dropna(how="any", axis=0)

xtrain, xtest, ytrain, ytest = train_test_split(
    conversational_data.iloc[:, 0],
    conversational_data.iloc[:, 1],
    train_size=0.8,
    random_state=42,
)

