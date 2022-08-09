import sklearn
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from database import pos_extractor

# training pos structure from dialogue 1 then 2. input: pos person 1, output: pos person 2
# sentence subject. input: sentence 1 subject reply, output: sentence 2 subject reply
# train the inverse to be able to classify reply vs answer
# markov sentence generator to generate x length setences and appropriate length

conversational_data = pd.read_csv("casual_data_windows.csv", index_col=0)


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


model_pos_forward = tf.keras.Sequential()

