import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from database import pos_extractor
from sklearn.model_selection import train_test_split
import functools as ft

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


# takes list of index of pos taggers
def t_plus_encoder(data, point):
    data = list(data)
    m = list(map(lambda a: 20 - a, data))
    flattened_indexing = [
        sum(data[0 : i + 1]) + x + sum(m[0 : i + 1]) if i >= 0 else x
        for x, i in zip(data, range(-1, 5))
    ]
    if point == "i":
        individual_encoding = np.array([np.zeros(shape=(20, 1)) for x in data])
        individual_encoding.put(flattened_indexing, 1)
        return individual_encoding
    elif point == "o":
        data_mod = data[1:]
        flattened_indexing_t = [
            sum(data_mod[0 : i + 1]) + x + sum(m[0 : i + 1]) if i >= 0 else x
            for x, i in zip(data, range(-1, 4))
        ].insert(5, -500)
        individual_encoding = np.array([np.zeros(shape=(20, 1)) for x in data])
        individual_encoding.put(flattened_indexing, 1)
        t_minus_encoding = np.array([np.zeros(shape=(20, 1)) for x in data])
        t_minus_encoding.put(flattened_indexing_t, 1)
        t_minus_superposed = np.array()
        for num, (i, t) in enumerate(zip(individual_encoding, t_minus_encoding)):
            if num == 5:
                np.append(t_minus_superposed, np.add(i, 0), axis=1)
            else:
                np.append(t_minus_superposed, np.add(i, t), axis=1)
        t_minus_superposed[t_minus_superposed == 2] = 1
        return individual_encoding, t_minus_superposed
    else:
        raise ValueError("need proper point value")


data = (
    pd.read_csv("casual_data_windows.csv", index_col=0, nrows=1000, dtype="string")
    .iloc[:, 0:2]
    .astype("string")
)

conversational_data = data.applymap(
    func=lambda x: sequence_encoder(x)[0:6] if len(x.split()) >= 6 else np.nan
).dropna(how="any", axis=0)

# input_matrix = list()
# output_matrix = list()
# output_t = list()

# for i in conversational_data.index.tolist():
#     input_matrix = np.append(
#         input_matrix, t_plus_encoder(conversational_data.iloc[i, 0], "i"), axis=0
#     )
# input_matrix = np.array(input_matrix)
# output_matrix = np.array(output_matrix)
# output_t = np.array(output_t)
# print(input_matrix)
print(conversational_data.iloc[0, 1])
print(t_plus_encoder(conversational_data.iloc[0, 0], "o"))

# xtrain, xtest, ytrain, ytest = train_test_split(
#     conversational_data.iloc[:, 0],
#     conversational_data.iloc[:, 1],
#     train_size=0.8,
#     random_state=42,
# )

