import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from database import pos_extractor
from sklearn.model_selection import train_test_split
import functools as ft


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
            for x, i in zip(data_mod, range(-1, 4))
        ]

        individual_encoding = np.array([np.zeros(shape=(20, 1)) for x in data])
        individual_encoding.put(flattened_indexing, 1)
        t_minus_encoding = np.array([np.zeros(shape=(20, 1)) for x in data])
        t_minus_encoding.put(flattened_indexing_t, 1)
        t_minus_superposed = np.add(individual_encoding, t_minus_encoding)
        t_minus_superposed[t_minus_superposed == 2] = 1
        return individual_encoding, t_minus_superposed
    else:
        raise ValueError("need proper point value")


def martix_creation(data, direction):
    input_matrix = None
    output_matrix = None
    output_t = None

    if direction == "i":
        for i in data.index.tolist():
            if i == 0:
                input_matrix = t_plus_encoder(data.iloc[i, 0], "i")
            else:
                input_matrix = np.append(
                    input_matrix, t_plus_encoder(data.iloc[i, 0], "i"), axis=0
                )
        return input_matrix
    if direction == "o":
        for i in data.index.tolist():
            if i == 0:
                output_matrix, output_t = t_plus_encoder(data.iloc[i, 1], "o")
            else:
                p1, p2 = t_plus_encoder(data.iloc[i, 1], "o")
                output_matrix = np.append(output_matrix, p1, axis=0)
                output_t = np.append(output_matrix, p2, axis=0)
        return output_matrix, output_t
    else:
        raise ValueError("need proper point value")


def seq2seq_model(encoder_input, decoder_input, decoder_target, batch_size, epochs):
    latent_space_dimension = 256
    encoder_inputs = tf.keras.Input(shape=(None, 6))
    encoder = tf.keras.layers.LSTM(latent_space_dimension, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    decoder_inputs = keras.Input(shape=(None, 6))
    decoder_lstm = keras.layers.LSTM(
        latent_space_dimension, return_sequences=True, return_state=True
    )
    decoder_outputs, *_ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = keras.layers.Dense(20, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    def training():
        model.compile(
            optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        model.fit(
            [encoder_input, decoder_input],
            decoder_target,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
        )

        return model

    training()


data = (
    pd.read_csv("casual_data_windows.csv", index_col=0, nrows=1000, dtype="string")
    .iloc[:, 0:2]
    .astype("string")
    .applymap(
        func=lambda x: sequence_encoder(x)[0:6] if len(x.split()) >= 6 else np.nan
    )
    .dropna(how="any", axis=0)
    .reset_index(drop=True)
)

xtrain, xtest, ytrain, ytest = train_test_split(
    data.iloc[:, 0], data.iloc[:, 1], train_size=0.8, random_state=42,
)

