import pandas as pd
import spacy as sp
from spacy.pipeline.textcat import *
import numpy as np
from spacy.strings import StringStore
from multiprocessing import Process
from psutil import cpu_count
import math


twitter_file = (
    r"C:\Users\crapg\OneDrive\Documents\datasets\twitter customer support\twcs.csv"
)

nlp = sp.load("en_core_web_sm")

vectors = pd.DataFrame()

string_store = StringStore()

# form any column base dataset, enables selection of the text data column
def tagger(data_csv):
    columns = dict({})
    for i in range(1, len(data_csv.columns)):
        columns[i] = data_csv.columns[i - 1]
    print(data_csv.iloc[1, :])
    print(list(data_csv.columns), "pick between 1 and " + str(len(data_csv.columns)))
    number = int(input())
    text_data = data_csv.iloc[:, number - 1]
    return text_data


# stores text in string store to avoid duplication and named entitities. hash lookup
def string_storing(text):
    for i in range(1, len(text.index)):
        doc = nlp(text[i - 1])
        for token in doc:
            if token.text in [x.text for x in doc.ents]:
                continue
            string_store.add(token.text)


# converts words to vectors and stores them
def vector_storing():
    global vectors
    for s in string_store:
        word = nlp(s)[0]
        vectors = pd.concat(
            [vectors, pd.DataFrame(word.vector, columns=[word.text])], axis=1
        )
    vectors.to_excel("vectors_database.xlsx", index=False)


def csv_to_json(fileName):
    data_csv = pd.read_csv(fileName)
    text_column = tagger(data_csv)
    nm_datasets = math.floor(len(data_csv.index) / cpu_count())
    processes = []
    rows = nm_datasets

    # avoids dataframe unpacking
    class wrapper:
        def __init__(self, num1, num2):
            self.txt = text_column[num1:num2]

    for subprocess in range(0, cpu_count()):
        processes.append(
            Process(target=string_storing, args=wrapper(rows - nm_datasets, rows).txt)
        )
        rows += nm_datasets
    for process in processes:
        process.start()
    vector_storing()

