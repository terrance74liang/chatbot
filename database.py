import multiprocessing
import pandas as pd
import spacy as sp
import numpy as np
from multiprocessing import Process, Manager
from psutil import cpu_count
import math
from sqlalchemy import true

nlp = sp.load("en_core_web_sm")

vectors = pd.DataFrame()

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


# stores text dicitonnary and eliminates named entitities, text is a pandas series
def string_storing(text, string_store, list, inc):
    text = text.reset_index(drop=True)
    for i in range(1, len(text.index)):
        doc = nlp(text[i - 1])
        for token in doc:
            if (token.text in [x.text for x in doc.ents]) or (
                str(token.tag_).upper() in ["SYM", "X", "NUM"]
            ):
                continue
            elif str(token.tag_).upper() == "PUNCT" and (
                token.text not in string_store.get("PUNCT")
            ):
                string_store.get("PUNCT").append(token.text)
            elif token.text[0].lower() not in string_store.keys():
                string_store[token.text[0].lower()] = list[inc.value]
                inc.value += 1
            else:
                string_store.get(token.text[0].lower()).append(token.text.lower())
                string_store.get(token.text[0].lower()).sort()


# converts words to vectors and stores them
# def vector_storing():
#     global vectors
#     for s in string_store:
#         word = nlp(s)[0]
#         vectors = pd.concat(
#             [vectors, pd.DataFrame(word.vector, columns=[word.text])], axis=1
#         )
#     vectors.to_excel("vectors_database.xlsx", index=False)


def csv_to_json(fileName):
    inc = multiprocessing.Value("i", 0)
    with multiprocessing.Manager() as manager_, multiprocessing.Manager() as list_manager:
        list_of_managers = [list_manager.list() for i in range(0, 100)]
        string_store = manager_.dict()
        string_store["PUNCT"] = list_of_managers[0]
        inc.value = 1
        data_csv = pd.read_csv(fileName)
        text_column = tagger(data_csv)
        nm_datasets = math.floor(len(data_csv.index) / cpu_count())
        processes = []
        rows = nm_datasets

        for subprocess in range(0, cpu_count()):
            processes.append(
                Process(
                    target=string_storing,
                    args=(
                        text_column[rows - nm_datasets : rows],
                        string_store,
                        list_of_managers,
                        inc,
                    ),
                )
            )
            rows += nm_datasets
        for process in processes:
            process.start()
        for subprocess in processes:
            subprocess.join()

        print(string_store)

