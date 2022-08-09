import multiprocessing
import pandas as pd
import spacy as sp
import numpy as np
from multiprocessing import Process, Manager
from psutil import cpu_count, Process, cpu_times
import psutil
import math
import os
import pickle

# add r python library for punctuation redo

nlp = sp.load("en_core_web_sm")

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
def string_storing(text, string_st, list_man, incr, num):
    psutil.Process(pid=os.getpid()).cpu_affinity(cpus=[num])
    text = text.reset_index(drop=True)
    for i in range(1, len(text.index)):
        doc = nlp(text[i - 1])
        for token in doc:
            if (token.text in [x.text for x in doc.ents]) or (
                token.pos_.upper() in ["SYM", "X", "NUM"]
            ):
                continue
            elif token.pos_.upper() == "PUNCT" and (
                token.text not in string_st.get("PUNCT")
            ):
                string_st.get("PUNCT").append((token.text, token.pos_, token.vector))
            elif token.text[0].lower() not in string_st.keys():
                string_st[token.text[0].lower()] = list_man[incr.value]
                string_st.get(token.text[0].lower()).append(
                    (token.text.lower(), token.pos_, token.vector)
                )
                incr.value += 1
            elif token.text.lower() not in [
                x[0] for x in string_st.get(token.text[0].lower())
            ]:
                string_st.get(token.text[0].lower()).append(
                    (token.text.lower(), token.pos_, token.vector)
                )
            else:
                continue


def csv_to_json(fileName):
    inc = multiprocessing.Value("i", 0)
    with multiprocessing.Manager() as manager_:
        list_of_managers = [manager_.list() for i in range(0, 5000)]
        string_store = manager_.dict()
        string_store["PUNCT"] = list_of_managers[0]
        inc.value = 1
        data_csv = pd.read_csv(fileName)
        text_column = tagger(data_csv)
        nm_datasets = math.floor(len(data_csv.index) / cpu_count())
        processes = []
        rows = nm_datasets
        for subprocess in range(1, cpu_count() + 1):
            processes.append(
                multiprocessing.Process(
                    target=string_storing,
                    args=(
                        text_column[rows - nm_datasets : rows],
                        string_store,
                        list_of_managers,
                        inc,
                        subprocess - 1,
                    ),
                )
            )
            rows += nm_datasets

        for process in processes:
            process.start()

        for subprocess in processes:
            subprocess.join()

        normal_list = [list(sublist) for sublist in list_of_managers[0 : inc.value]]

        with open("list_of_vectors.pickle", "wb") as pickler:
            pickle.dump(obj=normal_list, file=pickler)
            return pickler.name


def pos_extractor(sentence):
    return [x.pos_ for x in nlp(sentence)]

