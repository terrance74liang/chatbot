from multiprocessing import Pool, Manager
import pandas as pd
import numpy as np


def is_slice(data, num, shared):
    data = list(data["words"])
    count = [0] * len(data)
    length = iter(range(0, len(data)))
    pointer = np.nan
    for x in data:
        if len(x) <= 1:
            pointer = next(length, None)
        if pointer == None:
            return None
    for words in range(0, len(data)):
        for comp in range(0, len(data)):
            if words == comp:
                continue
            if data[words] in data[comp]:
                count[words] += 1
    # condition = True
    # word_index = None
    # while condition:
    #     for word, count_freq, index in zip(data, count, range(0, len(data))):
    #         if count_freq == max(count) and len(word) <= 2:
    #             count[index] = 0
    #     for word, count_freq, index in zip(data, count, range(0, len(data))):
    #         if count_freq == max(count) and len(word) <= 2:
    #             break
    #         if index == len(data):
    #             condition = False

    shared.append((data[word_index], num))


def parallel_distributions(num_process, data):
    with Pool(processes=num_process) as pool:
        with Manager() as mana:
            max_label = max(list(data["labels"])) + 1
            collection = mana.list()
            for num in range(0, max_label):
                pool.apply(
                    is_slice,
                    (
                        data[data["labels"] == num],
                        num,
                        collection,
                    ),
                )

            pool.close()
            pool.join()
            return list(collection)
