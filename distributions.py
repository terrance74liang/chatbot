from multiprocessing import Pool, Manager
import pandas as pd
import numpy as np
import string


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
    combined = list(map(lambda x, y: (x, y), data, count))
    combined.sort(key=lambda x: x[1], reverse=True)
    for x, y in zip(combined, range(0, len(combined))):
        if len(str(x[0]).strip()) <= 4 or str(x[0]) in list(string.punctuation):
            continue
        else:
            shared.append((data[y], num))
            break


def parallel_distributions(num_process, data):
    with Pool(processes=num_process) as pool:
        with Manager() as mana:
            max_label = max(list(data["labels"])) + 1
            collection = mana.list()
            for num in range(0, max_label):
                pool.apply(
                    is_slice, (data[data["labels"] == num], num, collection,),
                )

            pool.close()
            pool.join()
            return list(collection)
