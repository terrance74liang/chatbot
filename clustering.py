import pandas as pd
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import numpy as np
import pickle
from scipy.sparse.csgraph import laplacian
import math


def pickle_reader(pickle_file_name):
    with open(pickle_file_name, "rb") as pickler:
        data = pickle.load(pickler)
        expanded_array = [y for x in data for y in x]
        array_of_vectors = np.array([x[1] for x in expanded_array])
        array_of_words = np.array([x[0] for x in expanded_array])
        return array_of_words, array_of_vectors


# takes the vectors as data
def cluster_optimizer(data):
    length = 4
    kngraph = kneighbors_graph(
        data, n_neighbors=length, mode="connectivity", metric="euclidean",
    ).toarray()
    laplacian_graph = laplacian(kngraph, use_out_degree=False)
    eigenvalues, eigenvectors = np.linalg.eig(laplacian_graph)
    count = 0
    for x in list(eigenvalues):
        if x == 0:
            count += 1
        else:
            break
    count2 = 0
    index1 = iter(range(0, count))
    index2 = iter(range(0, count))
    list_of_eigenvectors = [list(x) for x in eigenvectors]
    for x in list_of_eigenvectors[0:count]:
        for y in list_of_eigenvectors[0:count]:
            if index1 == index2:
                next(index2)
                continue
            if x == y:
                count2 += 1
                next(index2)

        next(index1)

    return count2


def spec_cluster(vectors, words, num):
    sc = SpectralClustering(
        n_clusters=num, affinity="nearest_neighbors", assign_labels="cluster_qr"
    )
    cluster_labels = sc.fit_predict(vectors)
    word_to_label = pd.DataFrame(words, columns=["words"])
    word_to_label["labels"] = cluster_labels
    return word_to_label

