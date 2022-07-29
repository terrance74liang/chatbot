import pandas as pd
import sklearn
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import numpy as np
import pickle
from scipy.sparse.csgraph import laplacian
import math
from sklearn.preprocessing import StandardScaler


def pickle_reader(pickle_file_name):
    with open(pickle_file_name, "rb") as pickler:
        data = pickle.load(pickler)
        expanded_array = [y for x in data for y in x]
        array_of_vectors = np.array([x[1] for x in expanded_array])
        array_of_words = np.array([x[0] for x in expanded_array])
        return array_of_words, array_of_vectors


# takes the vectors as data
def cluster_optimizer(data):
    data = StandardScaler().fit_transform(X=data)
    length = int(math.sqrt(math.sqrt(len(data))))
    kngraph = kneighbors_graph(
        data,
        n_neighbors=length,
        mode="connectivity",
        metric="euclidean",
    ).toarray()
    laplacian_graph = laplacian(kngraph, use_out_degree=False)
    eigenvalues, eigenvectors = np.linalg.eig(laplacian_graph)
    count = 0
    for x in list(eigenvalues):
        if x == 0:
            count += 1
        else:
            break
    eigenvectors = len(set(eigenvectors[0, count]))
    return eigenvectors


def spec_cluster(vectors, words, num):
    num = max(num, int(math.sqrt(len(vectors))))
    sc = SpectralClustering(
        n_clusters=num, affinity="nearest_neighbors", assign_labels="cluster_qr"
    )
    cluster_labels = sc.fit_predict(vectors)
    word_to_label = pd.DataFrame(words, columns=["words"])
    word_to_label["labels"] = cluster_labels
    return word_to_label
