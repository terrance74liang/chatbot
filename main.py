import numpy as np
import pandas as pd

# from database import csv_to_json
from distributions import parallel_distributions
from clustering import pickle_reader, cluster_optimizer, spec_cluster

twitter_file = (
    r"C:\Users\crapg\OneDrive\Documents\datasets\twitter customer support\sample.csv"
)

if __name__ == "__main__":
    # DATASET PREPARATION

    # name = csv_to_json(twitter_file)
    # words, vectors, pos = pickle_reader("list_of_vectors.pickle")
    # n_clusters = cluster_optimizer(vectors)
    # labels_and_words = spec_cluster(vectors, words, n_clusters, pos)
    # labels_and_words.to_excel("testfile.xlsx", index=False)
    data = pd.read_excel("testfile.xlsx")
    list_n_labels = parallel_distributions(
        num_process=max(list(data["labels"])) + 1, data=data
    )

