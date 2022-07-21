from database import csv_to_json
from clustering import *

twitter_file = (
    r"C:\Users\crapg\OneDrive\Documents\datasets\twitter customer support\sample.csv"
)

if __name__ == "__main__":
    name = csv_to_json(twitter_file)
    words, vectors = pickle_reader(name)
    n_clusters = cluster_optimizer(vectors)
    labels_and_words = spec_cluster(vectors, words, n_clusters)
