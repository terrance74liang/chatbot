from database import csv_to_json
from psutil import cpu_count

twitter_file = (
    r"C:\Users\crapg\OneDrive\Documents\datasets\twitter customer support\sample.csv"
)
if __name__ == "__main__":
    csv_to_json(twitter_file)

