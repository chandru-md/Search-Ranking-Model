import pandas as pd

def load_documents(path="data/raw/documents.csv"):
    df = pd.read_csv(path)

    # combine text fields
    df["text"] = df["title"] + " " + df["description"]

    return df