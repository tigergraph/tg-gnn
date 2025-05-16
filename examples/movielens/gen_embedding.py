import pandas as pd
import getpass
import os
from sentence_transformers import SentenceTransformer

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

def run():
    datafile = "ml-latest-small/movies.csv"
    embfile = "ml-latest-small/embedding.csv"
    model_name = "all-mpnet-base-v2"
    if os.path.isfile(datafile):
        df = pd.read_csv(datafile, sep=',')

        model = SentenceTransformer(model_name)
        print(f"Generating embedding with model {model_name}")

        # Insert embedding 8d into the items table
        df['embedding'] = df['title'].apply(lambda t: ",".join(list(map(str, model.encode(t).tolist()[:8]))))
        df.to_csv(embfile, sep='|', header=True, index=False, columns=["movieId", "embedding"])
    else:
        print(f"Error reading file {datafile}!")

# This check ensures that the function is only run when the script is executed directly, not when it's imported as a module.
if __name__ == "__main__":
    run()
