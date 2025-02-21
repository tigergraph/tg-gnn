import pandas as pd
import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

def run():
    datafile = "ml-latest-small/movies.csv"
    embfile = "ml-latest-small/embedding.csv"
    if os.path.isfile(datafile):
        df = pd.read_csv(datafile, sep=',')

        embeddings = OpenAIEmbeddings()

        # Insert embedding 8d into the items table
        df['embedding'] = df['title'].apply(lambda t: embeddings.embed_query(t))
        df['embedding'] = df['embedding'].apply(lambda t: ",".join(t.replace("[", "").replace("]","").split(",")[:8]))
        df.to_csv(embfile, sep='|', header=True, index=False, columns=["movieId", "embedding"])
    else:
        print(f"Error reading file {datafile}!")

# This check ensures that the function is only run when the script is executed directly, not when it's imported as a module.
if __name__ == "__main__":
    run()
