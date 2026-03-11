#!/bin/bash

if [[ ! -f ml-latest-small.zip ]]; then
   curl -OL https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
fi
if [[ ! -d ml-latest-small ]]; then
   unzip ml-latest-small.zip
fi
if [[ ! -f ml-latest-small/embedding.csv ]]; then
   echo "Generating ml-latest-small/embedding.csv"
   python3 -m pip install pandas langchain_openai langchain_community sentence_transformers
   python3 ./gen_embedding.py
fi
