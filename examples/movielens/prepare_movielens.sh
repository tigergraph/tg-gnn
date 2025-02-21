#!/bin/bash

if [[ ! -f ml-latest-small.zip ]]; then
   curl -OL https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
fi
if [[ ! -d ml-latest-small ]]; then
   unzip ml-latest-small.zip
fi
if [[ ! -f ml-latest-small/embedding.csv ]]; then
   echo "Generating ml-latest-small/embedding.csv"
   python3 -m pip install pandas
   python3 -m pip install langchain_openai
   python3 ./add_embedding.py
fi
if [[ ! -f ml-latest-small/ratings-with-split.csv ]]; then
   echo "Generating ml-latest-small/ratings-with-split.csv"
   while IFS= read -r line; do
      line="${line%$'\r'}"
      if [[ "$line" =~ ^[^0-9] ]]; then
	 echo "${line},split" > ml-latest-small/ratings-with-split.csv  
      else
	 echo "${line},$((RANDOM%3))" >> ml-latest-small/ratings-with-split.csv
      fi
   done < ml-latest-small/ratings.csv
fi
