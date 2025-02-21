#!/bin/bash

if [[ ! -f products.zip ]]; then
   curl -OL http://snap.stanford.edu/ogb/data/nodeproppred/products.zip
fi
if [[ ! -d products/raw ]]; then
   unzip products.zip
fi
if [[ ! -f products/raw/node-feat-with-id.csv ]]; then
   echo "Adding id"
   cd products/raw
   gunzip *
   nl -w1 -v0 -s'|' node-label.csv > node-label-with-id.csv
   nl -w1 -v0 -s'|' node-feat.csv > node-feat-with-id.csv
   cd -
fi
if [[ ! -f products/split/sales_ranking/train-with-split.csv ]]; then
   echo "Adding split"
   cd products/split/sales_ranking/
   if [[ ! -f train.csv ]]; then
     gunzip *
   fi
   sed 's/$/|0/' train.csv > train-with-split.csv
   sed 's/$/|1/' valid.csv > valid-with-split.csv
   sed 's/$/|2/' test.csv > test-with-split.csv
   cd -
fi
