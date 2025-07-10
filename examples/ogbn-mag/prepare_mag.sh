#!/bin/bash

mkdir -p /tmp/ogbn-mag
sudo chmod 777 /tmp/ogbn-mag
cd /tmp/ogbn-mag

if [[ ! -f mag.zip ]]; then
   curl -OL http://snap.stanford.edu/ogb/data/nodeproppred/mag.zip
fi
if [[ ! -d mag/raw ]]; then
   unzip mag.zip
fi
if [[ ! -f mag/raw/node-feat/paper/node-feat-with-id.csv ]]; then
   echo "Adding id to node-feat"
   cd mag/raw/node-feat/paper/
   gunzip *
   nl -w1 -v0 -s'|' node-feat.csv > node-feat-with-id.csv
   cd -
fi
if [[ ! -f mag/raw/node-label/paper/node-label-with-id.csv ]]; then
   echo "Adding id to node-label"
   cd mag/raw/node-label/paper/
   gunzip *
   nl -w1 -v0 -s'|' node-label.csv > node-label-with-id.csv
   cd -
fi
if [[ ! -f mag/raw/relations/author___writes___paper/edge.csv ]]; then
   echo "Extracting edge author_writes_paper"
   cd mag/raw/relations/author___writes___paper/
   gunzip *
   cd -
fi
if [[ ! -f mag/raw/relations/paper___cites___paper/edge.csv ]]; then
   echo "Extracting edge paper_cites_paper"
   cd mag/raw/relations/paper___cites___paper/
   gunzip *
   cd -
fi
if [[ ! -f mag/mapping/paper_entidx2name.csv ]]; then
   echo "Extracting node names"
   cd mag/mapping/
   gunzip * || true
   cd -
fi
if [[ ! -f mag/split/time/parper/train-with-split.csv ]]; then
   echo "Adding split"
   cd mag/split/time/paper/
   gunzip *
   sed 's/$/|0/' train.csv > train-with-split.csv
   sed 's/$/|1/' valid.csv > valid-with-split.csv
   sed 's/$/|2/' test.csv > test-with-split.csv
   cd -
fi
