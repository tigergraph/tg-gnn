#!/bin/bash

dns_name=$1
node_type=${2:-gpu}
tg_nodes=${3:-0}

if [[ -z "$dns_name" ]]; then
  echo "EFS dns name is required."
  exit 1
fi

sudo apt-get install -y nfs-common

sudo mkdir /tg_export || true
if [[ "$node_type" == "gpu" ]]; then
  sudo mount -t nfs4 -o vers=4.1 ${dns_name}:/ /tg_export
  for i in $(seq 1 $tg_nodes); do
    sudo mkdir /tg_export/m${i}
  done
else
  sudo mount -t nfs4 -o vers=4.1 ${dns_name}:/m${tg_nodes} /tg_export
fi
sudo chmod -R 777 /tg_export
