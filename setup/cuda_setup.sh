#!/bin/bash

if ! grep Ubuntu /etc/os-release &> /dev/null; then
  echo "Only Ubuntu is supported."
  exit 1
fi

ubuntu_version=$(cat /etc/os-release | grep VERSION_ID | cut -d '"' -f2 | sed 's/\.//g')

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${ubuntu_version}/x86_64/cuda-ubuntu${ubuntu_version}.pin
sudo mv cuda-ubuntu${ubuntu_version}.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda-repo-ubuntu${ubuntu_version}-12-8-local_12.8.1-570.124.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu${ubuntu_version}-12-8-local_12.8.1-570.124.06-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu${ubuntu_version}-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
sudo apt-get install -y cuda-drivers
