# Multi-Agent-Motion-Prediction-using-Scene-Graph

## Table of Contents
* [Getting Started](#getting-started)
* [Training & Evaluation](#training--evaluation)
* [Pretrained Models & Results](#pretrained-models--results)
* [Citation](#citation)

* ## Getting Started

**Step 1**: clone this repository

**Step 2**: create a docker environment and install the dependencies:
1. docker image
   '''
   docker build -t qcnet-env:torch2.1.0 -f Dockerfile_qcnet_torch2.1.0 .
   '''
    
2. container
   '''
   docker run --gpus all -it --name ETRI --ipc=host --shm-size=8g -v /mnt/hdd1/jeyoung/kadif:/workspace qcnet-env:torch2.1.0 /bin/bash
   docker run --gpus all -it --name ETRI --ipc=host --shm-size=8g -v /home/kgh/ad:/workspace qcnet-env:torch2.1.0 /bin/bash
   '''

3. Execution
   '''
   docker start ETRI
   docker exec -it ETRI /bin/bash
   '''

**Step 3**: 
