# CSCI567-Final-Project
# Enhancing Viral Identification in Metagenomics: A Hybrid Deep Learning Model

**Group Members:** Cecilia Liu, Cathy Ma, Shengmiao Huang

## Overview
This repository contains the code and models developed for our project on viral sequence identification in metagenomic data. We replicated the DeepVirFinder model as a benchmark and introduced a hybrid deep learning model that integrates Graph Convolutional Networks (GCNs), Convolutional Neural Networks (CNNs), and Bidirectional Long Short-Term Memory networks (BiLSTMs) to enhance classification performance, especially for short contigs.

## Repository Structure

- **Baseline Model:**
  - `encode.py`: Script for downloading and encoding DNA sequences into numerical representations.
  - `model_training.py`: Script for training the baseline CNN model.
  - `evaluate.py`: Script for generating ROC performance metrics and visualizations.
  - `testing.py`: Script for evaluating and generating classfication reports of the trained model on test datasets.
  - `*.sh`: Bash scripts designed for execution on high-performance computing (HPC) services, facilitating the encoding, training, and testing processes.

- **Hybrid Model:**
  - `hybrid_GCN/`: Directory containing scripts and models for the hybrid GCN-CNN-BiLSTM architecture.
    - `GCN_CNN_BiLSTM.py`: Script for training the hybrid model.
    - `evaluate_gcn.py`: Script for evaluating the hybrid model.
    - `*.sh`: Bash scripts for HPC execution related to the hybrid model.

- **Pre-trained Models:**
  - `model_20ep_0.15/`: Directory containing the trained baseline model and history for 150 bp contigs.
  - `model_20ep_0.3/`: Directory containing the trained baseline model and history for 300 bp contigs.

## References

- Miao Y, Bian J, Dong G, Dai T. DETIRE: a hybrid deep learning model for identifying viral sequences from metagenomes. *Front Microbiol*. 2023 Jun 16;14:1169791.
- Ren J, Song K, Deng C, et al. Identifying viruses from metagenomic data using deep learning. *Quant Biol*. 2020 Mar;8(1):64-77.
- Shang J, Sun Y. CHEER: HierarCHical taxonomic classification for viral mEtagEnomic data via deep leaRning. *Methods*. 2021 May;189:95-103.

For more details, please refer to our [project report](./report.pdf). 
