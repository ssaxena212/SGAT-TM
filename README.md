# SGAT-TM
Stacked Graph Attention Network with Temporal Modeling for lncRNA-miRNA Association Network

## Project Overview
The project involves the development and application of SGAT-TM to extract and analyze features from Long Non-Coding RNAs (lncRNAs) and micro RNAs (miRNAs). The objective is to train and test machine learning models using biological features to understand lncRNA-miRNA behavior and their potential roles in biological processes.

## Data Description
The dataset used in this project is stored in a data directory and includes lncRNA and miRNA feautures for training and testing. These datasets contain arrays where each sample corresponds to different features. 

## Code Description
The code used in this project is stored in a code directory and includes:
* main.py: Entry point to train and evaluate the SGAT-TM model with customizable hyperparameters.
* model.py: Defines the SGAT-TM architecture combining self-attention, graph attention layers, GRU cell and MLP module.
* dataset.py: Loads .pkl data and prepares it for PyTorch models.
* funcs.py: Functions for metrics calculation and optional visualizations.

## File Structure
The file structure of the repository is as follows:

. \
├── data/ \
│   └── lncRNA_feature.pkl  # The pickle file containing all lncRNA features. \
│   └── miRNA_feature.pkl   # The pickle file containing all miRNA features.  \
│   └── lncRNA_idx.csv      # The .csv file containing all lncRNA names.  \
│   └── miRNA_idx.csv       # The .csv file containing all miRNA names.  \
│   └── splits.pkl          # The pickle file containing all association data.  \
├── code/ \
│   ├── model.py            # SGAT-TM model architecture. \
│   ├── dataset.py          # Data handling and preprocessing code. \
│   ├── funcs.py            # Functions (metrics, plots, etc.). \
│   └── main.py             # Main script to train and evaluate the model. \
│   └── layer.py            # custom network layers used in the SGAT-TM architecture. \
├── README.md               # This file \
└── requirements.txt     # Python dependencies for the project \
