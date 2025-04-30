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
```.
├── data/
│   ├── lncRNA_feature.pkl   # Pickle file containing all lncRNA features
│   ├── miRNA_feature.pkl    # Pickle file containing all miRNA features
│   ├── lncRNA_idx.csv       # CSV file with all lncRNA names
│   ├── miRNA_idx.csv        # CSV file with all miRNA names
│   └── splits.pkl           # Pickle file containing train/test split data
├── code/
│   ├── model.py             # SGAT-TM model architecture
│   ├── dataset.py           # Data loading and preprocessing
│   ├── funcs.py             # Metric and utility functions
│   ├── main.py              # Training and evaluation script
│   └── layer.py             # Custom layers for the model
├── README.md                # Project overview and instructions
└── requirements.txt         # Python dependencies
```

## Installation
1. Prerequisites
Make sure you have following libraries installed.
* matplotlib==3.10.1
* numpy==2.2.5
* pandas==2.2.3
* scikit_learn==1.6.1
* torch==2.4.1
* torch_geometric==2.6.1
* torch_sparse==0.6.18+pt24cu121

You can install the dependencies by running:
```bash
pip3 install -r requirements.txt
```
2. Run

To train the model, run the main.py script. You can customize training parameters such as the number of epochs, learning rate, and other hyperparameters by passing arguments to the script.

```bash
python3 code/main.py --epoch XXXX --lr XXXX
```
