"""
Sample Dataset loader class for lncRNA–miRNA interaction prediction.

This is a placeholder version. Actual data files and full implementation
will be released post-publication.
"""

import pickle
import numpy as np
import pandas as pd
import torch
import torch_geometric.utils as pyg_utils

class Dataset:
    def __init__(self):
        # Sample dummy initialization instead of actual file loading
        self.lncRNA_feature = np.random.rand(100, 128)  # Placeholder shape
        self.miRNA_feature = np.random.rand(50, 128)    # Placeholder shape
        self.num_lncRNAs = self.lncRNA_feature.shape[0]
        self.num_miRNAs = self.miRNA_feature.shape[0]
        self.num_nodes = self.num_lncRNAs + self.num_miRNAs

        # Dummy splits for example purposes
        self.train_pos = np.array([[0, 1], [2, 3]])
        self.train_neg = np.array([[4, 5], [6, 7]])
        self.test_pos = np.array([[8, 9]])
        self.test_neg = np.array([[10, 11]])

        edge_index = torch.tensor(self.train_pos.T, dtype=torch.long)
        self.edge_index = pyg_utils.to_undirected(edge_index)

    def train_data(self):
        interactions = np.concatenate([self.train_pos, self.train_neg])
        labels = np.concatenate([np.ones(len(self.train_pos)), np.zeros(len(self.train_neg))])
        return interactions, labels

    def test_data(self):
        interactions = np.concatenate([self.test_pos, self.test_neg])
        labels = np.concatenate([np.ones(len(self.test_pos)), np.zeros(len(self.test_neg))])
        return interactions, labels
