"""
main.py — Outline for training and testing the SGAT-TM model.
Note: This version does not include full implementation or dataset logic.
"""

import argparse
import torch
import torch.optim as optim

from dataset import Dataset       # Assumes a dataset loader class
from model import Model           # Model architecture (structure only)
from funcs import metrics_calc    # Evaluation metric function

def train_model(e, indexes, labels):
    print(f'-------- Epoch {e + 1} --------')
    model.train()
    optimizer.zero_grad()
    _, loss = model(indexes, labels)
    loss.backward()
    optimizer.step()
    print('Loss:', loss.item())

@torch.no_grad()
def test_model(indexes, labels):
    output, _ = model(indexes, labels)
    y_true_test = labels.to('cpu').numpy()
    y_score_test = output.flatten().tolist()
    metrics_calc(y_true_test, y_score_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gl', type=int, default=4, help='Number of GAT layers')
    parser.add_argument('--es', type=int, default=256, help='Embedding size')
    parser.add_argument('--sah', type=int, default=2, help='Self-attention heads')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--epoch', type=int, default=500, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Dataset()
    train_indexes, train_labels = ...  # Placeholder
    test_indexes, test_labels = ...    # Placeholder

    model = Model(dataset, args.gl, args.es, args.sah).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    for epoch in range(args.epoch):
        train_model(epoch, train_indexes, train_labels)

    print('Test results:')
    test_model(test_indexes, test_labels)
