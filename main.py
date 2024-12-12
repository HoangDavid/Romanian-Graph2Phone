from g2p_seq2seq.g2p import G2P
import json
import torch as tc
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Note to self: use permute or tranpose instead, reshape and view mess up the data !!
def main():
    model = G2P(hidden_size=128, lr=0.007)
    model.train_model(batch_size=16, epochs=20, log_every=100)
    # model.test()

if __name__ == "__main__":
    main()