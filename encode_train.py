import pickle
#from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import torch.nn as nn
import torch.optim as optim
from focal_loss.focal_loss import FocalLoss
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler, RandomSampler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from autoencoder_utils import *

# Filepath to embeddings
fname = '/mnt/mimic/data/HAIM/mimic_extras/embeddings.csv'


# Read data & extract labels and features
df = pd.read_csv(fname)


# Load train/val sets and create data loaders
batch_size = 32

Data = DataSplit(df)
Data.split_data('mortality')

X_pe,V_pe,T = Data.get_type('ts_pe_')
X_ce,V_ce,T = Data.get_type('ts_ce_')
X_le,V_le,T = Data.get_type('ts_le_')

torch.manual_seed(42)

concatenated_train = [l1 + l2 + l3 for l1, l2, l3 in zip(X_pe.values.tolist(),X_ce.values.tolist(),X_le.values.tolist())]
concatenated_val = [l1 + l2 + l3 for l1, l2, l3 in zip(V_pe.values.tolist(),V_ce.values.tolist(),V_le.values.tolist())]


#train_set = CustomDataset(X.values.tolist(), Data.y_train.tolist())
#val_set = CustomDataset(V.values.tolist(), Data.y_validation.tolist())
train_set = CustomDataset(concatenated_train, Data.y_train.tolist())
val_set = CustomDataset(concatenated_val, Data.y_validation.tolist())

sampler = RandomSampler(train_set, replacement=False)

train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=5)
val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=5)



# Setting model and hyperparameters
model = AutoEncoder(451,2048)
optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=0.0003) #0.00003
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=7)
loss_fn = nn.MSELoss()

num_epochs = 50

# Run training

fine_tuned, train_losses, val_losses = training_loop(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, scheduler)