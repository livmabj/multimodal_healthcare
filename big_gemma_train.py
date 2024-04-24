import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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
from big_utils import *

# Filepath to embeddings
fname = '/mnt/mimic/data/HAIM/mimic_extras/embeddings.csv'

quantization_config = BitsAndBytesConfig(load_in_4bit=True, 
                                         bnb_4bit_use_double_quant=True,
                                         bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
gemma = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto", quantization_config=quantization_config)

# Read data & extract labels and features
df = pd.read_csv(fname)


# Load train/val sets and create data loaders
batch_size = 8

Data = DataSplit(df)
Data.split_data('mortality')

X, V, T = Data.get_type('vd_')

torch.manual_seed(42)

train_set = CustomDataset(X.values.tolist(), Data.y_train.tolist())
val_set = CustomDataset(V.values.tolist(), Data.y_validation.tolist())

w0 = len(Data.y_train)/(2*sum(Data.y_train == 0))
w1 = len(Data.y_train)/(2*sum(Data.y_train == 1))
weights = torch.tensor([w0, w1], dtype = torch.float).to("cuda")

sampler = RandomSampler(train_set, replacement=False)

train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=5)
val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=5)



# Setting model and hyperparameters
vd_model = AutoEncoder(1024,2048)
ts_model = AutoEncoder(451,2048)
n_rad_model = AutoEncoder(768,2048)
vd_optimizer = optim.Adam(vd_model.parameters(), lr=0.0005, weight_decay=0.0003)
ts_optimizer = optim.Adam(vd_model.parameters(), lr=0.0005, weight_decay=0.0003)
n_rad_optimizer = optim.Adam(vd_model.parameters(), lr=0.003, weight_decay=0.003)
optimizers = [vd_optimizer, ts_optimizer, n_rad_optimizer]
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=7)
loss_mse = nn.MSELoss()
loss_bce = nn.CrossEntropyLoss(weight=weights)

num_epochs = 50
beta = 0.1


models = [vd_model]
optimizers = [vd_optimizer]
# Run training

fine_tuned, train_losses, val_losses = training_loop(vd_model, ts_model, n_rad_model, optimizers, loss_mse, loss_bce, train_loader, val_loader, num_epochs, gemma, beta)