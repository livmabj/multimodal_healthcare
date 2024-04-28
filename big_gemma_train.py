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
batch_size = 32

Data = DataSplit(df)
Data.split_data('all')
X, V = Data.get_data()

torch.manual_seed(42)


Data.y_train = Data.y_train.apply(lambda lst: [2 if x == -1 else x for x in lst])
Data.y_val = Data.y_val.apply(lambda lst: [2 if x == -1 else x for x in lst])

train_set = CustomDataset(X.values.tolist(), Data.y_train.tolist())
val_set = CustomDataset(V.values.tolist(), Data.y_val.tolist())

transposed_Y = list(map(list, zip(*Data.y_train.tolist())))

weight_per_class = []

for y in transposed_Y[:-2]:
    y = torch.tensor(y)
    w0 = len(y)/(2*sum(y == 0))
    w1 = len(y)/(2*sum(y == 1))
    w2 = len(y)/(2*sum(y == 2))
    weight_per_class.append(torch.tensor([w0, w1, w2], dtype = torch.float).to("cuda"))

for y in transposed_Y[-2:]:
    y = torch.tensor(y)
    w0 = len(y)/(2*sum(y == 0))
    w1 = len(y)/(2*sum(y == 1))
    weight_per_class.append(torch.tensor([w0, w1], dtype = torch.float).to("cuda"))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=5)
val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=5)


# Setting model and hyperparameters
vd_model = AutoEncoder(1024,2048)
vmd_model = AutoEncoder(1024,2048)

ts_pe_model = AutoEncoder(110,2048)
ts_ce_model = AutoEncoder(99,2048)
ts_le_model = AutoEncoder(242,2048)

n_rad_model = AutoEncoder(768,2048)
models = [vd_model, vmd_model, ts_pe_model, ts_ce_model, ts_le_model, n_rad_model]

vd_optimizer = optim.Adam(vd_model.parameters(), lr=0.0005, weight_decay=0.0003)
vmd_optimizer = optim.Adam(vmd_model.parameters(), lr=0.0005, weight_decay=0.0003)

ts_pe_optimizer = optim.Adam(ts_pe_model.parameters(), lr=0.0005, weight_decay=0.0003)
ts_ce_optimizer = optim.Adam(ts_ce_model.parameters(), lr=0.0005, weight_decay=0.0003)
ts_le_optimizer = optim.Adam(ts_le_model.parameters(), lr=0.0005, weight_decay=0.0003)

n_rad_optimizer = optim.Adam(n_rad_model.parameters(), lr=0.0005, weight_decay=0.0003)
optimizers = [vd_optimizer, vmd_optimizer, ts_pe_optimizer, ts_ce_optimizer, ts_le_optimizer, n_rad_optimizer]

loss_mse = nn.MSELoss()
loss_fns = []
for weight in weight_per_class:
    loss_fns.append(nn.CrossEntropyLoss(weight=weight))

num_epochs = 50
beta = 0.1

# Run training

fine_tuned, train_losses, val_losses = training_loop(models, optimizers, loss_mse, loss_fns, train_loader, val_loader, num_epochs, gemma, beta)