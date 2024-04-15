import pickle
#from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import torch.nn as nn
import torch.optim as optim
from focal_loss.focal_loss import FocalLoss
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from mlp_utils import *

# Filepath to embeddings
#fname = "/mnt/mimic/data/HAIM/mimic_extras/embeddings.csv"
fname = '/mnt/mimic/data/HAIM/mimic_extras/embeddings.csv'

# YES-TOKEN: 3276
# NO-TOKEN: 956
"""
quantization_config = BitsAndBytesConfig(load_in_4bit=True, 
                                         bnb_4bit_use_double_quant=True,
                                         bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
gemma = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto", quantization_config=quantization_config)
"""
# Read data & extract labels and features
df = pd.read_csv(fname)


# Load train/val sets and create data loaders
batch_size = 32

Data = DataSplit(df)
Data.split_data('mortality')

X,V,T = Data.get_type('vd_')

# Since the classes are very imbalanced, we weigh the classes to increase performance
w0 = len(Data.y_train)/(2*sum(Data.y_train == 0))
w1 = len(Data.y_train)/(2*sum(Data.y_train == 1))
weights = torch.tensor([w0, w1], dtype = torch.float).to("cuda")

class_sample_count = np.array([len(np.where(Data.y_train == t)[0]) for t in np.unique(Data.y_train)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in Data.y_train])

samples_weight = torch.from_numpy(samples_weight)
samples_weight = samples_weight.double()

sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

train_set = CustomDataset(X.values.tolist(), Data.y_train.tolist())
val_set = CustomDataset(V.values.tolist(), Data.y_validation.tolist())

train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=5)
val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=5)



# Setting model and hyperparameters
model = ProjectionNN()
optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=3e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=7)
loss_fn = nn.CrossEntropyLoss(weight=weights)
#loss_fn = FocalLoss(gamma=1)

num_epochs = 50

# Run training
fine_tuned, train_losses, train_accs, val_losses, val_accs = training_loop(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, scheduler)
