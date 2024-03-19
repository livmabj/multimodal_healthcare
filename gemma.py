# ALL THE NECESSARY IMPORTS

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import pickle
from utils import *

from dataclasses import dataclass, field
from typing import Optional
from sklearn.model_selection import train_test_split

from functools import partial
from peft import LoraConfig, TaskType, get_peft_model, get_peft_config
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Filepath to embeddings
fname = "/mnt/mimic/data/HAIM/mimic_extras/embeddings.csv"

# LoRA parameter efficient fine-tuning
# Parameters are freezed and small submodules with low-rank matrices ar inserted at the target layers.
# initialization of model
quantization_config = BitsAndBytesConfig(load_in_4bit=True, 
                                         bnb_4bit_use_double_quant=True,
                                         bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
tokenizer.pad_token_id = tokenizer.eos_token_id
gemma = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto", quantization_config=quantization_config)
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
    lora_alpha=16,
    lora_dropout=0.1
)

gemma = get_peft_model(gemma, lora_config)


# Get the data

df = pd.read_csv(fname)
condition_death_small48 = (df['img_length_of_stay'] < 48) & (df['death_status'] == 1)
condition_alive_big48 = (df['img_length_of_stay'] >= 48) & (df['death_status'] == 0)
condition_death_big48 = (df['img_length_of_stay'] >= 48) & (df['death_status'] == 1)


y = [0]*len(df)
for i, condition in enumerate(condition_death_small48):
    if condition:
        y[i] = 1

vd_cols = df.filter(regex='^vd_')
y_col = pd.Series(y, name='y')
haim_col = df[['haim_id']]
df = pd.concat([haim_col, vd_cols, y_col], axis=1)

pkl_list = df['haim_id'].unique().tolist()


x_train, x_val, x_test, y_train, y_val, y_test, = data_split(df, pkl_list)

# Create dataset and dataloader
bsz = 8

Trainset = CustomDataset(x_train.tolist(), y_train)
Valset = CustomDataset(x_val.tolist(), y_val)
Testset = CustomDataset(x_test.tolist(), y_test)

Train_loader = DataLoader(Trainset, batch_size=bsz, collate_fn=collate_batch)
Val_loader = DataLoader(Valset, batch_size=bsz, collate_fn=collate_batch)
Test_loader = DataLoader(Testset, batch_size=bsz, collate_fn=collate_batch)

# Run training

model = ProjectionNN()
optimizer = torch.optim.Adam(gemma.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()
num_epochs = 5
fine_tuned, train_losses, train_accs, val_losses, val_accs = training_loop(model, gemma, optimizer, loss_fn, Train_loader, Val_loader, num_epochs)

torch.save(fine_tuned, 'finetuned.pth')

with open('train_losses.pkl', 'wb') as f1:
    pickle.dump(train_losses, f1)

with open('train_accs.pkl', 'wb') as f2:
    pickle.dump(train_accs, f2)

with open('val_losses.pkl', 'wb') as f3:
    pickle.dump(val_losses, f3)

with open('val_accs.pkl', 'wb') as f4:
    pickle.dump(val_accs, f4)