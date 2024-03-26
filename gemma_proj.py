import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import torch.nn as nn
import torch.optim as optim
from focal_loss.focal_loss import FocalLoss
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from utils import *

# Filepath to embeddings
fname = "/mnt/mimic/data/HAIM/mimic_extras/embeddings.csv"

# YES-TOKEN: 3276
# NO-TOKEN: 956

quantization_config = BitsAndBytesConfig(load_in_4bit=True, 
                                         bnb_4bit_use_double_quant=True,
                                         bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
gemma = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto", quantization_config=quantization_config)

# Read data & extract labels and features
df = pd.read_csv(fname)

###condition_death_small48 = (df['img_length_of_stay'] < 48) & (df['death_status'] == 1)
###condition_alive_big48 = (df['img_length_of_stay'] >= 48) & (df['death_status'] == 0)
###condition_death_big48 = (df['img_length_of_stay'] >= 48) & (df['death_status'] == 1)

###y = [0]*len(df)
###for i, condition in enumerate(condition_death_small48):
###    if condition:
###        y[i] = 1

# Use .loc to avoid SettingWithCopyWarning
#df.loc[condition_death_small48, 'y'] = 1
#df.loc[condition_alive_big48, 'y'] = 0
#df.loc[condition_death_big48, 'y'] = 0
        
###vd_cols = df.filter(regex='^vd_')
###y_col = pd.Series(y, name='y')
###haim_col = df[['haim_id']]
###df = pd.concat([haim_col, vd_cols, y_col], axis=1)

###pkl_list = df['haim_id'].unique().tolist()

# Initial prompt
#input_text = "Based on the following image, output yes if the patient is likely to die and no otherwise."
#input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
#word_embs = gemma.get_input_embeddings().weight[input_ids.input_ids].to("cuda")


# Load train/val sets and create data loaders
batch_size = 8

###x_train, x_val, _, y_train, y_val, _ = data_split(df, pkl_list)
# x_train_small, x_val_small, y_train_small, y_val_small = data_split(df.iloc[:500], pkl_list)
Data = DataSplit(df)
Data.get_data('mortality')

train_set = CustomDataset(Data.x_train['vd_'], Data.y_train)
val_set = CustomDataset(Data.x_validation['vd_'], Data.y_validation)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=5)
val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=5)


# Since the classes are very imbalanced, we weigh the classes to increase performance
w0 = len(Data.y_train)/(2*sum(Data.y_train == 0))
w1 = len(Data.y_train)/(2*sum(Data.y_train == 1))
weights = torch.tensor([w0, w1], dtype = torch.float).to("cuda")


# Setting model and hyperparameters
model = ProjectionNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#loss_fn = nn.CrossEntropyLoss(weight=weights)
loss_fn = FocalLoss(gamma=12)

num_epochs = 5

# Run training
fine_tuned, train_losses, train_accs, val_losses, val_accs = training_loop(model, gemma, optimizer, loss_fn, train_loader, val_loader, num_epochs)


# Save model and results
folder = 'results_focal_12_001'
torch.save(fine_tuned, f"{folder}/finetuned.pth")

with open(f"{folder}/train_losses.pkl", 'wb') as f1:
    pickle.dump(train_losses, f1)

with open(f"{folder}/train_accs.pkl", 'wb') as f2:
    pickle.dump(train_accs, f2)

with open(f"{folder}/val_losses.pkl", 'wb') as f3:
    pickle.dump(val_losses, f3)

with open(f"{folder}/val_accs.pkl", 'wb') as f4:
    pickle.dump(val_accs, f4)

    
