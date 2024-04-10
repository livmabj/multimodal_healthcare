import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import torch.nn as nn
import torch.optim as optim
from focal_loss.focal_loss import FocalLoss
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from peft import LoraConfig, TaskType, get_peft_model, get_peft_config
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from g_utils import *

# Filepath to embeddings
fname = "/mnt/mimic/data/HAIM/mimic_extras/embeddings.csv"

# YES-TOKEN: 3276
# NO-TOKEN: 956

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


# Read data & extract labels and features
df = pd.read_csv(fname)


# Load train/val sets and create data loaders
batch_size = 8

Data = DataSplit(df)
Data.split_data('mortality', random_state=1)

X,V,T = Data.get_type('ts_pe_')
train_set = CustomDatasetG(X.values.tolist(), Data.y_train)
val_set = CustomDatasetG(V.values.tolist(), Data.y_validation)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=collate_batch)

y_train = [0 if value == ' No' else 1 for value in Data.y_train]

#class_0_count = y_train.count(0)
#class_1_count = y_train.count(1)
#print(class_0_count, class_1_count)
# Since the classes are very imbalanced, we weigh the classes to increase performance
#w0 = len(Data.y_train)/(2*class_0_count)
#w1 = len(Data.y_train)/(2*class_1_count)
#weights = torch.tensor([w0, w1], dtype = torch.float).to("cuda")


num_classes = 256000
default_weight = 0.0
yes_class =  [3276, 3553, 6287, 7778] #  These classes are less frequent
no_class = [793, 956, 1294, 1307]  # These classes are more frequent

class_weights = [default_weight] * num_classes
for class_idx in yes_class:
    class_weights[class_idx] = 0.019374344908002687  # Assigning lower weights to less frequent classes
for class_idx in no_class:
    class_weights[class_idx] = 0.0006543663641287499
weights = torch.tensor(class_weights, dtype = torch.float).to("cuda")

# Setting model and hyperparameters
model = torch.load('results_0/model.pth').to('cuda')
optimizer = optim.Adam(gemma.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss(weight=weights)
#loss_fn = FocalLoss(gamma=12)

num_epochs = 10

# Run training
fine_tuned, train_losses, train_accs, val_losses, val_accs, preds = training_loop(model, gemma, optimizer, loss_fn, train_loader, val_loader, num_epochs)

#f1 = metrics.f1_score(Data.y_validation.tolist(), preds)
#auc = metrics.roc_auc_score(Data.y_validation.tolist(), preds)

#conf_matrix = metrics.confusion_matrix(Data.y_validation.tolist(), preds)
#disp = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
#disp.plot()

#fig = plt.gcf()

# Save model and results
folder = 'resultsgemmatest'
torch.save(fine_tuned, f"{folder}/finetuned.pth")

with open(f"{folder}/train_losses.pkl", 'wb') as f1:
    pickle.dump(train_losses, f1)

with open(f"{folder}/train_accs.pkl", 'wb') as f2:
    pickle.dump(train_accs, f2)

with open(f"{folder}/val_losses.pkl", 'wb') as f3:
    pickle.dump(val_losses, f3)

with open(f"{folder}/val_accs.pkl", 'wb') as f4:
    pickle.dump(val_accs, f4)

#fig.savefig(f"{folder}/confusion_matrix.png", dpi=300)

save_df = pd.DataFrame({'labels': Data.y_validation.tolist(), 'predictions': preds})

# Write the DataFrame to a CSV file
df.to_csv(f"{folder}/preds.csv", index=False)