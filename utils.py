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
import numpy as np

from dataclasses import dataclass, field
from typing import Optional
from sklearn.model_selection import train_test_split

from functools import partial
from peft import LoraConfig, TaskType, get_peft_model, get_peft_config


tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
tokenizer.pad_token_id = tokenizer.eos_token_id

#embedding_size = 1024
#projection_size = 6

EMBEDDING_SIZE = 1024
PROJECTION_SIZE = 6

class ProjectionNN(nn.Module):
    def __init__(self):
        super(ProjectionNN, self).__init__()

        # Architecture enhancements
        self.fc1 = nn.Linear(EMBEDDING_SIZE, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)

        self.fc3 = nn.Linear(256, 2048 * PROJECTION_SIZE)
        self.bn3 = nn.BatchNorm1d(2048 * PROJECTION_SIZE) 


    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout1(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = x.view(-1, PROJECTION_SIZE, 2048)
        return x

class ProjectionNNG(nn.Module):
    def __init__(self):
        super(ProjectionNNG, self).__init__()

        # Architecture
        self.fc1 = nn.Linear(embedding_size, 128).cuda()
        self.relu = nn.ReLU().cuda()
        self.fc2 = nn.Linear(128, 2048 * projection_size).cuda()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.view(-1,6,2048)
        return x

class DataSplit():

    def __init__(self, df):
        self.df = df
        self.types = ['vd_', 'vp_', 'vmd_', 'vmp', 'ts_ce_', 'ts_le_', 'ts_pe_', 'n_rad_']
        self.partition = None

    def partitiondata(self, partition):
        self.pkl_list = []
        if partition == 'mortality':

            df_death_small48 = self.df[((self.df['img_length_of_stay'] < 48) & (self.df['death_status'] == 1))]
            df_alive_big48 = self.df[((self.df['img_length_of_stay'] >= 48) & (self.df['death_status'] == 0))]
            df_death_big48 = self.df[((self.df['img_length_of_stay'] >= 48) & (self.df['death_status'] == 1))]

            df_death_small48['y'] = 1
            df_alive_big48['y'] = 0
            df_death_big48['y'] = 0
            self.df = pd.concat([df_death_small48, df_alive_big48, df_death_big48], axis = 0)
            self.df = self.df.drop(['img_id', 'img_charttime', 'img_deltacharttime', 'discharge_location', 'img_length_of_stay', 
                    'death_status'], axis = 1)

        if partition == 'los':

            df_alive_small48 = self.df[((self.df['img_length_of_stay'] < 48) & (self.df['death_status'] == 0))]
            df_alive_big48 = self.df[((self.df['img_length_of_stay'] >= 48) & (self.df['death_status'] == 0))]
            df_death = self.df[(self.df['death_status'] == 1)]

            df_alive_small48['y'] = 1
            df_alive_big48['y'] = 0
            df_death['y'] = 0
            self.df = pd.concat([df_alive_small48, df_alive_big48, df_death], axis = 0)

            self.df = self.df.drop(['img_id', 'img_charttime', 'img_deltacharttime', 'discharge_location', 'img_length_of_stay', 
                    'death_status'], axis = 1)

    def split_data(self, partition, test_size=0.3, validation_size=0.1, random_state=1):

        self.partition = partition

        self.partitiondata(partition)
        pkl_list = self.df['haim_id'].unique().tolist()

        # Split into training and test sets
        train_id, test_id = train_test_split(pkl_list, test_size=test_size, random_state=random_state)

        remaining_data_size = 1.0 - test_size
        validation_size = validation_size*remaining_data_size

        # Further split the training set into training and validation sets
        train_id, validation_id = train_test_split(train_id, test_size=validation_size, random_state=random_state)

        train_idx = self.df[self.df['haim_id'].isin(train_id)]['haim_id'].tolist()
        validation_idx = self.df[self.df['haim_id'].isin(validation_id)]['haim_id'].tolist()
        test_idx = self.df[self.df['haim_id'].isin(test_id)]['haim_id'].tolist()

        self.x_train = self.df[self.df['haim_id'].isin(train_idx)].drop(['y','haim_id'],axis=1)
        self.x_validation = self.df[self.df['haim_id'].isin(validation_idx)].drop(['y','haim_id'],axis=1)
        self.x_test = self.df[self.df['haim_id'].isin(test_idx)].drop(['y','haim_id'],axis=1)

        self.y_train = self.df[self.df['haim_id'].isin(train_idx)]['y']
        self.y_validation = self.df[self.df['haim_id'].isin(validation_idx)]['y']
        self.y_test = self.df[self.df['haim_id'].isin(test_idx)]['y']
        #self.y_train = [' No' if value == 0 else ' Yes' for value in self.df[self.df['haim_id'].isin(train_idx)]['y'].values]
        #self.y_validation = [' No' if value == 0 else ' Yes' for value in self.df[self.df['haim_id'].isin(validation_idx)]['y'].values]
        #self.y_test = [' No' if value == 0 else ' Yes' for value in self.df[self.df['haim_id'].isin(test_idx)]['y'].values]

    def get_type(self, requested):
        train_cols = self.x_train.filter(regex='^'+requested)
        val_cols = self.x_validation.filter(regex='^'+requested)
        test_cols = self.x_test.filter(regex='^'+requested)
        return train_cols, val_cols, test_cols


class CustomDataset(Dataset):
    def __init__(self, vectors, labels):
        self.vectors = vectors
        self.labels = labels
    
    def __len__(self):
        return len(self.vectors)
    
    def __getitem__(self, index):
        vector = torch.tensor(self.vectors[index]).float()
        label = torch.from_numpy(np.array([self.labels[index]]))
        return vector, label.squeeze()

# dataset for generative

class CustomDatasetG(Dataset):
    def __init__(self, embedding, labels):
        self.labels = labels
        self.embedding = embedding
        self.instruction = "###INSTRUCTION: Did this patient stay longer than 48 h? ###MODALITY: "

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        emb = self.embedding[idx]
        inst = self.instruction
        sample = {"Emb": emb, "Class": label, 'Inst': inst}
        return sample

# collate_batch for generative

def collate_batch(batch):
     
    emb_list, classes, instructions = [], [], []
    for thing in batch:
        emb_list.append(thing['Emb'])
        classes.append(tokenizer(thing['Class'], return_tensors="pt"))
        instructions.append(tokenizer(thing['Inst'], return_tensors="pt"))
    text = torch.tensor(emb_list)
    classes = torch.cat([item['input_ids'] for item in classes], dim=0)
    instructions = torch.cat([item['input_ids'] for item in instructions], dim=0)
    return text, instructions, classes

# SECOND APPROACH GEN TRYOUT

def output_to_label(logits):
    probs = torch.softmax(logits, dim=-1)
    predicted_token_id = torch.argmax(probs, dim=-1)
    return predicted_token_id

def left_padding(concatenated_emb, labels, device):
    padded_labels = []
    for concatenated_emb_item, label in zip(concatenated_emb, labels):
        prompt_length = concatenated_emb_item.size(0) - label.size(0)
        padded_label = torch.cat([torch.full((prompt_length,), -100, device=device), label.to(device)])
        padded_labels.append(padded_label)
    padded_labels = torch.stack(padded_labels, dim=0)  # Stack padded labels into a batch tensor
    return padded_labels

def masking(logits, padded_target):
    mask = (padded_target != -100)
    logits_masked = logits[mask]
    targets_masked = padded_target[mask]
    return logits_masked, targets_masked

def train_epoch(model, gemma, optimizer, loss_fn, train_loader, device):
    # Train:
    gemma.train()
    train_loss_batches, train_acc_batches = [], []
    num_batches = len(train_loader)
    embedding_matrix = gemma.get_input_embeddings().weight
    with tqdm(total=num_batches, desc="Training", leave=False) as pbar:
        for batch_index, (mod, inst, label) in enumerate(train_loader, 1):
            mod_embeddings = model(mod.to(device))
            inst_list = [embedding_matrix[token_id] for token_id in inst.to(dtype=torch.long)]
            label_list = [embedding_matrix[token_id] for token_id in label.to(dtype=torch.long)]
            inst_embeddings = torch.stack(inst_list)
            label_embeddings = torch.stack(label_list)
            optimizer.zero_grad()

            conc_emb = torch.cat([inst_embeddings.to(dtype=torch.float16), mod_embeddings, label_embeddings.to(dtype=torch.float16)], dim=1).to(device)
            padded_target = left_padding(conc_emb, label, device)
            output = gemma(inputs_embeds=conc_emb.to(dtype=torch.float16), labels=padded_target)

            logits = output['logits'] #torch.Size([1, 31, 256000])
            logits_masked, targets_masked = masking(logits, padded_target)

            loss = loss_fn(logits_masked.squeeze(0), targets_masked)
            loss.backward()
            optimizer.step()
            train_loss_batches.append(loss.item())

            hard_preds = output_to_label(logits_masked)
            acc_batch_avg = (hard_preds == targets_masked).float().mean().item()
            train_acc_batches.append(acc_batch_avg)

            # Update progress bar every 500 iterations
            if batch_index % 500 == 0 or batch_index == num_batches:
                pbar.update(500 if batch_index + 500 < num_batches else num_batches - batch_index)

    return model, train_loss_batches, train_acc_batches

def validate(model, gemma, loss_fn, val_loader, device):
    val_loss_cum = 0
    val_acc_cum = 0
    gemma.eval()
    num_batches = len(val_loader)
    embedding_matrix = gemma.get_input_embeddings().weight
    with torch.no_grad():
        with tqdm(total=num_batches, desc="Validation", leave=False) as pbar:
            for batch_index, (mod, inst, label) in enumerate(val_loader, 1):
                mod_embeddings = model(mod.to(device))
                inst_list = [embedding_matrix[token_id] for token_id in inst.to(dtype=torch.long)]
                label_list = [embedding_matrix[token_id] for token_id in label.to(dtype=torch.long)]
                inst_embeddings = torch.stack(inst_list)
                label_embeddings = torch.stack(label_list)

                conc_emb = torch.cat([inst_embeddings.to(dtype=torch.float16), mod_embeddings, label_embeddings.to(dtype=torch.float16)], dim=1).to(device)
                padded_target = left_padding(conc_emb, label, device)

                output = gemma(inputs_embeds=conc_emb.to(dtype=torch.float16), labels=padded_target)

                logits = output['logits'] #torch.Size([1, 31, 256000])
                logits_masked, targets_masked = masking(logits, padded_target)


                loss = loss_fn(logits_masked.squeeze(0), targets_masked)
                val_loss_cum += loss.item()
                hard_preds = output_to_label(logits_masked)
                acc_batch_avg = (hard_preds == targets_masked).float().mean().item()
                val_acc_cum += acc_batch_avg

                # Update progress bar every 500 iterations
                if batch_index % 500 == 0 or batch_index == num_batches:
                    pbar.update(500 if batch_index + 500 < num_batches else num_batches - batch_index)
                    
    return val_loss_cum/len(val_loader), val_acc_cum/len(val_loader)

def training_loop(model, gemma, optimizer, loss_fn, train_loader, val_loader, num_epochs):
    print("Starting training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    gemma.to(device)
    #for name, param in gemma.named_parameters():
    #    print(f"{name}: {param.device}")
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    for epoch in range(1, num_epochs+1):
        model, train_loss, train_acc = train_epoch(model, gemma,
                                                   optimizer,
                                                   loss_fn,
                                                   train_loader,
                                                   device)
        val_loss, val_acc = validate(model, gemma, loss_fn, val_loader, device)
        print(f"Epoch {epoch}/{num_epochs}: "
              f"Train loss: {sum(train_loss)/len(train_loss):.3f}, "
              f"Train acc.: {sum(train_acc)/len(train_acc):.3f}, "
              f"Val. loss: {val_loss:.3f}, "
              f"Val. acc.: {val_acc:.3f}")
        train_losses.extend(train_loss)
        train_accs.extend(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    return gemma, train_losses, train_accs, val_losses, val_accs