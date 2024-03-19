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

from dataclasses import dataclass, field
from typing import Optional
from sklearn.model_selection import train_test_split

from functools import partial
from peft import LoraConfig, TaskType, get_peft_model, get_peft_config


tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
tokenizer.pad_token_id = tokenizer.eos_token_id

embedding_size = 1024
projection_size = 6

class ProjectionNN(nn.Module):
    def __init__(self):
        super(ProjectionNN, self).__init__()

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

def data_split(df, pkl_list, test_size=0.3, validation_size=0.1, random_state=None):
    # Split into training and test sets
    train_set, test_set = train_test_split(pkl_list, test_size=test_size, random_state=random_state)

    # Further split the training set into training and validation sets
    train_set, validation_set = train_test_split(train_set, test_size=validation_size, random_state=random_state)

    train_idx = df[df['haim_id'].isin(train_set)]['haim_id'].tolist()
    validation_idx = df[df['haim_id'].isin(validation_set)]['haim_id'].tolist()
    test_idx = df[df['haim_id'].isin(test_set)]['haim_id'].tolist()

    x_train = df[df['haim_id'].isin(train_idx)].drop(['haim_id', 'y'], axis=1).values
    x_validation = df[df['haim_id'].isin(validation_idx)].drop(['haim_id', 'y'], axis=1).values
    x_test = df[df['haim_id'].isin(test_idx)].drop(['haim_id', 'y'], axis=1).values

    y_train = df[df['haim_id'].isin(train_idx)]['y'].values
    y_train = [' ###ANSWER: No' if value == 0 else ' ###ANSWER: Yes' for value in y_train]
    y_validation = df[df['haim_id'].isin(validation_idx)]['y'].values
    y_validation = [' ###ANSWER: No' if value == 0 else ' ###ANSWER: Yes' for value in y_validation]
    y_test = df[df['haim_id'].isin(test_idx)]['y'].values
    y_test = [' ###ANSWER: No' if value == 0 else ' ###ANSWER: Yes' for value in y_test]

    return x_train, x_validation, x_test, y_train, y_validation, y_test

# dataset for generative

class CustomDataset(Dataset):
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
        #print(sample)
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