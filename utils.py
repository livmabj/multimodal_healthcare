import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from focal_loss.focal_loss import FocalLoss
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

EMBEDDING_SIZE = 1024
PROJECTION_SIZE = 6


class ProjectionNN(nn.Module):
    def __init__(self):
        super(ProjectionNN, self).__init__()

        # Architecture
        self.fc1 = nn.Linear(EMBEDDING_SIZE, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2048 * PROJECTION_SIZE)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.view(-1,6,2048)
        return x


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
    

def data_split(df, pkl_list):
    train_id, val_id = train_test_split(pkl_list, test_size=0.3, random_state=42)
    val_id, test_id = train_test_split(val_id, test_size=0.25, random_state=42)

    train_idx = df[df['haim_id'].isin(train_id)]['haim_id'].tolist()
    val_idx = df[df['haim_id'].isin(val_id)]['haim_id'].tolist()
    test_idx = df[df['haim_id'].isin(test_id)]['haim_id'].tolist()

    x_train = df[df['haim_id'].isin(train_idx)].drop(['haim_id','y'],axis=1).values
    x_val = df[df['haim_id'].isin(val_idx)].drop(['haim_id','y'],axis=1).values
    x_test = df[df['haim_id'].isin(test_idx)].drop(['haim_id','y'],axis=1).values

    y_train = df[df['haim_id'].isin(train_idx)]['y'].values
    y_val = df[df['haim_id'].isin(val_idx)]['y'].values
    y_test = df[df['haim_id'].isin(test_idx)]['y'].values

    return x_train, x_val, x_test, y_train, y_val, y_test


def custom_output(emb, gemma):
    outputs = gemma(inputs_embeds=emb)
    noyes = [956, 3276]
    logits = outputs['logits']
    logits = logits[:,-6:,noyes].mean(dim=1)
    return logits


def output_to_label(logits):
    probs = torch.softmax(logits, dim=-1)
    predicted_token_id = torch.argmax(probs, dim=-1)
    return predicted_token_id

    
def train_epoch(model, gemma, optimizer, loss_fn, train_loader, device, word_embs):
    # Train:
    model.train()
    train_loss_batches, train_acc_batches = [], []
    for batch_index, (x, y) in enumerate(train_loader, 1):
        inputs, labels = x.to(device), y.to(device)

        optimizer.zero_grad()

        emb = model.forward(inputs)
        word_embs_extended = word_embs.repeat(len(inputs),1,1).detach()

        concatted = torch.cat((word_embs_extended, emb), dim=1).to(torch.float16)
        logits = custom_output(concatted, gemma).float()
        
        loss = loss_fn(logits, labels.long())
        loss.backward()
        optimizer.step()
        train_loss_batches.append(loss.item())

        hard_preds = output_to_label(logits)
        acc_batch_avg = (hard_preds == labels).float().mean().item()
        train_acc_batches.append(acc_batch_avg)

    return model, train_loss_batches, train_acc_batches


def validate(model, gemma, loss_fn, val_loader, device, word_embs):
    val_loss_cum = 0
    val_acc_cum = 0
    model.eval()
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(val_loader, 1):
            inputs, labels = x.to(device), y.to(device)

            emb = model.forward(inputs)
            word_embs_extended = word_embs.repeat(len(inputs),1,1).detach()

            concatted = torch.cat((word_embs_extended, emb), dim=1).to(torch.float16)
            logits = custom_output(concatted, gemma)

            batch_loss = loss_fn(logits, labels.long())
            val_loss_cum += batch_loss.item()
            hard_preds = output_to_label(logits)
            acc_batch_avg = (hard_preds == labels).float().mean().item()
            val_acc_cum += acc_batch_avg
    return val_loss_cum/len(val_loader), val_acc_cum/len(val_loader)


def training_loop(model, gemma, optimizer, loss_fn, train_loader, val_loader, num_epochs, word_embs):
    print("Starting training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    for epoch in range(1, num_epochs+1):
        model, train_loss, train_acc = train_epoch(model,
                                                   gemma,
                                                   optimizer,
                                                   loss_fn,
                                                   train_loader,
                                                   device,
                                                   word_embs)
        val_loss, val_acc = validate(model, gemma, loss_fn, val_loader, device, word_embs)
        print(f"Epoch {epoch}/{num_epochs}: "
              f"Train loss: {sum(train_loss)/len(train_loss):.3f}, "
              f"Train acc.: {sum(train_acc)/len(train_acc):.3f}, "
              f"Val. loss: {val_loss:.3f}, "
              f"Val. acc.: {val_acc:.3f}")
        train_losses.extend(train_loss)
        train_accs.extend(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    return model, train_losses, train_accs, val_losses, val_accs


def save_test_set(df,pkl_list):
    train_id, test_id = train_test_split(pkl_list, test_size=0.05)
    
    train_idx = df[df['haim_id'].isin(train_id)]['haim_id'].tolist()
    test_idx = df[df['haim_id'].isin(test_id)]['haim_id'].tolist()

    df_train = df[df['haim_id'].isin(train_idx)]

    x_test = df[df['haim_id'].isin(test_idx)].drop(['haim_id','y'],axis=1).values

    y_test = df[df['haim_id'].isin(test_idx)]['y'].values

    with open('/mnt/mimic/data/HAIM/mimic_extras/x_test.pkl', 'wb') as f1:
        pickle.dump(x_test, f1)

    with open('/mnt/mimic/data/HAIM/mimic_extras/y_test.pkl', 'wb') as f2:
        pickle.dump(y_test, f2)

    return df_train