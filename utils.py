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
PROJECTION_SIZE = 1


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
        x = x.view(-1,PROJECTION_SIZE,2048)
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
    

class DataSplit():

    '''Class to get the correct data-partition and split up in train, validation and test.
    Initialized with full df, and then datasplit is created through calling get_data(partition='los' or 'mortality')
    Size of validation and test can also be customized.
    '''

    def __init__(self, df):
        self.df = df
        self.types = ['vd_', 'vp_', 'vmd_', 'ts_ce_', 'ts_le_', 'ts_pe_', 'n_rad_']
        self.partition = None

    def partitiondata(self, partition):
        self.pkl_list = []
        if partition == 'mortality':
            condition_death_small48 = (self.df['img_length_of_stay'] < 48) & (self.df['death_status'] == 1)

            y = [0]*len(self.df)
            for i, condition in enumerate(condition_death_small48):
                if condition:
                    y[i] = 1

        if partition == 'los':
            condition_alive_small48 = self.df[((self.df['img_length_of_stay'] < 48) & (self.df['death_status'] == 0))]

            y = [0]*len(self.df)
            for i, condition in enumerate(condition_alive_small48):
                if condition:
                    y[i] = 1

        self.df['y'] = y

    def get_data(self, partition, test_size=0.3, validation_size=0.1, random_state=None):

        self.partition = partition

        self.partitiondata(partition)
        pkl_list = self.df['haim_id'].unique().tolist()

        # Split into training and test sets
        train_set, test_set = train_test_split(pkl_list, test_size=test_size, random_state=random_state)

        remaining_data_size = 1.0 - test_size
        validation_size = validation_size*remaining_data_size

        # Further split the training set into training and validation sets
        train_set, validation_set = train_test_split(train_set, test_size=validation_size, random_state=random_state)

        train_idx = self.df[self.df['haim_id'].isin(train_set)]['haim_id'].tolist()
        validation_idx = self.df[self.df['haim_id'].isin(validation_set)]['haim_id'].tolist()
        test_idx = self.df[self.df['haim_id'].isin(test_set)]['haim_id'].tolist()

        self.x_train = {t: self.df[self.df['haim_id'].isin(train_idx)].filter(regex='^'+t).values for t in self.types}
        self.x_validation = {t: self.df[self.df['haim_id'].isin(validation_idx)].filter(regex='^'+t).values for t in self.types}
        self.x_test = {t: self.df[self.df['haim_id'].isin(test_idx)].filter(regex='^'+t).values for t in self.types}

        self.y_train = self.df[self.df['haim_id'].isin(train_idx)]['y'].values
        self.y_validation = self.df[self.df['haim_id'].isin(validation_idx)]['y'].values
        self.y_test = self.df[self.df['haim_id'].isin(test_idx)]['y'].values


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
    logits = logits[:,-1:,noyes].mean(dim=1)
    return logits


def output_to_label(logits):
    probs = torch.softmax(logits, dim=-1)
    predicted_token_id = torch.argmax(probs, dim=-1)
    return predicted_token_id

    
def train_epoch(model, gemma, optimizer, loss_fn, train_loader, device):
    # Train:
    model.train()
    train_loss_batches, train_acc_batches = [], []
    for batch_index, (x, y) in enumerate(train_loader, 1):
        inputs, labels = x.to(device), y.to(device)

        optimizer.zero_grad()

        emb = model.forward(inputs).to(torch.float16)
        #word_embs_extended = word_embs.repeat(len(inputs),1,1).detach()

        #concatted = torch.cat((word_embs_extended, emb), dim=1).to(torch.float16)
        logits = custom_output(emb, gemma).float()
        
        loss = loss_fn(logits, labels.long())
        loss.backward()
        optimizer.step()
        train_loss_batches.append(loss.item())

        hard_preds = output_to_label(logits)
        acc_batch_avg = (hard_preds == labels).float().mean().item()
        train_acc_batches.append(acc_batch_avg)

    return model, train_loss_batches, train_acc_batches


def validate(model, gemma, loss_fn, val_loader, device):
    val_loss_cum = 0
    val_acc_cum = 0
    model.eval()
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(val_loader, 1):
            inputs, labels = x.to(device), y.to(device)

            emb = model.forward(inputs).to(torch.float16)
            #word_embs_extended = word_embs.repeat(len(inputs),1,1).detach()

            #concatted = torch.cat((word_embs_extended, emb), dim=1).to(torch.float16)
            logits = custom_output(emb, gemma)

            batch_loss = loss_fn(logits, labels.long())
            val_loss_cum += batch_loss.item()
            hard_preds = output_to_label(logits)
            acc_batch_avg = (hard_preds == labels).float().mean().item()
            val_acc_cum += acc_batch_avg
    return val_loss_cum/len(val_loader), val_acc_cum/len(val_loader)


def training_loop(model, gemma, optimizer, loss_fn, train_loader, val_loader, num_epochs):
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