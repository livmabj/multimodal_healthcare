import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from focal_loss.focal_loss import FocalLoss
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pickle

EMBEDDING_SIZE = 1024
PROJECTION_SIZE = 32


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



class ClfHead3(nn.Module):
    def __init__(self):
        super(ClfHead3, self).__init__()

        self.input_size = 2048

        # Define the classification head
        self.classification_head = nn.Sequential(
            nn.BatchNorm1d(self.input_size),
            nn.Linear(self.input_size,self.input_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.input_size,self.input_size//2),
            nn.ReLU(),
            nn.Linear(self.input_size//2, 2),
            #nn.Flatten()

        )

    def forward(self, x):

        output = self.classification_head(x)


        return output


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

    def __init__(self, df):
        self.df = df
        self.types = ['demo_', 'vd_', 'vp_', 'vmd_', 'vmp', 'ts_ce_', 'ts_le_', 'ts_pe_', 'n_rad_']
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

    def split_data(self, partition, test_size=0.1, validation_size=0.25, random_state=42):

        self.partition = partition

        self.partitiondata(partition)
        pkl_list = self.df['haim_id'].unique().tolist()

        # Split into training and test sets
        train_id, val_id = train_test_split(pkl_list, test_size=validation_size, random_state=random_state)

        remaining_data_size = 1.0 - validation_size
        test_size = test_size*remaining_data_size

        # Further split the training set into training and validation sets
        train_id, test_id = train_test_split(train_id, test_size=test_size, random_state=random_state)

        train_idx = self.df[self.df['haim_id'].isin(train_id)]['haim_id'].tolist()
        validation_idx = self.df[self.df['haim_id'].isin(val_id)]['haim_id'].tolist()
        test_idx = self.df[self.df['haim_id'].isin(test_id)]['haim_id'].tolist()

        self.x_train = self.df[self.df['haim_id'].isin(train_idx)].drop(['y','haim_id'],axis=1)
        self.x_validation = self.df[self.df['haim_id'].isin(validation_idx)].drop(['y','haim_id'],axis=1)
        self.x_test = self.df[self.df['haim_id'].isin(test_idx)].drop(['y','haim_id'],axis=1)

        self.y_train = self.df[self.df['haim_id'].isin(train_idx)]['y']
        self.y_validation = self.df[self.df['haim_id'].isin(validation_idx)]['y']
        self.y_test = self.df[self.df['haim_id'].isin(test_idx)]['y']

    def get_type(self, requested):
        train_cols = self.x_train.filter(regex='^'+requested)
        val_cols = self.x_validation.filter(regex='^'+requested)
        test_cols = self.x_test.filter(regex='^'+requested)
        return train_cols, val_cols, test_cols
        


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
    logits = logits[:,-32:,noyes].mean(dim=1)
    probs = torch.softmax(logits, dim=-1)
    return logits

def output_to_label(z):
    """Map network output z to a hard label {0, 1}
    
    Args:
        z (Tensor): Probabilities for each sample in a batch.
    Returns:
        c (Tensor): Hard label {0, 1} for each sample in a batch
    """
    # YOUR CODE HERE
    c = torch.round(z)
    return c.long()


"""
def output_to_label(logits):
    probs = torch.softmax(logits, dim=-1)
    predicted_token_id = torch.argmax(probs, dim=-1)
    return predicted_token_id
"""


def train_epoch(model, optimizer, loss_fn, train_loader, device):
    # Train:
    model.train()
    train_loss_batches = []
    for batch_index, (x, y) in enumerate(train_loader, 1):
        inputs, labels = x.to(device), y.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        
        loss = loss_fn(outputs, inputs)
        loss.backward()
        optimizer.step()
        train_loss_batches.append(loss.item())

    return model, train_loss_batches


def validate(model, loss_fn, val_loader, device):
    val_loss_cum = 0
    val_acc_cum = 0
    model.eval()
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(val_loader, 1):
            inputs, labels = x.to(device), y.to(device)

            outputs = model(inputs)
        
            batch_loss = loss_fn(outputs, inputs)

            val_loss_cum += batch_loss.item()

    return val_loss_cum/len(val_loader)


def training_loop(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, scheduler):
    print("Starting training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_losses, val_losses= [], []
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs+1):
        model, train_loss = train_epoch(model,
                                                   optimizer,
                                                   loss_fn,
                                                   train_loader,
                                                   device)
        val_loss = validate(model, loss_fn, val_loader, device)
        scheduler.step(val_loss)
        print(f"Epoch {epoch}/{num_epochs}: "
              f"Train loss: {sum(train_loss)/len(train_loss):.3f}, "
              f"Val. loss: {val_loss:.3f}, ")
        train_losses.append(sum(train_loss)/len(train_loss))
        val_losses.append(val_loss)
        folder = 'results/ts_pe/autoencoder_allts'

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, f"{folder}/model.pth")

        with open(f"{folder}/train_losses.pkl", 'wb') as f1:
            pickle.dump(train_losses, f1)

        with open(f"{folder}/val_losses.pkl", 'wb') as f3:
            pickle.dump(val_losses, f3)

    return model, train_losses, val_losses


def train_epoch_clf(proj_model, clf_head, clf_optimizer, loss_fn, train_loader, device):
    # Train:
    #proj_model.train()
    clf_head.train()
    train_loss_batches, train_acc_batches = [], []
    for batch_index, (x, y) in enumerate(train_loader, 1):
        inputs, labels = x.to(device), y.to(device)
        #optimizer.zero_grad()
        clf_optimizer.zero_grad()

        #emb = proj_model.forward(inputs)
        enc = proj_model.encoder(inputs)
        emb = clf_head(enc)
        #word_embs_extended = word_embs.repeat(len(inputs),1,1).detach()

        #concatted = torch.cat((word_embs_extended, emb), dim=1).to(torch.float16)
        #logits = custom_output(emb, gemma).float()
        
        loss = loss_fn(emb, labels.long())
        loss.backward()
        #optimizer.step()
        clf_optimizer.step()
        train_loss_batches.append(loss.item())

        hard_preds = output_to_label(emb)
        hard_preds = torch.argmax(hard_preds, dim=1)
        acc_batch_avg = (hard_preds == labels).float().mean().item()
        train_acc_batches.append(acc_batch_avg)

    return clf_head, train_loss_batches, train_acc_batches


def validate_clf(proj_model, clf_head, loss_fn, val_loader, device):
    val_loss_cum = 0
    val_acc_cum = 0
    #proj_model.eval()
    clf_head.eval()
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(val_loader, 1):
            inputs, labels = x.to(device), y.to(device)

            #emb = proj_model.forward(inputs)
            enc = proj_model.encoder(inputs)
            emb = clf_head(enc)
            #word_embs_extended = word_embs.repeat(len(inputs),1,1).detach()

            #concatted = torch.cat((word_embs_extended, emb), dim=1).to(torch.float16)
            #logits = custom_output(emb, gemma)

            batch_loss = loss_fn(emb, labels.long())
            val_loss_cum += batch_loss.item()
            hard_preds = output_to_label(emb)
            hard_preds = torch.argmax(hard_preds, dim=1)
            acc_batch_avg = (hard_preds == labels).float().mean().item()
            val_acc_cum += acc_batch_avg
    return val_loss_cum/len(val_loader), val_acc_cum/len(val_loader)

def training_loop_clf(proj_model, clf_head, clf_optimizer, loss_fn, train_loader, val_loader, num_epochs, scheduler):
    print("Starting training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proj_model.to(device)
    clf_head.to(device)
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs+1):
        clf_head, train_loss, train_acc = train_epoch_clf(proj_model, clf_head,
                                                clf_optimizer,
                                                   loss_fn,
                                                   train_loader,
                                                   device)
        val_loss, val_acc = validate_clf(proj_model, clf_head, loss_fn, val_loader, device)
        scheduler.step(val_loss)
        print(f"Epoch {epoch}/{num_epochs}: "
              f"Train loss: {sum(train_loss)/len(train_loss):.3f}, "
              f"Train acc.: {sum(train_acc)/len(train_acc):.3f}, "
              f"Val. loss: {val_loss:.3f}, "
              f"Val. acc.: {val_acc:.3f}")
        train_losses.append(sum(train_loss)/len(train_loss))
        train_accs.append(sum(train_acc)/len(train_acc))
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    return clf_head, train_losses, train_accs, val_losses, val_accs