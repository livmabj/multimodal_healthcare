import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from focal_loss.focal_loss import FocalLoss
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pickle

EMBEDDING_SIZE = 1024
PROJECTION_SIZE = 32


class AttentionProjectTsNN(nn.Module):
    def __init__(self, projection_size):
        super(AttentionProjectTsNN, self).__init__()

        self.projection_size = projection_size
        
        # Architecture
        self.fc1 = nn.Linear(110, 2048).cuda()
        self.ln1 = nn.LayerNorm(2048).cuda()
        self.fc2 = nn.Linear(2048, 1024).cuda() ###
        self.ln2 = nn.LayerNorm(1024).cuda() ###
        self.fc4 = nn.Linear(1024, 2048).cuda() ###
        self.ln4 = nn.LayerNorm(2048).cuda() ###
        self.fc3 = nn.Linear(2048, 2048 * self.projection_size).cuda()
        self.ln3 = nn.LayerNorm(2048 * self.projection_size).cuda()
        
        # Attention Mechanism
        self.attention = nn.Linear(2048, 1).cuda()

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln4(self.fc4(x)))
        
        # Attention Mechanism
        attention_weights = F.softmax(self.attention(x), dim=1)
        x = torch.mul(x, attention_weights)
        
        x = F.relu(self.ln3(self.fc3(x)))
        x = x.view(-1, self.projection_size, 2048)
        #print(x.shape)
        return x

class ClfHead(nn.Module):
    def __init__(self, projection_size):
        super(ClfHead, self).__init__()

        self.projection_size = projection_size

        # Define the classification head
        self.classification_head = nn.Sequential(
            nn.Linear(2048*self.projection_size,1024),
            #nn.Linear(110*1,1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 2)# Output layer with 1 unit for binary classification
        )

    def forward(self, x):
        # Flatten the output for the classification head
        batch_size, n_tokens, embedding_dimensions = x.size()
        #flattened = x.view(batch_size, -1)
        flattened = x.reshape(batch_size, -1)
        # Forward pass through the classification head
        output = self.classification_head(flattened)

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

    
def train_epoch(proj_model, clf_head, optimizer, loss_fn, train_loader, device):
    # Train:
    proj_model.train()
    train_loss_batches, train_acc_batches = [], []
    for batch_index, (x, y) in tqdm(enumerate(train_loader, 1)):
        inputs, labels = x.to(device), y.to(device)
        optimizer.zero_grad()

        proj_outputs = proj_model(inputs)

        emb = clf_head(proj_outputs)
        #word_embs_extended = word_embs.repeat(len(inputs),1,1).detach()

        #concatted = torch.cat((word_embs_extended, emb), dim=1).to(torch.float16)
        #logits = custom_output(emb, gemma).float()
        
        loss = loss_fn(emb, labels.long())
        loss.backward()
        optimizer.step()
        train_loss_batches.append(loss.item())

        hard_preds = output_to_label(emb)
        hard_preds = torch.argmax(hard_preds, dim=1)
        acc_batch_avg = (hard_preds == labels).float().mean().item()
        train_acc_batches.append(acc_batch_avg)

    return proj_model, train_loss_batches, train_acc_batches


def validate(proj_model, clf_head, loss_fn, val_loader, device):
    val_loss_cum = 0
    val_acc_cum = 0
    proj_model.eval()
    all_preds = []
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(val_loader, 1):
            inputs, labels = x.to(device), y.to(device)

            proj_outputs = proj_model(inputs)

            emb = clf_head(proj_outputs)
            #word_embs_extended = word_embs.repeat(len(inputs),1,1).detach()

            #concatted = torch.cat((word_embs_extended, emb), dim=1).to(torch.float16)
            #logits = custom_output(emb, gemma)

            batch_loss = loss_fn(emb, labels.long())
            val_loss_cum += batch_loss.item()
            hard_preds = output_to_label(emb)
            hard_preds = torch.argmax(hard_preds, dim=1)
            all_preds.extend(hard_preds)
            acc_batch_avg = (hard_preds == labels).float().mean().item()
            val_acc_cum += acc_batch_avg
    numpy_arrays = [t.cpu().numpy() for t in all_preds]
    preds = np.array(numpy_arrays)
    return val_loss_cum/len(val_loader), val_acc_cum/len(val_loader), preds


def training_loop(proj_model, clf_head, optimizer, criterion, train_loader, val_loader, device, num_epochs):
    print("Starting training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proj_model.to(device)
    clf_head.to(device)
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    for epoch in range(1, num_epochs+1):
        proj_model, train_loss, train_acc = train_epoch(proj_model, clf_head,
                                                   optimizer,
                                                   criterion,
                                                   train_loader,
                                                   device)
        val_loss, val_acc, preds = validate(proj_model, clf_head, criterion, val_loader, device)
        print(f"Epoch {epoch}/{num_epochs}: "
              f"Train loss: {sum(train_loss)/len(train_loss):.3f}, "
              f"Train acc.: {sum(train_acc)/len(train_acc):.3f}, "
              f"Val. loss: {val_loss:.3f}, "
              f"Val. acc.: {val_acc:.3f}")
        train_losses.append(sum(train_loss)/len(train_loss))
        train_accs.append(sum(train_acc)/len(train_acc))
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    return proj_model, train_losses, train_accs, val_losses, val_accs, preds

def select_random_subset(data, subset_fraction=0.1):
    num_samples = int(len(data) * subset_fraction)
    indices = np.random.choice(len(data), num_samples, replace=False)
    subset = data[indices]
    return subset
