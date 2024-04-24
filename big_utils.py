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
from sklearn import metrics

EMBEDDING_SIZE = 1024
PROJECTION_SIZE = 32


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, latent_dim),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.LeakyReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


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
    class_labels = [101, 102, 201, 202, 301, 302, 303, 401, 402, 403, 501, 502, 503, 601, 602, 603, 701, 702, 703, 801, 802, 803, 901, 902, 903, 1001, 1002, 1003, 1201, 1202, 1203] # 956, 3276
    logits = outputs['logits']
    logits = logits[:,-3:,class_labels].mean(dim=1)
    binary_logits = logits[:4].float()
    ternary_logits = logits[4:].float()
    return binary_logits, ternary_logits

def binary_loss(logits, labels, loss_fns):
    labels = labels.long()[:4]
    los_fn = loss_fns[0]
    mortality_fn = loss_fns[1]
    los_loss = los_fn(logits[:2], labels[:2])
    mortality_loss = mortality_fn(logits[2:], labels[2:])

    return [los_loss, mortality_loss]


def ternary_loss(logits, labels, loss_fns):
    labels = labels.long()[4:]
    class_logits = [logits[i:i+3] for i in range(0, logits.size(0), 3)]
    losses = []
    for i, fn in enumerate(loss_fns):
        losses.append(fn(class_logits[i], labels[i]))

    return losses


def custom_bce_loss(binary_logits, ternary_logits, labels, loss_fns):
    binary_losses = binary_loss(binary_logits, labels, loss_fns[:2]) #the first two loss functions are weighted for the binary classes
    ternary_losses = ternary_loss(ternary_logits, labels, loss_fns[2:]) #the rest of the loss functions are weighted for the pathogen classification
    losses = binary_losses + ternary_losses
    avg_bce = torch.mean(torch.tensor(losses))

    return avg_bce

def custom_mse_loss(decoded, input, mse_loss):
    losses = []
    for i, output in enumerate(decoded):
        mod, out = output
        losses.append(mse_loss(out, input[mod]))

    avg_mse = torch.mean(torch.tensor(losses))

    return avg_mse

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


def train_epoch(vd_model, ts_model, n_rad_model, optimizers, mse_loss, loss_fns, train_loader, device, gemma, beta):
    # Train:
    vd_model.train()
    ts_model.train()
    n_rad_model.train()
    train_loss_batches = []
    for batch_index, (x, y) in enumerate(train_loader, 1):
        inputs, labels = x.to(device), y.to(device)
        for optimizer in optimizers:
            optimizer.zero_grad()

        all_decoded = []
        vd_encoded = vd_model.encoder(inputs['vd'])
        vd_decoded = vd_model.decoder(vd_encoded)
        vd_encoded = vd_encoded.to(torch.float16).view(-1, 1, 2048)
        vd_tmp = vd_encoded.view(-1,1,2048).to(torch.float16)
        all_decoded.append(('vd',vd_decoded))

        ts_encoded = ts_model.encoder(inputs['ts'])
        ts_decoded = ts_model.decoder(ts_encoded)
        ts_encoded = ts_encoded.to(torch.float16).view(-1, 1, 2048)
        ts_tmp = ts_encoded.view(-1,1,2048).to(torch.float16)
        all_decoded.append(('ts',ts_decoded))

        n_rad_encoded = ts_model.encoder(inputs['n_rad'])
        n_rad_decoded = ts_model.decoder(n_rad_encoded)
        n_rad_encoded = n_rad_encoded.to(torch.float16).view(-1, 1, 2048)
        n_rad_tmp = n_rad_encoded.view(-1,1,2048).to(torch.float16)
        all_decoded.append(('n_rad',n_rad_decoded))

        concat_emb = torch.cat(vd_tmp, ts_tmp, n_rad_tmp)
        binary_logits, ternary_logits = custom_output(concat_emb, gemma)

        loss_bce = custom_bce_loss(binary_logits,ternary_logits,labels,loss_fns)
        loss_mse = custom_mse_loss(all_decoded, inputs)
        loss = loss_bce + beta*loss_mse

        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        train_loss_batches.append(loss.item())

    return vd_model, ts_model, n_rad_model, train_loss_batches


def validate(vd_model, ts_model, n_rad_model, mse_loss, loss_fns, val_loader, device, gemma, beta):
    val_loss_cum = 0
    val_acc_cum = 0
    preds = []
    list_labels = []

    vd_model.eval()
    ts_model.eval()
    n_rad_model.eval()
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(val_loader, 1):
            inputs, labels = x.to(device), y.to(device)

            all_decoded = []
            vd_encoded = vd_model.encoder(inputs['vd'])
            vd_decoded = vd_model.decoder(vd_encoded)
            vd_encoded = vd_encoded.to(torch.float16).view(-1, 1, 2048)
            vd_tmp = vd_encoded.view(-1,1,2048).to(torch.float16)
            all_decoded.append(('vd',vd_decoded))

            ts_encoded = ts_model.encoder(inputs['ts'])
            ts_decoded = ts_model.decoder(ts_encoded)
            ts_encoded = ts_encoded.to(torch.float16).view(-1, 1, 2048)
            ts_tmp = ts_encoded.view(-1,1,2048).to(torch.float16)
            all_decoded.append(('ts',ts_decoded))

            n_rad_encoded = ts_model.encoder(inputs['n_rad'])
            n_rad_decoded = ts_model.decoder(n_rad_encoded)
            n_rad_encoded = n_rad_encoded.to(torch.float16).view(-1, 1, 2048)
            n_rad_tmp = n_rad_encoded.view(-1,1,2048).to(torch.float16)
            all_decoded.append(('n_rad',n_rad_decoded))

            concat_emb = torch.cat(vd_tmp, ts_tmp, n_rad_tmp)
            binary_logits, ternary_logits = custom_output(concat_emb, gemma)

            loss_bce = custom_bce_loss(binary_logits,ternary_logits,labels,loss_fns)
            loss_mse = custom_mse_loss(all_decoded, inputs)
            loss = loss_bce + beta*loss_mse

            hard_preds = output_to_label(logits)
            hard_preds = torch.argmax(hard_preds, dim=1)

            preds.extend(hard_preds)
            list_labels.extend(labels)

            val_loss_cum += loss.item()

    return val_loss_cum/len(val_loader), preds, list_labels


def training_loop(vd_model, ts_model, n_rad_model, optimizers, mse_loss, loss_fns, train_loader, val_loader, num_epochs, gemma, beta):
    print("Starting training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vd_model.to(device)
    ts_model.to(device)
    n_rad_model.to(device)
    train_losses, val_losses= [], []
    #best_val_loss = float('inf')
    best_f1 = 0

    for epoch in range(1, num_epochs+1):
        model, train_loss = train_epoch(vd_model,
                                        ts_model,
                                        n_rad_model,
                                        optimizers,
                                        mse_loss,
                                        loss_fns,
                                        train_loader,
                                        device,
                                        gemma,
                                        beta)
        val_loss, preds, labels = validate(vd_model, ts_model, n_rad_model, mse_loss, loss_fns, val_loader, device, gemma, beta)

        preds_arrays = [t.cpu().numpy() for t in preds]
        preds = np.array(preds_arrays)

        labels_arrays = [t.cpu().numpy() for t in labels]
        labels = np.array(labels_arrays)

        f1_score = metrics.f1_score(labels, preds)

        #scheduler.step(val_loss)
        print(f"Epoch {epoch}/{num_epochs}: "
              f"Train loss: {sum(train_loss)/len(train_loss):.3f}, "
              f"Val. loss: {val_loss:.3f}, "
              f"F1-score: {f1_score:.3f}")
        train_losses.append(sum(train_loss)/len(train_loss))
        val_losses.append(val_loss)
        
        folder = 'results/auto_vd'

        if f1_score > best_f1:
            best_f1 = f1_score
            torch.save(model, f"{folder}/model.pth")

        with open(f"{folder}/train_losses.pkl", 'wb') as f1:
            pickle.dump(train_losses, f1)

        with open(f"{folder}/val_losses.pkl", 'wb') as f3:
            pickle.dump(val_losses, f3)

    return vd_model, ts_model, n_rad_model, train_losses, val_losses