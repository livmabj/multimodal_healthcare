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


class CustomDataset(Dataset):
    def __init__(self, vectors, labels):
        self.vectors = vectors
        self.labels = labels
    
    def __len__(self):
        return len(self.vectors)
    
    def __getitem__(self, index):
        vector = torch.tensor(self.vectors[index]).float()

        vd = vector[:1024]
        vmd = vector[1024:2048]
        ts = vector[2048:2499]
        #ts_ce = vector[2048:2147]
        #ts_le = vector[2147:2389]
        #ts_pe = vector[2389:2499]
        n_rad = vector[2499:]

        feature_dict = {'vd': vd, 'vmd': vmd, 'ts': ts, 'n_rad': n_rad} #'ts_ce': ts_ce, 'ts_le': ts_le, 'ts_pe': ts_pe,

        label = torch.from_numpy(np.array([self.labels[index]]))
        label[label == -1] = 2
        return feature_dict, label.squeeze()


class DataSplit():

    def __init__(self, df):
        self.df = df
        self.types = ['demo_', 'vd_', 'vp_', 'vmd_', 'vmp', 'ts_ce_', 'ts_le_', 'ts_pe_', 'n_rad_']
        self.partition = None

    def partitiondata(self, partition):
        self.pkl_list = []

        if partition == 'all':
            df_mor = self.df

            df_death_small48 = df_mor[((df_mor['img_length_of_stay'] < 48) & (df_mor['death_status'] == 1))]
            df_alive_big48 = df_mor[((df_mor['img_length_of_stay'] >= 48) & (df_mor['death_status'] == 0))]
            df_death_big48 = df_mor[((df_mor['img_length_of_stay'] >= 48) & (df_mor['death_status'] == 1))]
            df_alive_small48 = df_mor[((df_mor['img_length_of_stay'] < 48) & (df_mor['death_status'] == 0))]

            df_death_small48['y'] = 1
            df_alive_big48['y'] = 0
            df_death_big48['y'] = 0
            df_alive_small48['y'] = 0
            df_mor = pd.concat([df_death_small48, df_alive_big48, df_death_big48, df_alive_small48], axis = 0)

            df_los = self.df

            df_alive_small48 = df_los[((df_los['img_length_of_stay'] < 48) & (df_los['death_status'] == 0))]
            df_alive_big48 = df_los[((df_los['img_length_of_stay'] >= 48) & (df_los['death_status'] == 0))]
            df_death = df_los[(df_los['death_status'] == 1)]

            df_alive_small48['y'] = 1
            df_alive_big48['y'] = 0
            df_death['y'] = 0
            df_los = pd.concat([df_death_small48, df_alive_big48, df_death_big48, df_alive_small48], axis = 0)

            self.df['48-hour Mortality'] = df_mor['y']
            self.df['Length-of-Stay'] = df_los['y']

            self.df['y'] = self.df.apply(lambda row: [row['Fracture'], row['Lung Lesion'], row['Enlarged Cardiomediastinum'], 
                                          row['Consolidation'], row['Pneumonia'], row['Atelectasis'], row['Lung Opacity'], row['Pneumothorax'],
                                          row['Edema'], row['Cardiomegaly'], row['Length-of-Stay'], row['48-hour Mortality']], axis=1)

            self.df = self.df.drop(['img_id', 'img_charttime', 'img_deltacharttime', 'discharge_location', 'img_length_of_stay', 
                    'death_status', 'split', 'No Finding', 'Fracture', 'Lung Lesion', 'Enlarged Cardiomediastinum', 'Consolidation', 'Pneumonia', 
                    'Atelectasis', 'Lung Opacity', 'Lung Opacity', 'Pneumothorax', 'Edema', 'Cardiomegaly', 'Pleural Effusion', 
                    'Pleural Other', 'Support Devices', 'PerformedProcedureStepDescription', 'ViewPosition', 
                    '48-hour Mortality', 'Length-of-Stay'], axis = 1)
            self.df = self.df.drop(list(self.df.filter(regex='^de_')), axis = 1)
            self.df = self.df.drop(list(self.df.filter(regex='^vp_')), axis = 1)
            self.df = self.df.drop(list(self.df.filter(regex='^vmp_')), axis = 1)

            # Remove outliers from n_rad-features
            n_rad_columns = [col for col in self.df.columns if col.startswith('n_rad_')]
            self.df = self.df[~(self.df[n_rad_columns] > 2000).any(axis=1)]

        elif partition == 'mortality':

            df_death_small48 = self.df[((self.df['img_length_of_stay'] < 48) & (self.df['death_status'] == 1))]
            df_alive_big48 = self.df[((self.df['img_length_of_stay'] >= 48) & (self.df['death_status'] == 0))]
            df_death_big48 = self.df[((self.df['img_length_of_stay'] >= 48) & (self.df['death_status'] == 1))]

            df_death_small48['y'] = 1
            df_alive_big48['y'] = 0
            df_death_big48['y'] = 0
            self.df = pd.concat([df_death_small48, df_alive_big48, df_death_big48], axis = 0)
            self.df = self.df.drop(['img_id', 'img_charttime', 'img_deltacharttime', 'discharge_location', 'img_length_of_stay', 
                    'death_status'], axis = 1)

        elif partition == 'los':

            df_alive_small48 = self.df[((self.df['img_length_of_stay'] < 48) & (self.df['death_status'] == 0))]
            df_alive_big48 = self.df[((self.df['img_length_of_stay'] >= 48) & (self.df['death_status'] == 0))]
            df_death = self.df[(self.df['death_status'] == 1)]

            df_alive_small48['y'] = 1
            df_alive_big48['y'] = 0
            df_death['y'] = 0
            self.df = pd.concat([df_alive_small48, df_alive_big48, df_death], axis = 0)

            self.df = self.df.drop(['img_id', 'img_charttime', 'img_deltacharttime', 'discharge_location', 'img_length_of_stay', 
                    'death_status'], axis = 1)
            
        else:
            self.df = self.df[self.df[partition].isin([0,1])]
            self.df = self.df.drop(['img_id', 'img_charttime', 'img_deltacharttime', 'discharge_location', 'img_length_of_stay', 
                    'death_status'], axis = 1)

            self.df['y'] = self.df[partition]

    def split_data(self, partition, validation_size=0.25, random_state=42):

        self.partition = partition

        self.partitiondata(partition)
        pkl_list = self.df['haim_id'].unique().tolist()

        # Split into training and test sets
        train_id, val_id = train_test_split(pkl_list, test_size=validation_size, random_state=random_state)

        train_idx = self.df[self.df['haim_id'].isin(train_id)]['haim_id'].tolist()
        validation_idx = self.df[self.df['haim_id'].isin(val_id)]['haim_id'].tolist()

        self.x_train = self.df[self.df['haim_id'].isin(train_idx)].drop(['y','haim_id'],axis=1)
        self.x_val = self.df[self.df['haim_id'].isin(validation_idx)].drop(['y','haim_id'],axis=1)
        
        # Normalize according to mean and std of training set
        self.x_train = (self.x_train - self.x_train.mean())/self.x_train.std()
        self.x_val = (self.x_val - self.x_train.mean())/self.x_train.std()

        self.y_train = self.df[self.df['haim_id'].isin(train_idx)]['y']
        self.y_val = self.df[self.df['haim_id'].isin(validation_idx)]['y']

    def get_data(self):
        train_cols = self.x_train
        val_cols = self.x_val
        return train_cols, val_cols


def custom_output(emb, gemma):

    outputs = gemma(inputs_embeds=emb)
    class_labels = [101, 102, 103, 201, 202, 203, 301, 302, 303, 401, 402, 403, 501, 502, 503, 601, 602, 603, 701, 702, 703, 801, 802, 803, 901, 902, 903, 1001, 1002, 1003, 1101, 1102, 1201, 1202] # 956, 3276
    logits = outputs['logits']

    logits = logits[:,-1:,class_labels].mean(dim=1) #-3
    binary_logits = logits[:, -4:].float()
    ternary_logits = logits[:, :-4].float()
    return binary_logits, ternary_logits

def binary_loss(logits, labels, loss_fns):
    labels = labels[:, -2:]
    los_fn = loss_fns[0]
    mortality_fn = loss_fns[1]
    los_loss = los_fn(logits[:, :2], labels[:, 0].long())
    mortality_loss = mortality_fn(logits[:, -2:], labels[:, 1].long())

    return [los_loss, mortality_loss]


def ternary_loss(logits, labels, loss_fns):
    num_classes = 10
    labels = labels[:, :-2]

    class_logits = logits.view(8,num_classes,3)
    losses = []
    for i, fn in enumerate(loss_fns):
        class_tensor = class_logits[:, i:i+1, :]
        mask = torch.isnan(labels[:,i].float())
        masked_labels = labels[~mask, i]
        masked_logits = class_tensor.view(8, 3)[~mask, :]
        if masked_labels.size(0) == 0:
            losses.append(0.0)
        else:
            losses.append(fn(masked_logits, masked_labels.long()))

    return losses


def custom_bce_loss(binary_logits, ternary_logits, labels, loss_fns):
    binary_losses = binary_loss(binary_logits, labels, loss_fns[-2:]) #the last two loss functions are weighted for the binary classes
    ternary_losses = ternary_loss(ternary_logits, labels, loss_fns[:-2]) #the rest of the loss functions are weighted for the pathogen classification
    losses = binary_losses + ternary_losses
    avg_bce = torch.mean(torch.tensor(losses))

    return avg_bce

def custom_mse_loss(decoded, input, mse_loss, device):
    losses = []
    for i, output in enumerate(decoded):
        mod, out = output
        original_emb = input[mod].requires_grad_().to(device)
        print(torch.isnan(original_emb).any())
        print(torch.isnan(out).any())
        original_no_nan = torch.nan_to_num(original_emb, nan=0.0)
        out_no_nan = torch.nan_to_num(out, nan=0.0)
        print(torch.isnan(original_no_nan).any())
        print(torch.isnan(out_no_nan).any())
        print(original_no_nan.shape)
        print(out_no_nan.shape)
        losses.append(mse_loss(out_no_nan, original_no_nan))

    avg_mse = torch.mean(torch.tensor(losses))

    return avg_mse

def output_to_label(binary_logits, ternary_logits):
    binary = [binary_logits[:,i:i+2] for i in range(0, binary_logits.size(0), 2)]
    ternary = [ternary_logits[:,i:i+3] for i in range(0, ternary_logits.size(0), 3)]
    logits = binary + ternary
    probs = torch.softmax(torch.tensor(logits), dim=1)
    hard_preds = torch.argmax(probs, dim=1)
    return hard_preds


"""
def output_to_label(logits):
    probs = torch.softmax(logits, dim=-1)
    predicted_token_id = torch.argmax(probs, dim=-1)
    return predicted_token_id
"""


def train_epoch(vd_model, ts_model, n_rad_model, optimizers, mse_loss, loss_fns, train_loader, device, gemma, beta):
    # Train:
    #vd_model.train()
    #ts_model.train()
    n_rad_model.train()
    train_loss_batches = []
    for batch_index, (x, y) in enumerate(train_loader, 1):
        #print('stuck')
        inputs, labels = x, y.to(device)
        #for optimizer in optimizers:
        #    optimizer.zero_grad()
        optimizers[2].zero_grad()

        all_decoded = []
        # vd_encoded = vd_model.encoder(inputs['vd'].to(device))
        # print('vd_orig', inputs['vd'])
        # print('vd_encoded:', vd_encoded)
        # vd_decoded = vd_model.decoder(vd_encoded)
        # vd_encoded = vd_encoded.to(torch.float16).view(-1, 1, 2048)
        # vd_tmp = vd_encoded.view(-1,1,2048).to(torch.float16)
        # all_decoded.append(('vd',vd_decoded))

        # ts_encoded = ts_model.encoder(inputs['ts'].to(device))
        # ts_decoded = ts_model.decoder(ts_encoded)
        # ts_encoded = ts_encoded.to(torch.float16).view(-1, 1, 2048)
        # ts_tmp = ts_encoded.view(-1,1,2048).to(torch.float16)
        # all_decoded.append(('ts',ts_decoded))

        n_rad_encoded = n_rad_model.encoder(inputs['n_rad'].to(device))
        n_rad_decoded = n_rad_model.decoder(n_rad_encoded)
        n_rad_encoded = n_rad_encoded.to(torch.float16).view(-1, 1, 2048)
        n_rad_tmp = n_rad_encoded.view(-1,1,2048).to(torch.float16)
        all_decoded.append(('n_rad',n_rad_decoded))

        #concat_emb = torch.cat((vd_tmp, ts_tmp, n_rad_tmp), dim=1).to(device)
        concat_emb = n_rad_tmp.to(device)
        binary_logits, ternary_logits = custom_output(concat_emb, gemma)

        loss_bce = custom_bce_loss(binary_logits,ternary_logits,labels.float(),loss_fns)
        loss_mse = custom_mse_loss(all_decoded, inputs, mse_loss, device)
        loss = loss_bce + beta*loss_mse

        loss.backward()
        #for optimizer in optimizers:
        #    optimizer.step()
        optimizers[2].step
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

            n_rad_encoded = n_rad_model.encoder(inputs['n_rad'])
            n_rad_decoded = n_rad_model.decoder(n_rad_encoded)
            n_rad_encoded = n_rad_encoded.to(torch.float16).view(-1, 1, 2048)
            n_rad_tmp = n_rad_encoded.view(-1,1,2048).to(torch.float16)
            all_decoded.append(('n_rad',n_rad_decoded))

            concat_emb = torch.cat(vd_tmp, ts_tmp, n_rad_tmp)
            binary_logits, ternary_logits = custom_output(concat_emb, gemma)

            loss_bce = custom_bce_loss(binary_logits,ternary_logits,labels,loss_fns)
            loss_mse = custom_mse_loss(all_decoded, inputs)
            loss = loss_bce + beta*loss_mse

            hard_preds = output_to_label(binary_logits, ternary_logits)

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
        vd_model, ts_model, n_rad_model, train_loss = train_epoch(vd_model,
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
        
        folder = 'results/auto'

        if f1_score > best_f1:
            best_f1 = f1_score
            torch.save(vd_model, f"{folder}/vd_model.pth")
            torch.save(ts_model, f"{folder}/ts_model.pth")
            torch.save(n_rad_model, f"{folder}/n_rad_model.pth")

        with open(f"{folder}/train_losses.pkl", 'wb') as f1:
            pickle.dump(train_losses, f1)

        with open(f"{folder}/val_losses.pkl", 'wb') as f3:
            pickle.dump(val_losses, f3)

    return vd_model, ts_model, n_rad_model, train_losses, val_losses