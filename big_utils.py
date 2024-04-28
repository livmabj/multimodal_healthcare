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





####
####
####
####                                                                                ARCHITECTURES FOR PROJECTION MODULES
####
####
####
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









####
####
####
####                                                                                DATA-RELATED CODE
####                                                                        Includes a custom dataset class as well
####                                                                        as a preproccesing and data splitting class
####
####
####
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
        ts_ce = vector[2048:2147]
        ts_le = vector[2147:2389]
        ts_pe = vector[2389:2499]
        n_rad = vector[2499:]

        feature_dict = {'vd': vd, 'vmd': vmd, 'ts_ce': ts_ce, 'ts_le': ts_le, 'ts_pe': ts_pe, 'n_rad': n_rad}

        label = torch.from_numpy(np.array([self.labels[index]]))
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
            self.df = self.df[~(self.df[n_rad_columns] > 10).any(axis=1)]
            self.df = self.df.dropna()

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
        for column in self.x_train.columns:
            mean = self.x_train[column].mean()
            std = self.x_train[column].std()
            
            if std != 0:
                self.x_train[column] = (self.x_train[column] - mean) / std
                self.x_val[column] = (self.x_val[column] - mean) / std
            else:
                continue

        self.y_train = self.df[self.df['haim_id'].isin(train_idx)]['y']
        self.y_val = self.df[self.df['haim_id'].isin(validation_idx)]['y']

    def get_data(self):
        train_cols = self.x_train
        val_cols = self.x_val
        return train_cols, val_cols
        




####
####
####
####                                                                                HELPER FUNCTIONS
####
####
####
def custom_output(emb, gemma):
    outputs = gemma(inputs_embeds=emb)
    class_labels = [101, 102, 103, 201, 202, 203, 301, 302, 303, 401, 402, 403, 501, 502, 503, 601, 602, 603, 701, 702, 703, 801, 802, 803, 901, 902, 903, 1001, 1002, 1003, 1101, 1102, 1201, 1202] # 956, 3276
    logits = outputs['logits']
    logits = logits[:,-6:,class_labels].mean(dim=1) #-3
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
    batch_size = labels.size(0)
    num_classes = 10
    labels = labels[:, :-2]

    class_logits = logits.view(batch_size,num_classes,3)
    losses = []
    for i, fn in enumerate(loss_fns):
        class_tensor = class_logits[:, i:i+1, :]
        mask = torch.isnan(labels[:,i].float())
        masked_labels = labels[~mask, i]
        masked_logits = class_tensor.view(batch_size, 3)[~mask, :]
        if masked_labels.size(0) == 0:
            losses.append(0.0)
        else:
            losses.append(fn(masked_logits, masked_labels.long()))

    return losses


def custom_bce_loss(binary_logits, ternary_logits, labels, loss_fns):
    binary_losses = binary_loss(binary_logits, labels, loss_fns[-2:]) #the last two loss functions are weighted for the binary classes
    ternary_losses = ternary_loss(ternary_logits, labels, loss_fns[:-2]) #the rest of the loss functions are weighted for the pathogen classification
    losses = binary_losses + ternary_losses
    avg_bce = sum(losses) / len(losses)

    return avg_bce


def custom_mse_loss(decoded, inputs, mse_loss, device):

    vd_loss = mse_loss(decoded[0], inputs[0])
    vmd_loss = mse_loss(decoded[1], inputs[1])
    ts_pe_loss = mse_loss(decoded[2], inputs[2])
    ts_ce_loss = mse_loss(decoded[3], inputs[3])
    ts_le_loss = mse_loss(decoded[4], inputs[4])
    n_rad_loss = mse_loss(decoded[5], inputs[5])

    avg_mse = (vd_loss + vmd_loss + ts_pe_loss + ts_ce_loss + ts_le_loss + n_rad_loss) / 6

    return avg_mse


def output_to_label(binary_logits, ternary_logits, labels):
    batch_size = labels.size(0)
    num_classes = 10
    probs = []
    preds = []
    binary = [binary_logits[:, :2], binary_logits[:, 2:]]
    ternary_class_logits = ternary_logits.view(batch_size,num_classes,3)
    ternary = [ternary_class_logits[:, i, :] for i in range(10)]
    logits_list = ternary + binary
    for logits_tensor in logits_list:

        probs_tensor = F.softmax(logits_tensor, dim=1)
        max_probs, max_indices = torch.max(probs_tensor, dim=1)

        probs.append(probs_tensor)
        preds.append(max_indices)
        
    probs_tensor = torch.stack(probs)
    preds_tensor = torch.stack(preds)
    transposed_labels = labels.transpose(0, 1)

    return probs_tensor, preds_tensor, transposed_labels


def accuracy_score(probabilities, hard_preds, labels):
    per_class_probs = [[] for _ in range(12)]
    per_class_preds = [[] for _ in range(12)]
    per_class_labels = [[] for _ in range(12)]

    for i, class_tensor in enumerate(hard_preds):
            per_class_preds[i].append(class_tensor)
            per_class_labels[i].append(labels[i])
            per_class_probs[i].append(probabilities[i])
        
        
    per_class_acc_batch_avg = []
    for i,hard_preds in enumerate(per_class_preds):
        mask = torch.isnan(per_class_labels[i][0])
        masked_labels = per_class_labels[i][0][~mask]
        masked_preds = hard_preds[0][~mask]
        if len(masked_labels) == 0:
            continue
        else:
            per_class_acc_batch_avg.append((masked_preds == masked_labels).float().mean().item())
    
    acc_batch_avg = sum(per_class_acc_batch_avg) / len(per_class_acc_batch_avg)

    return per_class_preds, per_class_labels, acc_batch_avg




####
####
####
####                                                                                TRAINING LOOPS
####
####
####
def train_epoch(models, optimizers, mse_loss, loss_fns, train_loader, device, gemma, beta):
    # Train:
    for model in models:
        model.train()
    train_loss_batches, train_acc_batches = [], []
    for batch_index, (x, y) in enumerate(train_loader, 1):
        inputs, labels = x, y.to(device)

        for optimizer in optimizers:
            optimizer.zero_grad()


        vd_inputs = x['vd'].to(device)
        vmd_inputs = x['vmd'].to(device)
        ts_pe_inputs = x['ts_pe'].to(device)
        ts_ce_inputs = x['ts_ce'].to(device)
        ts_le_inputs = x['ts_le'].to(device)
        n_rad_inputs = x['n_rad'].to(device)

        encoded_vd = models[0].encoder(vd_inputs)
        encoded_vmd = models[1].encoder(vmd_inputs)
        encoded_ts_pe = models[2].encoder(ts_pe_inputs)
        encoded_ts_ce = models[3].encoder(ts_ce_inputs)
        encoded_ts_le = models[4].encoder(ts_le_inputs)
        encoded_n_rad = models[5].encoder(n_rad_inputs)

        decoded_vd = models[0].decoder(encoded_vd)
        decoded_vmd = models[1].decoder(encoded_vmd)
        decoded_ts_pe = models[2].decoder(encoded_ts_pe)
        decoded_ts_ce = models[3].decoder(encoded_ts_ce)
        decoded_ts_le = models[4].decoder(encoded_ts_le)
        decoded_n_rad = models[5].decoder(encoded_n_rad)

        inputs = [vd_inputs, vmd_inputs, ts_pe_inputs, ts_ce_inputs, ts_le_inputs, n_rad_inputs]
        decoded = [decoded_vd, decoded_vmd, decoded_ts_pe, decoded_ts_ce, decoded_ts_le, decoded_n_rad]

        concat_emb = torch.cat((encoded_vd.view(-1,1,2048).to(torch.float16), encoded_vmd.view(-1,1,2048).to(torch.float16), 
                                encoded_ts_pe.view(-1,1,2048).to(torch.float16), encoded_ts_ce.view(-1,1,2048).to(torch.float16), 
                                encoded_ts_le.view(-1,1,2048).to(torch.float16), encoded_n_rad.view(-1,1,2048).to(torch.float16)),
                                  dim=1).to(device)

        binary_logits, ternary_logits = custom_output(concat_emb, gemma) 

        loss_bce = custom_bce_loss(binary_logits, ternary_logits, labels.float(), loss_fns)
        loss_mse = custom_mse_loss(decoded, inputs, mse_loss, device)
        loss = loss_bce + beta*loss_mse

        loss.backward()

        for optimizer in optimizers:
            optimizer.step()
        

        probabilities, hard_preds, labels = output_to_label(binary_logits, ternary_logits, labels)
        _, _, acc_batch_avg = accuracy_score(probabilities, hard_preds, labels)

        train_acc_batches.append(acc_batch_avg)

        train_loss_batches.append(loss.item())

    return model, train_loss_batches, train_acc_batches


def validate(models, mse_loss, loss_fns, val_loader, device, gemma, beta):
    val_loss_cum = 0
    val_acc_cum = 0

    for model in models:
        model.eval()
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(val_loader, 1):
            inputs, labels = x, y.to(device)

            vd_inputs = x['vd'].to(device)
            vmd_inputs = x['vmd'].to(device)
            ts_pe_inputs = x['ts_pe'].to(device)
            ts_ce_inputs = x['ts_ce'].to(device)
            ts_le_inputs = x['ts_le'].to(device)
            n_rad_inputs = x['n_rad'].to(device)

            encoded_vd = models[0].encoder(vd_inputs)
            encoded_vmd = models[1].encoder(vmd_inputs)
            encoded_ts_pe = models[2].encoder(ts_pe_inputs)
            encoded_ts_ce = models[3].encoder(ts_ce_inputs)
            encoded_ts_le = models[4].encoder(ts_le_inputs)
            encoded_n_rad = models[5].encoder(n_rad_inputs)

            decoded_vd = models[0].decoder(encoded_vd)
            decoded_vmd = models[1].decoder(encoded_vmd)
            decoded_ts_pe = models[2].decoder(encoded_ts_pe)
            decoded_ts_ce = models[3].decoder(encoded_ts_ce)
            decoded_ts_le = models[4].decoder(encoded_ts_le)
            decoded_n_rad = models[5].decoder(encoded_n_rad)

            inputs = [vd_inputs, vmd_inputs, ts_pe_inputs, ts_ce_inputs, ts_le_inputs, n_rad_inputs]
            decoded = [decoded_vd, decoded_vmd, decoded_ts_pe, decoded_ts_ce, decoded_ts_le, decoded_n_rad]

            concat_emb = torch.cat((encoded_vd.view(-1,1,2048).to(torch.float16), encoded_vmd.view(-1,1,2048).to(torch.float16), 
                                encoded_ts_pe.view(-1,1,2048).to(torch.float16), encoded_ts_ce.view(-1,1,2048).to(torch.float16), 
                                encoded_ts_le.view(-1,1,2048).to(torch.float16), encoded_n_rad.view(-1,1,2048).to(torch.float16)),
                                  dim=1).to(device)

            binary_logits, ternary_logits = custom_output(concat_emb, gemma) 

            loss_bce = custom_bce_loss(binary_logits, ternary_logits, labels.float(), loss_fns)
            loss_mse = custom_mse_loss(decoded, inputs, mse_loss, device)
            loss = loss_bce + beta*loss_mse

            probabilities, hard_preds, labels = output_to_label(binary_logits, ternary_logits, labels)

            per_class_preds, per_class_labels, acc_batch_avg = accuracy_score(probabilities, hard_preds, labels)

            val_acc_cum += acc_batch_avg
            val_loss_cum += loss.item()

    return val_loss_cum/len(val_loader), val_acc_cum/len(val_loader), per_class_preds, per_class_labels


def training_loop(models, optimizers, mse_loss, loss_fns, train_loader, val_loader, num_epochs, gemma, beta):
    print("Starting training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model in models:
        model.to(device)
    
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    #best_val_loss = float('inf')
    best_f1 = 0

    for epoch in range(1, num_epochs+1):
        model, train_loss, train_acc = train_epoch(models,
                                        optimizers,
                                        mse_loss,
                                        loss_fns,
                                        train_loader,
                                        device,
                                        gemma,
                                        beta)
        val_loss, val_acc, per_class_preds, per_class_labels = validate(models, mse_loss, loss_fns, val_loader, device, gemma, beta)
        
        f1_scores = []
        for i,class_preds in enumerate(per_class_preds):
            preds_arrays = [t.cpu().numpy() for t in class_preds]
            preds = np.concatenate(preds_arrays)

            labels_arrays = [t.cpu().numpy() for t in per_class_labels[i]]
            labels = np.concatenate(labels_arrays)
            mask = np.isnan(labels)
            masked_labels = labels[~mask]
            masked_preds = preds[~mask]
            f1_scores.append(metrics.f1_score(masked_labels, masked_preds, average='micro'))

        f1_avg = sum(f1_scores) / len(f1_scores)

        print(f"Epoch {epoch}/{num_epochs}: "
              f"Train loss: {sum(train_loss)/len(train_loss):.3f}, "
              f"Train acc.: {sum(train_acc)/len(train_acc):.3f}, "
              f"Val. loss: {val_loss:.3f}, "
              f"Val. acc.: {val_acc:.3f}")
        train_losses.append(sum(train_loss)/len(train_loss))
        train_accs.append(sum(train_acc)/len(train_acc))
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        folder = 'multiresult/test3'

        if f1_avg > best_f1:
            best_f1 = f1_avg
            torch.save(models[0], f"{folder}/vd.pth")
            torch.save(models[1], f"{folder}/vmd.pth")
            torch.save(models[2], f"{folder}/ts_pe.pth")
            torch.save(models[3], f"{folder}/ts_ce.pth")
            torch.save(models[4], f"{folder}/ts_le.pth")
            torch.save(models[5], f"{folder}/n_rad.pth")

        with open(f"{folder}/train_losses.pkl", 'wb') as f1:
            pickle.dump(train_losses, f1)

        with open(f"{folder}/train_accs.pkl", 'wb') as f2:
            pickle.dump(train_accs, f2)

        with open(f"{folder}/val_losses.pkl", 'wb') as f3:
            pickle.dump(val_losses, f3)

        with open(f"{folder}/val_accs.pkl", 'wb') as f4:
            pickle.dump(val_accs, f4)

    return model, train_losses, val_losses
