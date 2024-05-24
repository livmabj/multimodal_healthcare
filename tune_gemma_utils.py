import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pickle
from sklearn import metrics




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



class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-6, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Mask NaN and 2 labels
        mask = ~torch.isnan(y) & (y != 2)
        x = x[mask]
        y = y[mask]

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w


        return -loss.sum()








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
        

class Adapter(nn.Module):
    def __init__(self, size = 6, model_dim = 2048):
        super().__init__()
        self.adapter_block = nn.Sequential(
            nn.Linear(model_dim, size),
            nn.ReLU(),
            nn.Linear(size, model_dim)
        )

    def forward(self, x):

        output = self.adapter_block(x)
        adapter_out = output + x

        return adapter_out


class Adaptered(nn.Module):
    def __init__(self, orig_layer):
        super().__init__()
        self.orig_layer = orig_layer
        self.adapter = Adapter()

    def forward(self, *x):
        orig_out = self.orig_layer(*x)
        output = (self.adapter.forward(orig_out[0].unsqueeze(0))[0],)

        return output



class model_with_adapter(nn.Module):

    def __init__(self):
        super().__init__()
        self.quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        self.model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto", quantization_config=self.quantization_config,attn_implementation="sdpa")
        # Freeze the original model parameters
        for params in self.model.parameters():
            params.requires_grad = False
        # Embed adapter layers into the transformer blocks 
        for i, gemma_layer in enumerate(self.model.model.layers):
            gemma_layer.self_attn.q_proj = Adaptered(gemma_layer.self_attn.q_proj)
            gemma_layer.self_attn.k_proj = Adaptered(gemma_layer.self_attn.k_proj)
            gemma_layer.self_attn.v_proj = Adaptered(gemma_layer.self_attn.v_proj)
            gemma_layer.self_attn.o_proj = Adaptered(gemma_layer.self_attn.o_proj)
    
            gemma_layer.mlp.gate_proj = Adaptered(gemma_layer.mlp.gate_proj)
            gemma_layer.mlp.up_proj = Adaptered(gemma_layer.mlp.up_proj)
            gemma_layer.mlp.down_proj = Adaptered(gemma_layer.mlp.down_proj)

    def get_model(self):

        return self.model


####
####
####
####                                                                                HELPER FUNCTIONS
####
####
####
def custom_output(emb, gemma):
    if torch.isnan(emb).any():
        print('emb',emb)
    outputs = gemma(inputs_embeds=emb)
    class_labels = [101, 201, 301, 401, 501, 601, 701, 801, 901, 1001, 1101, 1201] 
    logits = outputs['logits']
    if torch.isnan(logits[:,-6:,class_labels]).any():
        print('logits_outputs',logits[:,-6:,class_labels])
    logits = logits[:,-6:,class_labels].mean(dim=1).float()
    return logits



def custom_mse_loss(decoded, inputs, mse_loss):

    vd_loss = mse_loss(decoded[0], inputs[0])
    vmd_loss = mse_loss(decoded[1], inputs[1])
    ts_pe_loss = mse_loss(decoded[2], inputs[2])
    ts_ce_loss = mse_loss(decoded[3], inputs[3])
    ts_le_loss = mse_loss(decoded[4], inputs[4])
    n_rad_loss = mse_loss(decoded[5], inputs[5])

    avg_mse = (vd_loss + vmd_loss + ts_pe_loss + ts_ce_loss + ts_le_loss + n_rad_loss) / 6

    return avg_mse


def output_to_label(logits, labels):

    probs_tensor_pos = F.sigmoid(logits)

    pred = torch.round(probs_tensor_pos)

    probs_tensor_neg = 1-probs_tensor_pos

    prob = torch.stack((probs_tensor_neg, probs_tensor_pos), dim=1)


    return prob, pred, labels






####
####
####
####                                                                                TRAINING LOOPS
####
####
####
def train_epoch(models, loss_mse, optimizer_llm, optimizers, train_loader, device, gemma, beta, assymetric_loss):
    # Train:
    gemma.train()
    train_loss_batches = []
    torch.autograd.set_detect_anomaly(True)
    for batch_index, (x, y) in enumerate(train_loader, 1):
        inputs, labels = x, y.to(device)

        optimizer_llm.zero_grad()
        #for optimizer in optimizers:
        #    optimizer.zero_grad()


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

        logits = custom_output(concat_emb, gemma) 

        loss_assym = assymetric_loss.forward(logits, labels.float())

        loss_assym.backward(retain_graph =True)

        torch.nn.utils.clip_grad_norm_(gemma.parameters(), max_norm=1.0)
        optimizer_llm.step()

        train_loss_batches.append(loss_assym.item())

    return gemma, train_loss_batches


def validate(models, loss_mse, val_loader, device, gemma, assymetric_loss):
    val_loss_cum = 0
    val_acc_cum = 0
    per_class_probs = [[] for _ in range(12)]
    per_class_preds = [[] for _ in range(12)]
    per_class_labels = [[] for _ in range(12)]
    #for model in models:
    #    model.eval()
    gemma.eval()
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

            logits = custom_output(concat_emb, gemma) 

            loss_assym = assymetric_loss.forward(logits, labels.float())

            val_loss_cum += loss_assym.item()

    return val_loss_cum/len(val_loader) #, val_acc_cum/len(val_loader), per_class_preds, per_class_labels


def training_loop(models, loss_mse, optimizer_llm, optimizers, train_loader, val_loader, num_epochs, gemma, beta, assymetric_loss, scheduler):
    print("Starting training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model in models:
        model.to(device)
    train_losses, val_losses, val_accs = [], [], []
    best_val_loss = float('inf')
    #best_f1 = 0

    for epoch in range(1, num_epochs+1):
        gemma, train_loss = train_epoch(models, loss_mse,
                                        optimizer_llm, optimizers,
                                        train_loader,
                                        device,
                                        gemma, beta,
                                        assymetric_loss)
        val_loss = validate(models, loss_mse, val_loader, device, gemma, assymetric_loss)
        
        print(f"Epoch {epoch}/{num_epochs}: "
              f"Train loss: {sum(train_loss)/len(train_loss):.3f}, "
              f"Val. loss: {val_loss:.3f} ")
              #f"Val. acc.: {val_acc:.3f}")
        train_losses.append(sum(train_loss)/len(train_loss))
        val_losses.append(val_loss)
        scheduler.step()

        folder = 'multiresult/moregemma'

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(gemma.state_dict(), f"{folder}/weights_only.pth")
            gemma.save_pretrained(f"{folder}/tuned_gemma.pth")

        with open(f"{folder}/train_losses.pkl", 'wb') as f1:
            pickle.dump(train_losses, f1)

        with open(f"{folder}/val_losses.pkl", 'wb') as f2:
            pickle.dump(val_losses, f2)

    return gemma, train_losses, val_losses


def predict(models, val_loader, gemma):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_class_probs = [[] for _ in range(12)]
    per_class_preds = [[] for _ in range(12)]
    per_class_labels = [[] for _ in range(12)]

    for model in models:
        model.eval()
    gemma.eval()
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

            logits = custom_output(concat_emb, gemma)

            probabilities, hard_preds, labels = output_to_label(logits, labels)


            for i in range(probabilities.size(2)):
                class_prob = probabilities[:, :, i]  # Select probabilities for class i
                per_class_probs[i].append(probabilities[:, :, i])
                per_class_preds[i].append(hard_preds[:, i])
                per_class_labels[i].append(labels[:, i])


    return per_class_preds, per_class_labels, per_class_probs