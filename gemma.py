from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm

# Filepath to embeddings
fname = "/mnt/mimic/data/HAIM/mimic_extras/embeddings.csv"

# Loading model
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto", quantization_config=quantization_config)

# Model
class MLPModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# Get and preprocess data
df = pd.read_csv(fname)
df_death_small48 = df[((df['img_length_of_stay'] < 48) & (df['death_status'] == 1))]
df_alive_big48 = df[((df['img_length_of_stay'] >= 48) & (df['death_status'] == 0))]
df_death_big48 = df[((df['img_length_of_stay'] >= 48) & (df['death_status'] == 1))]

df_death_small48['y'] = 1
df_alive_big48['y'] = 0
df_death_big48['y'] = 0
df = pd.concat([df_death_small48, df_alive_big48, df_death_big48], axis = 0)

vd_cols = df.filter(regex='^vd_')
y_col = df[['y']]
haim_col = df[['haim_id']]
df = pd.concat([haim_col, vd_cols, y_col], axis=1)
print(df.head())

input_embeddings = torch.tensor(df.iloc[:, 1:1025].values, dtype=torch.float32)


# Transform embeddings into embeddingspace of gemma
projection_model = MLPModel(input_size=1024, output_size=250).cuda()

result_embeddings = []

for emb in tqdm(input_embeddings, desc="Processing embeddings", unit="embeddings"):
    emb = emb.cuda()
    output_tokens = projection_model(emb)
    normalized_output = torch.sigmoid(output_tokens)
    scaled_output = (normalized_output * 350) + 255649

    rounded_output = torch.round(scaled_output)
    result_embeddings.append(rounded_output)


# Done
transformed_embeddings = torch.stack(result_embeddings, dim=0)

print(transformed_embeddings)