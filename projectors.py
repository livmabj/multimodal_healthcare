import torch
import torch.nn as nn


EMBEDDING_SIZE = 1024
PROJECTION_SIZE = 32


class vdProjectionNN(nn.Module):
    def __init__(self):
        super(vdProjectionNN, self).__init__()

        self.bn1 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024,1024)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.15)

        #self.fc2 = nn.Linear(1024,1024)
        #self.ln2 = nn.LayerNorm(1024)
        #self.relu2 = nn.ReLU()
        #self.drop2 = nn.Dropout(p=0.15)

        self.fc4 = nn.Linear(1024,512)
        self.ln4 = nn.LayerNorm(512)
        self.relu4 = nn.ReLU()

        self.fc5 = nn.Linear(512,2)

        self.flatten = nn.Flatten()
    
    def forward(self,x):
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.relu1(x)

        #x = self.fc2(x)
        #x = self.ln2(x)
        #x = self.relu2(x)
        #x = self.drop2(x)

        x = self.fc4(x)
        x = self.ln4(x)
        x = self.relu4(x)

        x = self.fc5(x)
        x = self.flatten(x)
        return x