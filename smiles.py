import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from rdkit import Chem
from dataloader.image_dataloader import SmilesDataset
from torch.utils.data import DataLoader
from model.model import SMILESEncoder


# 数据预处理
def smiles_to_onehot(smiles, max_len, charset):
    onehot = torch.zeros((len(smiles), max_len, len(charset)), dtype=float)
    for i, s in enumerate(smiles):
        for j, c in enumerate(s):
            onehot[i, j, charset.index(c)] = 1
    return onehot

# 构建LSTM模型
class LSTMChem(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMChem, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def main():
    dir = './data/MPP/classification/bbbp/processed/bbbp_processed_ac.csv'
    batchsize = 256
    epochs = 50
    lr = 0.01
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    print("train device:", device)
    # 读取SMILES数据
    smiles_df = pd.read_csv(dir)
    smiles_list = smiles_df['smiles'].tolist()
    charset = list(set(''.join(smiles_list)))  
    max_len = max([len(s) for s in smiles_list])
    # 对SMILES进行onehot处理
    one_hot = smiles_to_onehot(smiles_list, max_len, charset)
    # 对label进行onehot处理
    y_one_hot = F.one_hot(torch.tensor(smiles_df['label'].values, dtype=int), num_classes = 2)
    # 构建自定义数据集
    dataset = SmilesDataset(one_hot, y_one_hot)
    train_dataloader = DataLoader(dataset, 
                                  batch_size=batchsize, 
                                  shuffle=True)
    

    # 读取模型
    smiles_encoder = LSTMChem(len(charset), 128, 2, 2)
    smiles_encoder.to(device)
    # 损失函数、优化器
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(smiles_encoder.parameters(), lr=lr)
    for epoch in range(epochs):
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            x = x.float()
            y = y.float()
            optimizer.zero_grad()
            outputs = smiles_encoder(x)
            loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()
            # clear GPU memery
            torch.cuda.empty_cache()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))
    print("Done!")         
   

if __name__ == "__main__":
    main()
