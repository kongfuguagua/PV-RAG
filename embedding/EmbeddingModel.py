import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,random_split, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import classification_report
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import math
import torch.optim as optim
import matplotlib.pyplot as plt

# 输入40*2的IV曲线，编码为5*1向量
class EmbeddingModel(nn.Module):
    def __init__(self):
        super(EmbeddingModel, self).__init__()
        def conv(input, output, ker_size=3, stride=1, pad=1):
            c = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input,             
                            out_channels=output,             
                            kernel_size=(ker_size,ker_size),          
                            stride=stride,                   
                            padding=pad),                 
            torch.nn.BatchNorm2d(output),                 
            torch.nn.ReLU()
            )
            return c 
        
        self.conv1 = conv(1, 3)
        self.conv2 = conv(3, 5)
        self.maxpool1 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2)   # 5*40*2 - 5*20*1
        )
        self.conv3 = conv(5, 8)
        self.conv4 = conv(8, 16)
        self.fc1 = torch.nn.Sequential(torch.nn.Linear(320,160))
        self.fc2 = torch.nn.Sequential(torch.nn.Linear(160,5))


    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.squeeze(x,dim=-1)
        x = self.fc1(x.view(x.size(0),-1))
        x = self.fc2(x)
        # x = F.tanh(x)
        return x
    
    def info(self):
        return {
            "input_shape": (40, 2),
            "output_shape": (5,),
            "model_type": "Convolutional Neural Network",
            "layers": [
                {"layer_type": "Conv2d", "in_channels": 1, "out_channels": 3, "kernel_size": 3, "stride": 1, "padding": 1},
                {"layer_type": "Conv2d", "in_channels": 3, "out_channels": 5, "kernel_size": 3, "stride": 1, "padding": 1},
                {"layer_type": "MaxPool2d", "kernel_size": 2, "stride": 2},
                {"layer_type": "Conv2d", "in_channels": 5, "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1},
                {"layer_type": "Conv2d", "in_channels": 8, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
                {"layer_type": "Linear", "in_features": 320, "out_features": 160},
                {"layer_type": "Linear", "in_features": 160, "out_features": 5}
            ]
        }
    

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, predictions, data):
        loss = 0
        i, v, irra, temp = data[:, :, 1], data[:, :, 0], data[:, 0, 2], data[:, 0, 3]
        batch_size = predictions.shape[0]
        T = temp 
        c, Is, n, Rs, Rp = predictions[:,0],predictions[:,1],predictions[:,2],predictions[:,3],predictions[:,4]
        Ip = c * irra   # 光电流与辐照度基本成正比

	# I = I_{ph} - I_{0}\left[\exp\left(\frac{V + IR_{s}}{nN_{s}kT/q}\right) - 1\right] - \frac{V + IR_{s}}{R_{sh}}

        exponent = qk * (v + Rs.unsqueeze(1) * i) / (n.unsqueeze(1) * T.unsqueeze(1)) # 指数项
        exponent = torch.clamp(exponent, max=16)
        loss_matrix = Ip.unsqueeze(1) - Is.unsqueeze(1) * (torch.exp(exponent) - 1) - (v + Rs.unsqueeze(1) * i) / Rp.unsqueeze(1) - i
        loss = torch.sum(abs(loss_matrix))
        return loss

def plot_loss_curve(total_loss):
    plt.plot(range(1, n_epochs + 1), total_loss, marker='o', color='b', label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

def train():
    total_loss = []
    lower_bound = torch.tensor([1,25e-6,25,1,100], dtype=torch.float32).to(device)
    upper_bound = torch.tensor([2,50e-6,50,2,2000], dtype=torch.float32).to(device)
    # 存在问题：未知光伏阵列型号以及串并联数量，未知参数范围
    # 原来Ip可能范围是[0,2]，转换为辐照度表达后未知

    loss_mse = torch.nn.MSELoss()
    for epoch in range(n_epochs):
        epoch_loss = 0
        Encoder.train()
        for _, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            en_input = data[:, :, :2]
            en_out = Encoder(en_input)
            # loss = loss_mse(en_out, low_bound)# + loss_mse(en_out, upper_bound)
            loss = loss_f(en_out, data)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                epoch_loss += loss.item()

        with torch.no_grad():
            avg_loss = epoch_loss / len(train_loader)
            total_loss.append(avg_loss)
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}')

        if (epoch + 1) % 100 == 0:
            torch.save(Encoder.state_dict(), f'F:\\task1\\rag\\model\\encoder_epoch_{epoch+1}.pth')

    plot_loss_curve(total_loss)

    

if __name__ == '__main__':
    n_epochs = 400
    batch_size = 10
    simu_data = loadmat(r'dataset/simu_data.mat')
    real_data = loadmat(r'dataset/real_data.mat') # 5*2201*40*5 5种，每种2201个样本，每个样本40个采样点，采样点5个特征【电压、电流、辐照、温度、label】
    q = 1.60217646e-19 # 电子电荷
    k = 1.3806503e-23 # 玻尔兹曼常数
    qk = q/k # 电荷与玻尔兹曼常数之比
    lr = 3e-3
    # 制作数据集
    simu_dataset = []
    nor_data = simu_data['nor'] # 1*2201*40*5，
    d = np.zeros((nor_data.shape[1], 40, 5)) # 2201*40*5
    for j in range(nor_data.shape[1]): # 2201，对电压、电流自归一化
        d[j] = nor_data[0][j] # 40*5
        max_v, min_v = np.max(d[j, :, 0]), np.min(d[j, :, 0])
        max_i, min_i = np.max(d[j, :, 1]), np.min(d[j, :, 1])
        d[j, :, 0] = (d[j, :, 0] - min_v) / (max_v - min_v)
        d[j, :, 1] = (d[j, :, 1] - min_i) / (max_i - min_i)
    # max_vals, min_vals = [0, 0, 0, 0], [100000, 100000, 10000, 10000]
    # for i in range(4):
    #     max_vals[i] = np.max(d[:, :, i])
    #     min_vals[i] = np.min(d[:, :, i])

    d[:, :, 3] += 273.15
    # for i in range(4):
    #     d[:, :, i] = (d[:, :, i] - min_vals[i]) / (max_vals[i] - min_vals[i])


    dataset = np.array(d, dtype=np.float32) 
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Encoder = EmbeddingModel().to(device)
    # 使用预训练的模型
    # Encoder.load_state_dict(torch.load(r'F:\task1\rag\model\pretrained.pth', map_location=device))
    optimizer = optim.Adam(Encoder.parameters(), lr=lr)
    loss_f = Loss()

    train()
