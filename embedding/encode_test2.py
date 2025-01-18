import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from encoder import encoder
from torch.utils.data import Dataset, DataLoader


def pca(X, Y, n=2):  
    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X)
    if n == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y, cmap='viridis')
        plt.colorbar()
        plt.title('KMeans')
        plt.xlabel('PCA first')
        plt.ylabel('PCA second')
        plt.show()
    elif n == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=Y, cmap='viridis')
        ax.set_xlabel('pca1')
        ax.set_ylabel('pca2')
        ax.set_zlabel('pca3')
        ax.set_title('PCA 3D Visualization')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Class Labels')
        plt.show()

def tsne(X, Y, n=2):
    tsne = TSNE(n_components=n)
    X_tsne = tsne.fit_transform(X)
    if n == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y, cmap='viridis')
        plt.colorbar()
        plt.title('t-SNE')
        plt.xlabel('First component')
        plt.ylabel('Second component')
        plt.show()
    elif n == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=Y, cmap='viridis')
        ax.set_xlabel('t-SNE1')
        ax.set_ylabel('t-SNE2')
        ax.set_zlabel('t-SNE3')
        ax.set_title('t-SNE 3D Visualization')
        plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Encoder = encoder()
Encoder.load_state_dict(torch.load(r'F:\task1\rag\model\nor_1001.pth', map_location=device))
Encoder.eval()

simu_data = loadmat(r'F:\task1\dataset\all_sample.mat')
lab = ['nor', 'open_circuit', 'partial_shading1', 'partial_shading2', 'partial_shading3', 'short_circuit1', 'short_circuit2']
lab1 = [0, 1, 2, 2, 2, 3, 3]
all_data, label = [], []
for i in range(7):
    data = simu_data['sample_'+lab[i]]
    for j in range(data.shape[1]):
        all_data.append(data[0, j][:, :2])
        label.append(lab1[i])
all_data = np.array(all_data)

num = len(all_data)//7
# 自归一化(用正常样本的相同条件)
for i in range(all_data.shape[0]):
    max_v, min_v = np.max(all_data[i%num, :, 0]), np.min(all_data[i%num, :, 0])
    max_i, min_i = np.max(all_data[i%num, :, 1]), np.min(all_data[i%num, :, 1])
    all_data[i, :, 0] = (all_data[i, :, 0] - min_v) / (max_v - min_v)
    all_data[i, :, 1] = (all_data[i, :, 1] - min_i) / (max_i - min_i)

class myDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.int16)

dataset = myDataset(all_data, label)
batch_size = 440  
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

vecs = []
for idx, (batch_data, batch_label) in enumerate(dataloader):
    batch_data.to(device)
    out = Encoder(batch_data)
    out = out.detach().cpu().numpy()
    vecs.extend(out)
vecs = np.array(vecs)
print(f'0:{vecs[10:11, :]}\n1:{vecs[110:111, :]}\n2:{vecs[210:211, :]}\n3:{vecs[310:311, :]}\n4:{vecs[410:411, :]}')
# print(f'0:{vecs[10:11, :]}\n1:{vecs[2211:2212, :]}\n2:{vecs[4412:4413, :]}\n3:{vecs[6613:6614, :]}\n4:{vecs[8814:8815, :]}')

# 加三个积分维度
p1, p2, divide = 15, 25, 0.1
new_vec = np.zeros((vecs.shape[0], vecs.shape[1]+3))
new_vec[:vecs.shape[0], :5] = vecs
for i in range(vecs.shape[0]):
    new_vec[i, 5] = divide * np.sum(all_data[i, :p1, 1])
    new_vec[i, 6] = divide * np.sum(all_data[i, p1:p2, 1])
    new_vec[i, 7] = divide * np.sum(all_data[i, p2:, 1])

# 测试直接用差值
# new_vec = np.zeros((vecs.shape[0], 40))
# for i in range(all_data.shape[0]):
#     for j in range(40):
#         new_vec[i, j] = all_data[i%num, j, 1] - all_data[i, j, 1]


# scaler = StandardScaler()
# vecs = scaler.fit_transform(vecs)
# new_vec = scaler.fit_transform(new_vec)
X_train, X_test, y_train, y_test = train_test_split(new_vec, label, test_size=0.2)
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN分类器的准确度: {accuracy:.2f}")
pca(X_train, y_train, 3)
pca(X_test, y_test, 3)
pca(X_test, y_pred, 3)
# tsne(X_test, y_pred, 3)