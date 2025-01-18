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
from torch.utils.data import Dataset


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

def load_encoder(model_path, device):
    Encoder = encoder()
    Encoder.load_state_dict(torch.load(model_path, map_location=device))
    Encoder.eval()
    return Encoder

def load_data(data_path):
    simu_data = loadmat(data_path)
    lab = ['nor', 'open_circuit', 'partial_shading1', 'partial_shading2', 'partial_shading3', 'short_circuit1', 'short_circuit2']
    lab1 = [0, 1, 2, 2, 2, 3, 3]
    all_data, label = [], []
    for i in range(7):
        data = simu_data['sample_'+lab[i]]
        for j in range(data.shape[1]):
            all_data.append(data[0, j][:, :2])
            label.append(lab1[i])
    all_data = np.array(all_data)
    return all_data, label

def normalize_data(all_data, norm_data):#all_data: 40*2, norm_data: 40*2
    max_v, min_v = np.max(norm_data[ :, 0]), np.min(norm_data[ :, 0])
    max_i, min_i = np.max(norm_data[ :, 1]), np.min(norm_data[ :, 1])
    all_data[ :, 0] = (all_data[ :, 0] - min_v) / (max_v - min_v)
    all_data[ :, 1] = (all_data[ :, 1] - min_i) / (max_i - min_i)
    return all_data

class myDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.int16)

def extract_features(encoder, data, device):
    data = data.unsqueeze(0).to(device)
    out = encoder(data)
    out = out.detach().cpu().numpy()
    return out

def add_integral_dimensions(vecs, all_data, p1=15, p2=25, divide=0.1):
    new_vec = np.zeros((vecs.shape[0], vecs.shape[1]+3))
    new_vec[:vecs.shape[0], :5] = vecs
    for i in range(vecs.shape[0]):
        new_vec[i, 5] = divide * np.sum(all_data[i, :p1, 1])
        new_vec[i, 6] = divide * np.sum(all_data[i, p1:p2, 1])
        new_vec[i, 7] = divide * np.sum(all_data[i, p2:, 1])
    return new_vec

def train_knn(X_train, y_train, n_neighbors=4):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Encoder = load_encoder(r'model/nor_1001.pth', device)
    all_data, label = load_data(r'dataset/all_sample.mat') # 1547*40*2, 1547
    num = len(all_data) // 7 # 221
    norm_data= all_data[:num]
    for i in range(all_data.shape[0]):
        all_data[i] = normalize_data(all_data[i],norm_data[i%num])
    vecs=[]
    for idx in range(len(all_data)):
        data=torch.tensor(all_data[idx], dtype=torch.float32)
        vecs.extend(extract_features(Encoder, data, device))
    vecs = np.array(vecs)
    print(f'0:{vecs[10:11, :]}\n1:{vecs[110:111, :]}\n2:{vecs[210:211, :]}\n3:{vecs[310:311, :]}\n4:{vecs[410:411, :]}')

    new_vec = add_integral_dimensions(vecs, all_data)
    X_train, X_test, y_train, y_test = train_test_split(new_vec, label, test_size=0.2)
    
    knn = train_knn(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"KNN分类器的准确度: {accuracy:.2f}")

    pca(X_train, y_train, 3)
    pca(X_test, y_test, 3)
    pca(X_test, y_pred, 3)
    # tsne(X_test, y_pred, 3)

if __name__ == "__main__":
    main()
