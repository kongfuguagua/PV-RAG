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


def pca_visualization(X, Y, n=2):
    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X)
    if n == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y, cmap='viridis')
        plt.colorbar()
        plt.title('PCA')
        plt.xlabel('PCA first')
        plt.ylabel('PCA second')
        plt.show()
    elif n == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=Y, cmap='viridis')
        ax.set_xlabel('PCA1')
        ax.set_ylabel('PCA2')
        ax.set_zlabel('PCA3')
        ax.set_title('PCA 3D Visualization')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Class Labels')
        plt.show()


def tsne_visualization(X, Y, n=2):
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


def load_data(data_path, labels):
    simu_data = loadmat(data_path)
    all_data, label = [], []
    for i in range(len(labels)):
        data = simu_data[labels[i]]
        for j in range(data.shape[1]):
            all_data.append(data[0, j][:, :2])
            label.append(i)
    return np.array(all_data), label


def self_normalize_data(all_data, num):
    for i in range(all_data.shape[0]):
        max_v, min_v = np.max(all_data[i % num, :, 0]), np.min(all_data[i % num, :, 0])
        max_i, min_i = np.max(all_data[i % num, :, 1]), np.min(all_data[i % num, :, 1])
        all_data[i, :, 0] = (all_data[i, :, 0] - min_v) / (max_v - min_v)
        all_data[i, :, 1] = (all_data[i, :, 1] - min_i) / (max_i - min_i)
    return all_data


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.int16)


def create_dataloader(data, labels, batch_size):
    dataset = MyDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def encode_data(encoder, dataloader, device):
    vecs = []
    for batch_data, _ in dataloader:
        batch_data = batch_data.to(device)
        out = encoder(batch_data) # 440*100
        out = out.detach().cpu().numpy()
        vecs.extend(out) # 440*100
    return np.array(vecs)


def add_integral_dimensions(vecs, all_data, p1, p2, divide):
    new_vec = np.zeros((vecs.shape[0], vecs.shape[1] + 3))
    new_vec[:, :vecs.shape[1]] = vecs
    for i in range(vecs.shape[0]):
        new_vec[i, vecs.shape[1]] = divide * np.sum(all_data[i, :p1, 1])
        new_vec[i, vecs.shape[1] + 1] = divide * np.sum(all_data[i, p1:p2, 1])
        new_vec[i, vecs.shape[1] + 2] = divide * np.sum(all_data[i, p2:, 1])
    return new_vec


def train_knn_classifier(X_train, y_train, n_neighbors=4):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_model_path = r'model/nor_1001.pth'
    data_path = r'dataset/real_data.mat'
    labels = ['nor', 'sc', 'ps', 'ad', 'oc']

    Encoder = load_encoder(encoder_model_path, device)
    all_data, label = load_data(data_path, labels) # 500*40*2, 500
    num = len(all_data) // 5 
    all_data = self_normalize_data(all_data, num)

    batch_size = 1
    dataloader = create_dataloader(all_data, label, batch_size)
    vecs = encode_data(Encoder, dataloader, device) # 500*5

    print(f'0:{vecs[10:11, :]}\n1:{vecs[110:111, :]}\n2:{vecs[210:211, :]}\n3:{vecs[310:311, :]}\n4:{vecs[410:411, :]}')

    #按区间重新编码（图像机理）
    p1, p2, divide = 10, 25, 0.1 # divide*sum[0,p1],divide*[p1,p2],divide*[p2,40]
    new_vec = add_integral_dimensions(vecs, all_data, p1, p2, divide)


    X_train, X_test, y_train, y_test = train_test_split(new_vec, label, test_size=0.2)
    knn = train_knn_classifier(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"KNN分类器的准确度: {accuracy:.2f}")

    pca_visualization(X_train, y_train, 3)
    pca_visualization(X_test, y_test, 3)
    pca_visualization(X_test, y_pred, 3)
    # tsne_visualization(X_test, y_pred, 3)


if __name__ == "__main__":
    main()