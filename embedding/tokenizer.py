import torch
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import EmbeddingModel

class Encoder:
    def __init__(self, model: str = "", device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.N_GPU = torch.cuda.device_count()
        self.DEVICE = device
        self.EmbeddingModel = self.load_model(model)
        self.embedding_cache = {}
        self.EMBEDDING_DIM = self.EmbeddingModel.info()["output_shape"][0]
        self.MAX_SEQ_LENGTH_IN_TOKENS = self.EmbeddingModel.info()["input_shape"]
        self.model_name = model + self.EmbeddingModel.info()["model_type"]

    def info(self):
        return {
            "N_GPU": self.N_GPU,
            "DEVICE": str(self.DEVICE),
            "model_name": self.model_name,
            "EMBEDDING_DIM": self.EMBEDDING_DIM,
            "MAX_SEQ_LENGTH_IN_TOKENS": self.MAX_SEQ_LENGTH_IN_TOKENS
        }

    def load_model(self, model_path):
        self.EmbeddingModel.load_state_dict(torch.load(model_path, map_location=self.DEVICE))
        self.EmbeddingModel.eval()

    def emb_data(self, data):
        data = data.unsqueeze(0).to(self.DEVICE)
        out = self.EmbeddingModel(data)
        out = out.detach().cpu().numpy()
        return out


#TODO:data和model分离，encode仅负责编码，不负责数据处理。可以提供batch编码和单独编码两种方式
    def normalize_data(self, all_data, norm_data):
        max_v, min_v = np.max(norm_data[:, 0]), np.min(norm_data[:, 0])
        max_i, min_i = np.max(norm_data[:, 1]), np.min(norm_data[:, 1])
        all_data[:, 0] = (all_data[:, 0] - min_v) / (max_v - min_v)
        all_data[:, 1] = (all_data[:, 1] - min_i) / (max_i - min_i)
        return all_data

    def encode_data(self, all_data):
        vecs = []
        for data in all_data:
            vecs.extend(self.extract_features(data))
        return np.array(vecs)

    def add_integral_dimensions(self, vecs, all_data, p1, p2, divide):
        new_vec = np.zeros((vecs.shape[0], vecs.shape[1] + 3))
        new_vec[:, :vecs.shape[1]] = vecs
        for i in range(vecs.shape[0]):
            new_vec[i, vecs.shape[1]] = divide * np.sum(all_data[i, :p1, 1])
            new_vec[i, vecs.shape[1] + 1] = divide * np.sum(all_data[i, p1:p2, 1])
            new_vec[i, vecs.shape[1] + 2] = divide * np.sum(all_data[i, p2:, 1])
        return new_vec

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

def load_data(data_path, labels):
    raw_data = loadmat(data_path)
    all_data, label = [], []
    for i in range(len(labels)):
        data = raw_data[labels[i]]
        for j in range(data.shape[1]):
            all_data.append(data[0, j][:, :2])
            label.append(i)
    return np.array(all_data), label

def train_knn_classifier(X_train, y_train, n_neighbors=4):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_model_path = r'model/nor_1001.pth'
    data_path = r'dataset/real_data.mat'
    labels = ['nor', 'sc', 'ps', 'ad', 'oc']

    encoder = Encoder()
    encoder.load_model(encoder_model_path)
    all_data, label = load_data(data_path, labels)  # 500*40*2, 500
    num = len(all_data) // 5
    norm_data = all_data[:num]
    for i in range(all_data.shape[0]):
        all_data[i] = encoder.normalize_data(all_data[i], norm_data[i % num])

    vecs = []
    for idx in range(len(all_data)):
        data = torch.tensor(all_data[idx], dtype=torch.float32)
        vecs.extend(encoder.extract_features(data))
    vecs = np.array(vecs)

    p1, p2, divide = 10, 25, 0.1  # divide*sum[0,p1],divide*[p1,p2],divide*[p2,40]
    new_vec = encoder.add_integral_dimensions(vecs, all_data, p1, p2, divide)

    X_train, X_test, y_train, y_test = train_test_split(new_vec, label, test_size=0.2)
    knn = train_knn_classifier(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"KNN分类器的准确度: {accuracy:.2f}")

    pca_visualization(X_train, y_train, 3)
    pca_visualization(X_test, y_test, 3)
    pca_visualization(X_test, y_pred, 3)

if __name__ == "__main__":
    main()