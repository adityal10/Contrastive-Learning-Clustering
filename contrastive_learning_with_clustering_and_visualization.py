# File: contrastive_learning_with_clustering_and_visualization.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import matplotlib.pyplot as plt
import numpy as np

# Data Augmentation: Minimal augmentation for evaluation
class SimpleTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __call__(self, x):
        return self.transform(x)


# SimCLR Model 
class SimCLR(nn.Module):
    def __init__(self, backbone, projection_dim):
        super(SimCLR, self).__init__()
        self.backbone = backbone
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove classification head

        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x, projection=True):
        h = self.backbone(x)  # Extract features
        if projection:
            z = self.projection_head(h)  # Project to latent space
            return z
        return h  # Return raw features for clustering


# Load Trained Model
def load_model():
    backbone = models.resnet18(pretrained=False)
    model = SimCLR(backbone, projection_dim=128)
    model.load_state_dict(torch.load('simclr_model.pth'))
    model.eval()
    return model


# Clustering and Visualization
def visualize_clusters(model, dataloader, num_clusters=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Extract embeddings
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            features = model(images, projection=False)  # Use raw features
            embeddings.append(features.cpu().numpy())
            labels.append(targets.numpy())

    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    predicted_labels = kmeans.fit_predict(embeddings)

    # Evaluate Clustering
    nmi = normalized_mutual_info_score(labels, predicted_labels)
    ari = adjusted_rand_score(labels, predicted_labels)

    print(f"Clustering Performance:")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")

    # Dimensionality Reduction with t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Plot Clusters
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=predicted_labels, cmap='tab10', s=5)
    plt.colorbar(scatter, label='Cluster Label')
    plt.title('t-SNE Visualization of Clusters')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig('clusters_visualization.png')
    plt.show()

    print("Cluster visualization saved as 'clusters_visualization.png'")


if __name__ == "__main__":
    # Load Data
    transform = SimpleTransform()
    dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

    # Load Trained Model
    model = load_model()

    # Visualize Clusters
    visualize_clusters(model, dataloader, num_clusters=10)