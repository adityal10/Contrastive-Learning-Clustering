# File: contrastive_learning.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Data Augmentation: SimCLR-like strong augmentations
class SimCLRTransform:
    def __init__(self):
        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __call__(self, x):
        return self.augment(x), self.augment(x)


# NT-Xent Loss (Contrastive Loss)
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)

        # Normalize embeddings
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)

        # Compute similarity matrix
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T)

        # Remove self-similarity
        mask = torch.eye(2 * batch_size, device=similarity_matrix.device).bool()
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)

        # Positive similarity
        pos_sim = torch.cat([torch.diag(similarity_matrix, batch_size),
                             torch.diag(similarity_matrix, -batch_size)])

        # Labels
        labels = torch.arange(batch_size, device=z_i.device)
        labels = torch.cat([labels, labels])

        # Scale by temperature and compute loss
        similarity_matrix /= self.temperature
        return self.criterion(similarity_matrix, labels)


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

    def forward(self, x):
        h = self.backbone(x)  # Extract features
        z = self.projection_head(h)  # Project to latent space
        return z


# Training Loop
def train_simclr():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # DataLoader
    transform = SimCLRTransform()
    dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    # Model
    backbone = models.resnet18(pretrained=False)
    model = SimCLR(backbone, projection_dim=128).to(device)
    criterion = NTXentLoss(temperature=0.5)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # Training
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for (x_i, x_j), _ in dataloader:
            x_i, x_j = x_i.to(device), x_j.to(device)

            # Forward pass
            z_i = model(x_i)
            z_j = model(x_j)

            # Compute loss
            loss = criterion(z_i, z_j)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

    # Save model
    torch.save(model.state_dict(), 'simclr_model.pth')
    print("Model saved as 'simclr_model.pth'")


if __name__ == "__main__":
    train_simclr()
