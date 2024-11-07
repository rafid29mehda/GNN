The paper does not mention an available code repository for Heart-Net. However, I can guide you on building a simplified version of Heart-Net using 3D U-Net for MRI data, a Temporal Convolutional Graph Neural Network (TCGN) for ECG data, and integrating both with an attention mechanism for classification.

Below is a comprehensive structure to implement the **Heart-Net model** in Google Colab:

1. **Dependencies**:
   - Install essential libraries such as PyTorch, TorchVision, and related deep learning packages.

2. **Dataset Preparation**:
   - This code assumes you have MRI and ECG data in compatible formats. If you don't have actual data, you can simulate data for testing.

3. **Define the Heart-Net Architecture**:
   - Implement the 3D U-Net for MRI.
   - Implement the Temporal Convolutional Graph Neural Network (TCGN) for ECG.
   - Fuse features from both networks using an attention mechanism.
   - Implement the final classifier for multi-class output.

Here's the code outline:

```python
# Step 1: Install required packages
!pip install torch torchvision torchaudio torch-geometric

# Step 2: Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Step 3: Simulate datasets for demonstration
# Simulate MRI dataset (3D data) and ECG dataset (1D sequential data)
class MRIDataset(Dataset):
    def __init__(self, num_samples=100, shape=(1, 32, 32, 32)):
        self.data = torch.rand(num_samples, *shape)
        self.labels = torch.randint(0, 2, (num_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class ECGDataset(Dataset):
    def __init__(self, num_samples=100, seq_len=256):
        self.data = torch.rand(num_samples, seq_len)
        self.labels = torch.randint(0, 2, (num_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Step 4: Define the 3D U-Net model for MRI feature extraction
class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv3d(16, 32, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose3d(16, 1, 3, padding=1), nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Step 5: Define the TCGN model for ECG feature extraction
class TCGN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2):
        super(TCGN, self).__init__()
        self.convs = nn.ModuleList([GCNConv(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        return global_mean_pool(x, batch=torch.zeros(x.size(0), dtype=torch.long))  # Global pooling for simplicity

# Step 6: Attention mechanism to combine MRI and ECG features
class AttentionFusion(nn.Module):
    def __init__(self, mri_dim, ecg_dim):
        super(AttentionFusion, self).__init__()
        self.attention = nn.Linear(mri_dim + ecg_dim, 1)

    def forward(self, mri_features, ecg_features):
        combined = torch.cat([mri_features, ecg_features], dim=1)
        weights = torch.sigmoid(self.attention(combined))
        return weights * mri_features + (1 - weights) * ecg_features

# Step 7: Define the final Heart-Net model
class HeartNet(nn.Module):
    def __init__(self):
        super(HeartNet, self).__init__()
        self.unet3d = UNet3D()
        self.tcgn = TCGN()
        self.fusion = AttentionFusion(mri_dim=32*32*32, ecg_dim=64)
        self.classifier = nn.Linear(32*32*32, 2)

    def forward(self, mri, ecg, ecg_edge_index):
        mri_features = self.unet3d(mri).view(mri.size(0), -1)
        ecg_features = self.tcgn(ecg, ecg_edge_index)
        fused_features = self.fusion(mri_features, ecg_features)
        return self.classifier(fused_features)

# Step 8: Train the model on the simulated dataset
def train_heartnet():
    # Data
    mri_data = MRIDataset()
    ecg_data = ECGDataset()
    loader = DataLoader(mri_data, batch_size=2, shuffle=True)
    
    # Model
    model = HeartNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        for (mri, label), (ecg, _) in zip(loader, loader):
            optimizer.zero_grad()
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # Dummy edge index for simplicity
            output = model(mri, ecg.unsqueeze(1), edge_index)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Run training
train_heartnet()
```

### Explanation of the Code Components:

1. **Data Simulation**: 
   - **MRIDataset** and **ECGDataset** simulate MRI and ECG data, respectively, to demonstrate functionality without actual data.

2. **UNet3D**:
   - Implements a simple 3D U-Net to extract MRI features by encoding and decoding spatial details.

3. **TCGN**:
   - Defines a simple Temporal Convolutional Graph Neural Network for ECG data, using `GCNConv` layers and global mean pooling to handle ECG graph data.

4. **AttentionFusion**:
   - Fuses the MRI and ECG features through an attention mechanism to weigh relevant features.

5. **HeartNet**:
   - Combines the modules (UNet3D, TCGN, and AttentionFusion) and a classifier layer for final classification.

6. **Training Script**:
   - Demonstrates a basic training loop on the simulated dataset using cross-entropy loss. The model parameters are updated via backpropagation.

### Running in Colab

To run this code:
1. Copy it into a new Colab notebook.
2. Run each cell sequentially.
3. Replace simulated data with actual MRI and ECG data (structured as 3D images for MRI and sequence graphs for ECG) for real-world use.

This code provides a starting point. Fine-tuning, hyperparameter adjustments, and more complex data handling are needed for application on actual datasets.
