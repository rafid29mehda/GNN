To provide a full implementation of the Heart-Net model, based on the paper sections you've shared, I will outline the following components:

1. **Data Preprocessing for MRI and ECG Data**: This includes normalization, resizing for MRI, and filtering and feature extraction for ECG.
2. **3D U-Net for MRI Feature Extraction**: Builds a 3D U-Net model to process MRI images.
3. **TCGNN for ECG Feature Extraction**: Implements the Temporal Convolutional Graph Neural Network to process ECG data.
4. **Hybrid Diagnosis Layer**: Combines the features from MRI and ECG for the final classification.
5. **Model Training and Evaluation**: Configures the training process, loss functions, metrics (precision, recall, F1 score, accuracy), and evaluation using cross-validation.

I will implement these components step-by-step in Python, compatible with PyTorch, which you can run in Google Colab.

---

### 1. **Data Preprocessing**

#### a) MRI Data Preprocessing

```python
import numpy as np
import torch
import torch.nn.functional as F
from skimage.transform import resize

def preprocess_mri(mri_image, target_shape=(32, 256, 256)):
    # Normalize the MRI image to range [0, 1]
    mri_normalized = (mri_image - mri_image.min()) / (mri_image.max() - mri_image.min())
    
    # Resize to target shape
    mri_resized = resize(mri_normalized, target_shape, mode='constant', anti_aliasing=True)
    
    # Convert to torch tensor
    mri_tensor = torch.tensor(mri_resized, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
    return mri_tensor
```

#### b) ECG Data Preprocessing

```python
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=1000.0, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def preprocess_ecg(ecg_signal, segment_length=256):
    # Apply bandpass filter
    ecg_filtered = bandpass_filter(ecg_signal)
    
    # Segment and normalize ECG signal
    scaler = StandardScaler()
    ecg_scaled = scaler.fit_transform(ecg_filtered.reshape(-1, 1)).reshape(-1)
    
    # Convert to tensor and reshape
    ecg_tensor = torch.tensor(ecg_scaled[:segment_length], dtype=torch.float32).unsqueeze(0)  # Add channel dimension
    return ecg_tensor
```

---

### 2. **3D U-Net for MRI Feature Extraction**

```python
import torch
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()
        
        # Encoder layers
        self.enc1 = self.conv_block(1, 16)
        self.enc2 = self.conv_block(16, 32)
        self.enc3 = self.conv_block(32, 64)
        self.pool = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(64, 128)
        
        # Decoder layers
        self.upconv3 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.dec3 = self.conv_block(128, 64)
        self.upconv2 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec2 = self.conv_block(64, 32)
        self.upconv1 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.dec1 = self.conv_block(32, 16)
        
        # Final layer
        self.final_conv = nn.Conv3d(16, 1, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.pool(x1)
        x2 = self.enc2(x2)
        x3 = self.pool(x2)
        x3 = self.enc3(x3)
        
        # Bottleneck
        x4 = self.pool(x3)
        x4 = self.bottleneck(x4)
        
        # Decoder
        x4 = self.upconv3(x4)
        x4 = torch.cat([x3, x4], dim=1)
        x4 = self.dec3(x4)
        
        x4 = self.upconv2(x4)
        x4 = torch.cat([x2, x4], dim=1)
        x4 = self.dec2(x4)
        
        x4 = self.upconv1(x4)
        x4 = torch.cat([x1, x4], dim=1)
        x4 = self.dec1(x4)
        
        # Final output
        return self.final_conv(x4)
```

---

### 3. **Temporal Convolutional Graph Neural Network (TCGNN) for ECG**

```python
import torch_geometric
from torch_geometric.nn import GCNConv

class TCGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TCGNN, self).__init__()
        
        # Temporal Conv Layer
        self.temporal_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Graph Conv Layer
        self.graph_conv1 = GCNConv(hidden_dim, hidden_dim)
        self.graph_conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Attention Layer
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, edge_index):
        # Temporal Convolution
        x = self.temporal_conv(x)
        x = F.relu(x)
        
        # Graph Convolution
        x = self.graph_conv1(x, edge_index)
        x = F.relu(x)
        x = self.graph_conv2(x, edge_index)
        
        # Attention Mechanism
        attention_weights = torch.softmax(self.attention(x), dim=0)
        x = (x * attention_weights).sum(dim=0)
        
        return x
```

---

### 4. **Hybrid Diagnosis Layer**

```python
class HeartNet(nn.Module):
    def __init__(self, mri_model, ecg_model, combined_dim=128):
        super(HeartNet, self).__init__()
        self.mri_model = mri_model
        self.ecg_model = ecg_model
        self.fc = nn.Linear(combined_dim, 2)  # 2 classes: healthy or disease
        
    def forward(self, mri, ecg, edge_index):
        # Extract features
        mri_features = self.mri_model(mri).flatten(1)
        ecg_features = self.ecg_model(ecg, edge_index)
        
        # Combine and classify
        combined_features = torch.cat((mri_features, ecg_features), dim=1)
        output = F.softmax(self.fc(combined_features), dim=1)
        
        return output
```

---

### 5. **Training and Evaluation**

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train(model, data_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for mri, ecg, edge_index, labels in data_loader:
        optimizer.zero_grad()
        output = model(mri, ecg, edge_index)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate(model, data_loader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for mri, ecg, edge_index, labels in data_loader:
            output = model(mri, ecg, edge_index)
            preds = output.argmax(dim=1)
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return acc, prec, rec, f1
```

You can further train and evaluate the model by setting up data loaders, adjusting hyperparameters, and testing with cross-validation to replicate results similar to the paper’s findings. Let me know if you'd like to proceed with additional details on setting up the data loaders and training loop in Colab.
