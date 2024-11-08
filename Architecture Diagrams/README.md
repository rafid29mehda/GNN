![image](https://github.com/user-attachments/assets/7a51390f-dcc4-4b9a-94bc-c9967bc9bcb4)


The image shows the overall architecture of **Heart-Net**, a multi-modal deep learning framework designed for heart disease diagnosis by integrating **Cardiac MRI** and **ECG data**. The architecture comprises two main parts: the **3D U-Net for MRI feature extraction** and the **Temporal Convolutional Graph Neural Network (TCGNN) for ECG feature extraction**. Here’s a breakdown of each component:

### 1. **3D U-Net for MRI Feature Extraction**

The upper part of the architecture uses a **3D U-Net** to process and extract features from MRI images of the heart. This component consists of:

- **Input Data**: 
  - Each MRI scan is associated with patient information, including Patient ID, Age, Gender, and Pathology (e.g., Heart Failure).
  - The MRI data is stored in a **Data Repository** and is retrieved as a DICOM batch for processing.
  
- **Encoder-Decoder Structure**:
  - The **Encoder Path** (left side) extracts features by progressively downsampling the image while increasing the number of channels. Each layer consists of:
    - **Convolution + ReLU + Batch Normalization (BN)**: This combination applies 3D convolutional layers to capture features from the MRI images.
    - **Pooling Layers**: These layers downsample the image, reducing spatial dimensions but preserving essential features.
    - **Skip Connections**: Skip connections pass features from the encoder directly to corresponding layers in the decoder, preserving spatial details for better reconstruction.
  - The **Bottleneck Layer** in the middle of the U-Net captures the most compressed representation with the smallest spatial size and the highest number of channels, containing the most significant features.

- **Decoder Path** (right side):
  - **Upsampling Layers**: The decoder reconstructs the image to its original resolution using upsampling layers.
  - **Softmax Output**: The final layer generates a probability map for different regions of the heart, such as the left ventricle, right ventricle, and myocardium. This segmentation map provides insights into the structure and condition of the heart.

### 2. **TCGNN for ECG Feature Extraction**

The **Temporal Convolutional Graph Neural Network (TCGNN)** processes ECG data to capture both time-based (temporal) and structural (spatial) dependencies.

- **Input Data**:
  - The ECG data consists of various cardiovascular features (e.g., heart rate, QR segments, pNN50, etc.) stored in the data repository and retrieved for processing.

- **Gated Temporal Attention Module**:
  - **Temporal Attention Layers**: These layers are responsible for capturing the most relevant time-based patterns in the ECG data.
    - **Tanh and Sigmoid Functions**: These activation functions (Tanh and Sigmoid) create a gating mechanism that helps emphasize important parts of the ECG data, allowing the model to focus on significant patterns and filter out irrelevant information.
  - **Attention Mechanism**: This mechanism dynamically adjusts the model’s focus on relevant parts of the ECG data, improving its ability to identify subtle indicators of heart disease.

- **Adaptive Graph Convolution Layer**:
  - **Graph Convolutional Network (GCN)**: The GCN layer captures relationships between different time steps (or nodes) in the ECG signal, representing the temporal dependencies within the ECG data.
  - **Dynamic and Static Graph Learning**:
    - **Local Dynamic Graph**: This component captures dynamic, changing relationships in the ECG data over time, allowing the model to adapt to varying patterns in the signal.
    - **Global Static Graph**: This component captures stable, consistent relationships across time, providing a baseline understanding of the structural dependencies in the ECG data.

### 3. **Concatenation and Classification**

- **Feature Concatenation**:
  - After feature extraction, the relevant features from both MRI and ECG data are concatenated. Different layers (M1, M2, etc.) of extracted features are stacked together in a **Concatenation Block** to create a comprehensive feature vector.

- **Classification Block**:
  - The concatenated features are fed into a classification block, which outputs the probability of different heart disease classes. This final layer uses the combined information from both MRI and ECG data to provide an accurate diagnosis.

### Overall Workflow

1. **3D U-Net processes MRI data** to capture spatial features, highlighting specific heart structures.
2. **TCGNN processes ECG data** to capture temporal and spatial relationships within the heart’s electrical activity.
3. **Features from MRI and ECG are combined** to create a multi-modal feature representation.
4. **Classification layer provides diagnosis**, using the combined information for more accurate predictions of heart disease types.

### Purpose of Heart-Net

The Heart-Net architecture leverages the complementary strengths of MRI and ECG data:
- **MRI Data** provides detailed anatomical structure.
- **ECG Data** provides insights into the heart’s electrical and functional patterns.

This combination allows Heart-Net to deliver a more comprehensive and accurate diagnosis of heart disease, potentially leading to better patient outcomes.
