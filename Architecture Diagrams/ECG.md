![image](https://github.com/user-attachments/assets/1269bff4-d387-46ec-bbca-87fbe8e661b0)

The uploaded image provides an overview of the **Temporal Convolutional Graph Neural Network (TCGNN)** architecture used in Heart-Net for processing ECG data. This TCGNN is designed to capture both temporal (time-based) and spatial (structural) relationships within ECG signals, enhancing its ability to diagnose heart disease by focusing on patterns over time and between different parts of the ECG. Here’s a breakdown of each part of the architecture:

### 1. **Input Data: Cardiovascular ECG Signals**

- **ECG Signals**: The input data consists of various cardiovascular ECG features such as heart rate, heart rhythm segments, and other ECG metrics (e.g., Hbpermin, QRSeg, etc.).
- **Data Repository**: The ECG signals are stored and retrieved from a data repository, organized in a structured format (e.g., records in a database or CSV files).

### 2. **Gated Temporal Attention Module**

The **Gated Temporal Attention Module** is responsible for focusing on the most relevant parts of the ECG signal over time. It uses attention mechanisms to highlight the important temporal patterns in the data, which improves the network’s ability to detect critical indicators of heart disease.

- **Temporal Attention Layers**: These layers use attention mechanisms to focus on the most relevant time steps in the ECG signal. 
  - **Tanh and Sigmoid Functions**: The temporal attention layers feed into Tanh and Sigmoid functions, which control the output and apply gating mechanisms (soft thresholds) that help the model focus on important signal portions.
- **Gating Mechanism**: The combination of Tanh and Sigmoid creates a gated output that allows the model to emphasize or ignore specific features, depending on their importance for the diagnosis. This helps improve the network’s focus on key information, filtering out less relevant data.

### 3. **Adaptive Graph Convolution Layer (GCN)**

The **Adaptive Graph Convolution Layer** processes the ECG data by understanding the relationships between different time steps (or nodes) within the ECG signal.

- **Graph Convolutional Network (GCN)**: The GCN takes the gated temporal attention output and applies graph convolutions, which help the model capture relationships and dependencies between different parts of the ECG signal.
- **Dynamic and Static Graph Learning**:
  - **Dynamic Graph Learning (Local Dynamic Graph)**: This part dynamically learns connections based on changing relationships in the ECG data over time. It adjusts the structure of the graph based on the features that are most relevant at different times.
  - **Global Static Graph**: The static graph captures stable, fixed relationships within the data, providing a consistent view of connections that do not change over time.

### 4. **Feature Concatenation and Classification Block**

After processing through the Temporal Attention and Graph Convolutional layers, the relevant features from each part of the network are combined and prepared for final classification.

- **Concatenation Block**: The extracted features are stacked and concatenated, combining multiple layers (M1, M2, etc.) from the network into a comprehensive feature vector. This vector incorporates both temporal and spatial dependencies captured in the previous layers.
- **Classification Block**: The concatenated features are then passed into a classification layer, which uses these features to predict the likelihood of different heart disease classes. This final output gives probabilities for each possible diagnosis.

### Summary of Key Components

- **Gated Temporal Attention Module**: Highlights important time-based features in the ECG signal, helping the model focus on significant patterns.
- **Adaptive Graph Convolution Layer**: Captures relationships between different parts of the ECG data by learning both dynamic (changing) and static (fixed) connections between nodes.
- **Concatenation and Classification**: Combines all relevant features for final classification, predicting heart disease types based on the processed ECG data.

### Overall Purpose

The TCGNN architecture is tailored to capture both the time-based changes (temporal dependencies) and the relationships between different segments of the ECG signal (spatial dependencies). This dual focus allows Heart-Net to accurately and reliably detect heart disease by analyzing patterns across time and within the structure of the ECG data.
