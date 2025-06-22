# Deep Learning Fundamentals

## Table of Contents
1. [Introduction to Neural Networks](#introduction-to-neural-networks)
2. [Activation Functions](#activation-functions)
3. [Loss Functions](#loss-functions)
4. [Optimization Algorithms](#optimization-algorithms)
5. [Neural Network Architectures](#neural-network-architectures)
6. [Training Deep Neural Networks](#training-deep-neural-networks)
7. [Regularization Techniques](#regularization-techniques)
8. [Batch Normalization](#batch-normalization)
9. [Transfer Learning](#transfer-learning)
10. [Best Practices](#best-practices)

## Introduction to Neural Networks

### What is a Neural Network?
- Computational models inspired by biological neural networks
- Composed of interconnected nodes (neurons) organized in layers
- Capable of learning complex patterns from data

### Key Components
1. **Neurons**: Basic computational units
2. **Weights**: Parameters that transform input data
3. **Biases**: Additional parameters that shift the activation function
4. **Activation Functions**: Introduce non-linearity into the network

### Forward Propagation
```python
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        # Input to hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Hidden to output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        
        return self.a2

# Example usage
input_size = 4
hidden_size = 5
output_size = 3

# Create network
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Generate random input
X = np.random.randn(2, input_size)

# Forward pass
output = nn.forward(X)
print("Network output shape:", output.shape)
print("Output probabilities:", output)
```

## Activation Functions

### Common Activation Functions
1. **Sigmoid**
   - Range: (0, 1)
   - Used in output layer for binary classification
   - Formula: σ(x) = 1 / (1 + e^(-x))

2. **Tanh (Hyperbolic Tangent)**
   - Range: (-1, 1)
   - Zero-centered, helps with optimization
   - Formula: tanh(x) = (e^x - e^-x) / (e^x + e^-x)

3. **ReLU (Rectified Linear Unit)**
   - Computationally efficient
   - Helps with vanishing gradient problem
   - Formula: ReLU(x) = max(0, x)

4. **Leaky ReLU**
   - Solves dying ReLU problem
   - Formula: LeakyReLU(x) = max(αx, x), where α is small

5. **Softmax**
   - Used in output layer for multi-class classification
   - Converts scores to probabilities
   - Formula: softmax(x_i) = e^x_i / Σ(e^x_j)

### Visualization of Activation Functions
```python
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

x = np.linspace(-5, 5, 100)
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x, sigmoid(x))
plt.title('Sigmoid')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(x, tanh(x))
plt.title('Tanh')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(x, relu(x))
plt.title('ReLU')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(x, leaky_relu(x))
plt.title('Leaky ReLU (α=0.01)')
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Loss Functions

### Common Loss Functions
1. **Mean Squared Error (MSE)**
   - Used for regression tasks
   - Formula: MSE = (1/n) * Σ(y_true - y_pred)²

2. **Binary Cross-Entropy**
   - Used for binary classification
   - Formula: BCE = -[y*log(p) + (1-y)*log(1-p)]

3. **Categorical Cross-Entropy**
   - Used for multi-class classification
   - Formula: CCE = -Σ(y_true * log(y_pred))

4. **Sparse Categorical Cross-Entropy**
   - More efficient for integer labels
   - Same as CCE but with integer labels

### Implementation Example
```python
import numpy as np

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def binary_crossentropy(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_crossentropy(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Example usage
y_true_reg = np.array([1.0, 2.0, 3.0])
y_pred_reg = np.array([1.1, 1.9, 3.2])
print("MSE:", mse_loss(y_true_reg, y_pred_reg))

y_true_bin = np.array([0, 1, 1, 0])
y_pred_bin = np.array([0.1, 0.9, 0.8, 0.3])
print("Binary Cross-Entropy:", binary_crossentropy(y_true_bin, y_pred_bin))

y_true_cat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
y_pred_cat = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
print("Categorical Cross-Entropy:", categorical_crossentropy(y_true_cat, y_pred_cat))
```

## Optimization Algorithms

### Common Optimizers
1. **Stochastic Gradient Descent (SGD)**
   - Basic optimizer
   - Can get stuck in local minima
   - Formula: θ = θ - η * ∇J(θ)

2. **Momentum**
   - Adds momentum to SGD
   - Helps accelerate convergence
   - Formula: v = γv + η∇J(θ), θ = θ - v

3. **RMSprop**
   - Adapts learning rate per parameter
   - Uses moving average of squared gradients

4. **Adam (Adaptive Moment Estimation)**
   - Combines Momentum and RMSprop
   - Default choice for many applications

### Implementation Example
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Create a simple model
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# Different optimizers
optimizers = {
    'SGD': optim.SGD(model.parameters(), lr=0.01),
    'Momentum': optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    'RMSprop': optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99),
    'Adam': optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
}

# Loss function
criterion = nn.MSELoss()

# Training loop example
def train_model(X, y, optimizer, num_epochs=100):
    losses = []
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    return losses

# Generate some random data
X = torch.randn(100, 10)
y = torch.randn(100, 1)

# Train with different optimizers
losses = {}
for name, optimizer in optimizers.items():
    print(f"Training with {name}...")
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )
    losses[name] = train_model(X, y, optimizer)

# Plot results
plt.figure(figsize=(10, 6))
for name, loss in losses.items():
    plt.plot(loss, label=name)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss with Different Optimizers')
plt.legend()
plt.grid(True)
plt.show()
```

## Neural Network Architectures

### Feedforward Neural Networks (FNN)
- Simplest type of neural network
- Information flows in one direction (input → hidden → output)
- Good for structured data

### Convolutional Neural Networks (CNN)
- Specialized for grid-like data (images, time series)
- Uses convolutional and pooling layers
- Captures local patterns and translation invariance

### Recurrent Neural Networks (RNN)
- For sequential data (text, time series)
- Has internal memory
- Suffers from vanishing/exploding gradients

### Long Short-Term Memory (LSTM)
- Type of RNN
- Solves vanishing gradient problem
- Better at learning long-term dependencies

### Gated Recurrent Unit (GRU)
- Simpler alternative to LSTM
- Fewer parameters
- Similar performance in many cases

### Example: CNN for Image Classification
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define CNN architecture
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize model, loss function, and optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Print statistics
        running_loss += loss.item()
        if i % 200 == 199:  # Print every 200 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0

print('Finished Training')
```

## Training Deep Neural Networks

### Challenges in Training Deep Networks
1. **Vanishing/Exploding Gradients**
   - Gradients become too small or too large
   - Makes training difficult or impossible

2. **Overfitting**
   - Model performs well on training data but poorly on test data
   - Solution: Regularization techniques

3. **Computational Resources**
   - Deep networks require significant computational power
   - GPUs are often necessary for training

### Training Process
1. **Forward Pass**
   - Compute predictions
   - Calculate loss

2. **Backward Pass**
   - Compute gradients
   - Update weights

3. **Validation**
   - Monitor performance on validation set
   - Early stopping to prevent overfitting

### Example: Training Loop with Validation
```python
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

# Generate some random data
X = torch.randn(1000, 10)
y = (X[:, 0] + 2*X[:, 1] - 1.5*X[:, 2]).view(-1, 1)

# Create dataset and split into train/val
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Simple model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with validation
num_epochs = 50
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_X.size(0)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_X.size(0)
    
    # Calculate average losses
    train_loss = train_loss / len(train_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f'Epoch {epoch+1}/{num_epochs}, ' \
          f'Train Loss: {train_loss:.6f}, ' \
          f'Val Loss: {val_loss:.6f}')

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()
```

## Regularization Techniques

### Common Regularization Methods
1. **L1/L2 Regularization**
   - L1 (Lasso): Adds absolute value of weights to loss
   - L2 (Ridge): Adds squared value of weights to loss

2. **Dropout**
   - Randomly sets activations to zero during training
   - Prevents co-adaptation of features

3. **Data Augmentation**
   - Artificially increases training data
   - Reduces overfitting

4. **Early Stopping**
   - Monitors validation loss
   - Stops training when validation loss stops improving

### Example: Regularization in PyTorch
```python
# L2 Regularization (weight decay)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# L1 Regularization
def l1_regularization(model, lambda_l1):
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.norm(param, 1)
    return lambda_l1 * l1_loss

# Dropout layer
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Dropout(0.5),  # 50% dropout
    nn.Linear(64, 1)
)

# Early stopping
best_val_loss = float('inf')
patience = 5
counter = 0

for epoch in range(num_epochs):
    # Training loop...
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_X.size(0)
    
    val_loss = val_loss / len(val_loader.dataset)
    
    # Early stopping
    if val_loss < best_val_loss - 1e-4:  # Improvement
        best_val_loss = val_loss
        counter = 0
        # Save best model
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
```

## Batch Normalization

### What is Batch Normalization?
- Normalizes layer inputs
- Reduces internal covariate shift
- Allows higher learning rates
- Acts as a form of regularization

### How It Works
1. Normalize each feature to zero mean and unit variance
2. Scale and shift with learnable parameters
3. Use running averages during inference

### Implementation in PyTorch
```python
class NetWithBN(nn.Module):
    def __init__(self):
        super(NetWithBN, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

# Usage
model = NetWithBN()
```

## Transfer Learning

### What is Transfer Learning?
- Using knowledge from one task to improve learning in another
- Particularly useful when you have limited data

### Common Approaches
1. **Feature Extraction**
   - Use pre-trained model as fixed feature extractor
   - Train only the final classification layer

2. **Fine-tuning**
   - Unfreeze some layers of pre-trained model
   - Train the entire network with a small learning rate

### Example: Transfer Learning with ResNet
```python
import torchvision.models as models
import torch.optim as optim

# Load pre-trained ResNet18
model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # 10 output classes

# Only parameters of final layer are being optimized
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# For fine-tuning, unfreeze some layers
for param in model.layer4.parameters():
    param.requires_grad = True

# Now we can train the model
# ...
```

## Best Practices

### Data Preprocessing
1. Normalize input data (mean=0, std=1)
2. Use data augmentation when possible
3. Shuffle your data

### Model Architecture
1. Start with a simple architecture
2. Gradually increase complexity
3. Use batch normalization
4. Use appropriate activation functions

### Training
1. Use learning rate scheduling
2. Monitor training and validation metrics
3. Use early stopping
4. Save the best model

### Debugging
1. Check input data
2. Monitor gradient flow
3. Visualize model predictions
4. Use tensorboard for logging

### Optimization
1. Use Adam as default optimizer
2. Tune learning rate first
3. Use learning rate warmup for transformers
4. Consider mixed precision training

## Attention Mechanisms and Transformers

### Self-Attention Mechanism

Self-attention, also known as intra-attention, is a mechanism that allows different positions of a single sequence to interact with each other to compute a representation of the sequence.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
            
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        out = self.fc_out(out)
        return out
```

### Transformer Architecture

The Transformer architecture, introduced in "Attention Is All You Need" (Vaswani et al., 2017), is based entirely on self-attention mechanisms.

Key components:
1. Multi-Head Attention
2. Position-wise Feed-Forward Networks
3. Positional Encoding
4. Layer Normalization and Residual Connections

## Generative Models

### Autoencoders

Autoencoders are neural networks designed to learn efficient representations of data in an unsupervised manner.

```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
```

### Generative Adversarial Networks (GANs)

GANs consist of two networks: a generator and a discriminator, trained simultaneously through adversarial process.

```python
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
        self.img_shape = img_shape
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
```

## Advanced Architectures

### Graph Neural Networks (GNNs)

GNNs operate on graph-structured data, learning node representations through message passing.

```python
import torch_geometric.nn as geom_nn

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = geom_nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = geom_nn.GCNConv(hidden_dim, hidden_dim)
        self.conv3 = geom_nn.GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)
```

## Model Compression and Optimization

### Pruning

Pruning removes unnecessary weights from a trained model to reduce its size and computational requirements.

```python
def prune_model(model, prune_percent=0.3):
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=prune_percent,
    )
    
    # Remove pruning reparameterization to make the pruning permanent
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')
    
    return model
```

## Explainable AI and Interpretability

### SHAP (SHapley Additive exPlanations)

SHAP values interpret the output of machine learning models by assigning each feature an importance value for a particular prediction.

```python
import shap

def explain_model(model, X_train, X_test):
    # Create explainer
    explainer = shap.DeepExplainer(model, X_train)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test)
    
    # Plot feature importance
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    
    return shap_values
```

## Self-Supervised and Contrastive Learning

### Simple Contrastive Learning with SimCLR

```python
class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super(SimCLR, self).__init__()
        self.encoder = base_encoder
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
        self.encoder.fc = nn.Identity()
    
    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)
        
        z_i = F.normalize(self.projector(h_i), dim=1)
        z_j = F.normalize(self.projector(h_j), dim=1)
        
        return z_i, z_j

def contrastive_loss(z_i, z_j, temperature=0.5):
    N = 2 * z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)
    
    # Compute similarity matrix
    sim = torch.mm(z, z.T) / temperature
    
    # Create labels: the positive pairs are on the diagonal
    labels = torch.cat([torch.arange(z_i.size(0)) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(z.device)
    
    # Remove diagonal (self-similarity)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    sim = sim[~mask].view(sim.shape[0], -1)
    
    # Select positive and negative pairs
    positives = sim[labels.bool()].view(labels.shape[0], -1)
    negatives = sim[~labels.bool()].view(sim.shape[0], -1)
    
    # Compute loss
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(z.device)
    
    return F.cross_entropy(logits, labels)
```

## Reinforcement Learning with Deep Learning

### Deep Q-Network (DQN)

```python
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
```

## Time Series and Sequential Data

### Temporal Convolutional Networks (TCN)

```python
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.net = nn.Sequential(
            self.conv1, nn.ReLU(), nn.Dropout(dropout),
            self.conv2, nn.ReLU(), nn.Dropout(dropout)
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, 
                                   dilation=dilation_size, padding=(kernel_size-1) * dilation_size, 
                                   dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x = x.transpose(1, 2)  # (batch_size, input_size, seq_len)
        x = self.network(x)
        x = x[:, :, -1]  # Take the last output
        return self.linear(x)
```

## Multi-Modal Learning

### Vision-Language Model with CLIP

```python
class CLIP(nn.Module):
    def __init__(self, embed_dim=512, image_encoder=None, text_encoder=None):
        super(CLIP, self).__init__()
        
        # Initialize encoders
        self.image_encoder = image_encoder or resnet50(pretrained=True)
        self.text_encoder = text_encoder or self._get_default_text_encoder()
        
        # Projection heads
        self.image_projection = nn.Linear(2048, embed_dim)
        self.text_projection = nn.Linear(512, embed_dim)  # Adjust based on text encoder
        
    def _get_default_text_encoder(self):
        # Simplified text encoder for demonstration
        return nn.Sequential(
            nn.Embedding(10000, 512),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=512, nhead=8),
                num_layers=4
            )
        )
    
    def encode_image(self, image):
        features = self.image_encoder(image)
        return self.image_projection(features)
    
    def encode_text(self, text):
        features = self.text_encoder(text)
        return self.text_projection(features.mean(dim=1))
    
    def forward(self, image, text):
        image_features = F.normalize(self.encode_image(image), dim=-1)
        text_features = F.normalize(self.encode_text(text), dim=-1)
        
        # Compute similarity matrix
        logits_per_image = image_features @ text_features.t()
        logits_per_text = text_features @ image_features.t()
        
        return logits_per_image, logits_per_text

def clip_loss(logits_per_image, logits_per_text, temperature=0.07):
    # Symmetric cross-entropy loss
    batch_size = logits_per_image.size(0)
    labels = torch.arange(batch_size, device=logits_per_image.device)
    
    loss_i = F.cross_entropy(logits_per_image/temperature, labels)
    loss_t = F.cross_entropy(logits_per_text/temperature, labels)
    
    return (loss_i + loss_t) / 2
```

## Ethics and Fairness in Deep Learning

### Bias Mitigation Techniques

```python
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.metrics import BinaryLabelDatasetMetric

def mitigate_bias(dataset, protected_attribute, privileged_groups, unprivileged_groups):
    # Measure bias before mitigation
    metric_orig = BinaryLabelDatasetMetric(
        dataset, 
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    print(f"Original dataset: \n\tDisparate Impact = {metric_orig.disparate_impact()}")
    
    # Apply reweighting
    RW = Reweighing(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    dataset_transf = RW.fit_transform(dataset)
    
    # Measure bias after mitigation
    metric_transf = BinaryLabelDatasetMetric(
        dataset_transf, 
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    print(f"Transformed dataset: \n\tDisparate Impact = {metric_transf.disparate_impact()}")
    
    return dataset_transf
```

## Conclusion

This document has covered a wide range of deep learning topics, from fundamental concepts to advanced techniques. The field of deep learning continues to evolve rapidly, with new architectures and methods being developed constantly. The best way to stay current is to:

1. Read research papers from top conferences (NeurIPS, ICML, ICLR, CVPR, ACL, etc.)
2. Implement models from scratch to understand their inner workings
3. Participate in open-source projects and competitions
4. Follow developments in related fields like neuroscience and cognitive science
5. Consider the ethical implications of your work

Remember that deep learning is a powerful tool, but it's not always the right solution. Always consider the problem requirements, available data, and computational resources when choosing an approach.

Deep learning is a powerful tool for solving complex machine learning problems. By understanding the fundamentals covered in this guide, you'll be well-equipped to build and train neural networks for a variety of tasks. Remember to start simple, experiment, and iterate based on your results.