"""new-app: A Flower / PyTorch app."""

from collections import OrderedDict
import os
import numpy as np
import wfdb
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

DATA_PATH = Path("mitdb")

class HybridModel(nn.Module):
    def __init__(self, config, num_classes):
        super(HybridModel, self).__init__()

        # Extract hyperparameters from config
        feature_extractor = config[8]  # CNN (0) or RCNN (1)
        sequence_model = config[9]  # BiLSTM (0) or GRU (1)
        num_cnn_layers = int(config[0])
        num_rnn_layers = int(config[1])
        dropout = config[3]
        initial_filters = 2 ** int(config[4])  # Convert to power of 2
        initial_kernel = int(config[5])  # Initial kernel size
        stride = int(config[6])
        initial_hidden_size = 2 ** int(config[10])  # Convert to power of 2

        # 🟢 Convolutional Feature Extractor (CNN Layers)
        self.conv_layers = nn.ModuleList()
        num_filters = initial_filters
        kernel_size = initial_kernel
        in_channels = 1  # ECG has 1 channel

        for _ in range(num_cnn_layers):
            kernel_size = max(2, min(kernel_size, in_channels))  # Ensure kernel size is valid
            stride = min(stride, kernel_size)  # Ensure stride is not larger than kernel
            padding = max(0, (kernel_size - stride) // 2)  # Ensure non-negative padding
            
            self.conv_layers.append(nn.Conv1d(in_channels, num_filters, kernel_size, stride=stride, padding=padding))
            self.conv_layers.append(nn.BatchNorm1d(num_filters))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.Dropout(dropout))

            in_channels = num_filters  # Update for next layer
            num_filters = min(256, num_filters * 2)  # Cap filters at 256
            kernel_size = max(3, kernel_size - 1)  # Decrease kernel size

        # 🟢 Handle RCNN (CNN + 1 LSTM/GRU Layer if enabled)
        self.use_rcnn = feature_extractor == 1
        rnn_input_size = in_channels  # Ensure input size matches CNN output

        if self.use_rcnn:
            self.rnn_layers_rcnn = nn.ModuleList()
            hidden_size = initial_hidden_size

            # RCNN should contain ONLY ONE LSTM/GRU layer
            rnn_layer = (nn.LSTM if sequence_model == 0 else nn.GRU)(
                rnn_input_size, hidden_size, bidirectional=True, batch_first=True
            )
            self.rnn_layers_rcnn.append(rnn_layer)
            self.rnn_layers_rcnn.append(nn.Dropout(dropout))

            # Update input size for the next LSTM/GRU layers
            rnn_input_size = hidden_size * 2  # Account for bidirectional RNN

        # 🟢 Sequence Model (BiLSTM or GRU)
        self.rnn_layers = nn.ModuleList()
        hidden_size = initial_hidden_size

        for _ in range(num_rnn_layers):
            rnn_layer = (nn.LSTM if sequence_model == 0 else nn.GRU)(
                rnn_input_size, hidden_size, bidirectional=True, batch_first=True
            )
            self.rnn_layers.append(rnn_layer)
            self.rnn_layers.append(nn.Dropout(dropout))

            # Update input size for next layers
            rnn_input_size = hidden_size * 2  # Account for bidirectional RNN
            hidden_size = max(16, hidden_size // 2)  # Reduce hidden size

        # 🟢 Fully Connected Layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Reduce time dimension to 1
        self.fc = nn.Linear(rnn_input_size, rnn_input_size // 2)  # Use dynamic size
        self.output_layer = nn.Linear(rnn_input_size // 2, num_classes)  # Final layer

        # 🟢 Initialize weights
        self._initialize_weights()

    def forward(self, x):
        # 🟢 CNN Feature Extraction
        for layer in self.conv_layers:
            x = layer(x)

        x = x.permute(0, 2, 1)  # (Batch, TimeSteps, Features)

        # 🟢 Handle RCNN (Reshape CNN output for RNN)
        if self.use_rcnn:            
            for layer in self.rnn_layers_rcnn:
                if isinstance(layer, (nn.LSTM, nn.GRU)):
                    x, _ = layer(x)  # Apply LSTM/GRU
                else:
                    x = layer(x)  # Apply Dropout

        for layer in self.rnn_layers:
            if isinstance(layer, (nn.LSTM, nn.GRU)):
                x, _ = layer(x)  # Get only output
            else:
                x = layer(x)  # Apply Dropout

        # 🟢 Global Pooling & Fully Connected Layers
        x = x.permute(0, 2, 1)  # (Batch, Features, TimeSteps)
        x = self.global_pool(x)  # (Batch, Features, 1)
        x = x.squeeze(-1)  # (Batch, Features)

        x = F.relu(self.fc(x))
        x = self.output_layer(x)

        return x

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

fds = None  # Cache FederatedDataset

def load_raw_signals(data_path):
    data_files, annot_files = [], []
    for file in os.listdir(data_path):
        if file.endswith(".dat"):
            data_files.append(file[:-4])
        elif file.endswith(".atr"):
            annot_files.append(file[:-4])
    
    all_signals = []
    all_labels = []

    for i in range(len(data_files)):
        signal, _ = wfdb.rdsamp(os.path.join(data_path, data_files[i]))
        signal = signal[:, 0]  # Use lead 1

        annot = wfdb.rdann(os.path.join(data_path, annot_files[i]), 'atr')
        peaks = annot.sample
        labels = annot.symbol

        # Extract windows around peaks
        segmented = [signal[max(0, p - 100):min(len(signal), p + 100)] for p in peaks]
        segmented = np.array([
            np.pad(s, (0, 200 - len(s)), mode='edge') if len(s) < 200 else s
            for s in segmented
        ])

        all_signals.append(segmented)
        all_labels.append(labels[:len(segmented)])

    all_signals = np.concatenate(all_signals, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_signals, all_labels

def encode_labels(all_labels):
    char_to_int = {}
    count = 0
    for label in all_labels:
        if label not in char_to_int:
            char_to_int[label] = count
            count += 1

    numeric_labels = np.array([char_to_int[label] for label in all_labels if label in char_to_int])
    valid_indices = [i for i, label in enumerate(all_labels) if label in char_to_int]

    return numeric_labels, valid_indices, char_to_int

class ECGDataset(Dataset):
    def __init__(self, signals, labels):
        self.data = torch.tensor(signals, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_weighted_sampler(labels, dataset_split):
    label_counts = Counter(labels)
    total = sum(label_counts.values())
    weights = {cls: total / count for cls, count in label_counts.items()}

    ds_labels = [dataset_split[i][1].item() for i in range(len(dataset_split))]
    sample_weights = torch.tensor([weights[label] for label in ds_labels], dtype=torch.float32)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

def load_data(partition_id: int, num_partitions: int):
    signals, symbols = load_raw_signals(DATA_PATH)
    numeric_labels, valid_indices, _ = encode_labels(symbols)
    filtered_signals = signals[valid_indices]
    filtered_labels = numeric_labels

    # Partition the dataset across nodes
    total_size = len(filtered_signals)
    partition_size = total_size // num_partitions
    start = partition_id * partition_size
    end = total_size if partition_id == num_partitions - 1 else start + partition_size

    part_signals = filtered_signals[start:end]
    part_labels = filtered_labels[start:end]

    dataset = ECGDataset(part_signals, part_labels)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    sampler = get_weighted_sampler(part_labels, train_ds)
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    return train_loader, test_loader

def train(net, trainloader, epochs, device):
    """Train model with reshaping and validation."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    net.train()
    running_loss = 0.0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, labels in trainloader:
            # Reshape to [B, 1, 200] if needed
            inputs = inputs.view(inputs.shape[0], 1, inputs.shape[-1])
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        running_loss += epoch_loss / len(trainloader)

    avg_train_loss = running_loss / epochs
    return avg_train_loss



def test(net, testloader, device):
    """Evaluate model using your ECG dataset format."""
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in testloader:
            # Reshape to [B, 1, 200] if needed by model
            inputs = inputs.view(inputs.shape[0], 1, inputs.shape[-1])
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(testloader)
    accuracy = correct / len(testloader.dataset)
    return avg_loss, accuracy



def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
