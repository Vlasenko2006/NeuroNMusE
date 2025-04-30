#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 12:37:20 2025

@author: andreyvlasenko
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from mp3_2_numpy import numpy_to_mp3  # Import the function for saving MP3 files

# Dataset class
class AudioDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_chunk, target_chunk = self.data[idx]
        return torch.tensor(input_chunk, dtype=torch.float32), torch.tensor(target_chunk, dtype=torch.float32)

# Attention-based neural network
class AttentionModel(nn.Module):
    def __init__(self, input_dim):
        super(AttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, 128, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(128 * 2, 1)  # Attention layer
        self.fc = nn.Linear(128 * 2, input_dim * 2)  # Output for both channels

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        output = self.fc(context_vector)
        return output.view(x.size(0), 2, -1)  # Reshape to [batch_size, 2, samples]

# Training and validation
def train_and_validate(model, train_loader, val_loader, epochs, criterion, optimizer, device, sample_rate):
    model = model.to(device)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Training
        model.train()
        train_loss = 0
        for inputs, targets in tqdm(train_loader, desc="Training"):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

    # Post-validation: Save input, output, and target as MP3
    save_sample_as_mp3(model, val_loader, device, sample_rate)

# Save one validation sample as MP3
def save_sample_as_mp3(model, val_loader, device, sample_rate):
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:  # Take one batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Convert to NumPy and save as MP3
            input_np = inputs.cpu().numpy()[0]  # Take the first sample
            output_np = outputs.cpu().numpy()[0]
            target_np = targets.cpu().numpy()[0]

            numpy_to_mp3(input_np, sample_rate, output_mp3_file="input.mp3")
            numpy_to_mp3(output_np, sample_rate, output_mp3_file="output.mp3")
            numpy_to_mp3(target_np, sample_rate, output_mp3_file="target.mp3")

            print("Saved input, output, and target as MP3 files.")
            break  # Save only one sample

# Main function
if __name__ == "__main__":
    # Constants
    dataset_folder = "dataset"
    batch_size = 16
    epochs = 10
    sample_rate = 16000
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_data = np.load(os.path.join(dataset_folder, "training_set.npy"), allow_pickle=True)
    val_data = np.load(os.path.join(dataset_folder, "validation_set.npy"), allow_pickle=True)

    train_dataset = AudioDataset(train_data)
    val_dataset = AudioDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, criterion, optimizer
    input_dim = train_data[0][0].shape[-1]  # Infer input dimension from data
    model = AttentionModel(input_dim=input_dim)
    criterion = nn.MSELoss()  # Use MSELoss for reconstruction tasks
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train and validate
    train_and_validate(model, train_loader, val_loader, epochs, criterion, optimizer, device, sample_rate)