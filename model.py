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

# Dataset class
class AudioDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_chunk, target_chunk = self.data[idx]
        return torch.tensor(input_chunk, dtype=torch.float32), torch.tensor(target_chunk, dtype=torch.float32)

# Attention-based neural network with Dropout
class AttentionModel(nn.Module):
    def __init__(self, input_dim):
        super(AttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(128 * 2, 64),  # Reduce dimensionality
            nn.ReLU(),
            nn.Linear(64, 1)         # Compute attention weights
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),      # Additional layer
            nn.ReLU(),
            nn.Dropout(p=0.5),       # Additional dropout
            nn.Linear(256, input_dim * 2)  # Output layer
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        output = self.fc(context_vector)
        return output.view(x.size(0), 2, -1) + x  # Add residual connection

# Training and validation
def train_and_validate(model, train_loader, val_loader, start_epoch, epochs, criterion, optimizer, device, sample_rate, checkpoint_folder, music_out_folder):
    model = model.to(device)
    for epoch in range(start_epoch, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")

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

        # Save checkpoint and validation sample every 10 epochs
        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch, checkpoint_folder)
            save_sample_as_numpy(model, val_loader, device, music_out_folder, epoch)

# Save one validation sample as NumPy files
def save_sample_as_numpy(model, val_loader, device, music_out_folder, epoch):
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:  # Take one batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Convert to NumPy and save
            input_np = inputs.cpu().numpy()[0]  # Take the first sample
            output_np = outputs.cpu().numpy()[0]
            target_np = targets.cpu().numpy()[0]

            os.makedirs(music_out_folder, exist_ok=True)
            np.save(os.path.join(music_out_folder, f"input_epoch_{epoch}.npy"), input_np)
            np.save(os.path.join(music_out_folder, f"output_epoch_{epoch}.npy"), output_np)
            np.save(os.path.join(music_out_folder, f"target_epoch_{epoch}.npy"), target_np)

            print(f"Saved input, output, and target as NumPy files for epoch {epoch}.")
            break  # Save only one sample

# Save model checkpoint
def save_checkpoint(model, optimizer, epoch, checkpoint_folder):
    os.makedirs(checkpoint_folder, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_folder, f"model_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

# Load model checkpoint
def load_checkpoint(checkpoint_path, model, optimizer):
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")
        return start_epoch
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting from epoch 1.")
        return 1  # Start from the first epoch if no checkpoint is found

# Main function
if __name__ == "__main__":
    # Constants
    dataset_folder = "../dataset"
    batch_size = 16
    epochs = 3000
    sample_rate = 16000
    learning_rate = 0.0001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_folder = "checkpoints"
    music_out_folder = "music_out"
    resume_from_checkpoint = "checkpoints/model_epoch_100.pt"  # Change this to the checkpoint path if resuming

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

    # Load checkpoint if specified
    start_epoch = 1
    if resume_from_checkpoint:
        start_epoch = load_checkpoint(resume_from_checkpoint, model, optimizer)

    # Train and validate
    train_and_validate(model, train_loader, val_loader, start_epoch, epochs, criterion, optimizer, device, sample_rate, checkpoint_folder, music_out_folder)