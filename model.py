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
from torch.nn import TransformerEncoder, TransformerEncoderLayer

sf = 6  # Scale factor for hidden dimensions
DROPOUT_RATE = 0.3  # Global dropout rate
DROPOUT_RATE_ed = 0.3  # Global dropout rate
FREEZE_ENCODER_DECODER_AFTER = 10000  # Number of steps after which encoder-decoder weights are frozen

# Dataset class
class AudioDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_chunk, target_chunk = self.data[idx]
        return torch.tensor(input_chunk, dtype=torch.float32), torch.tensor(target_chunk, dtype=torch.float32)


# Enhanced Attention-based neural network with Encoder-Decoder architecture
class AttentionModel(nn.Module):
    def __init__(self, input_dim, num_heads=4, num_layers=2, compression_dim=64 * sf):
        super(AttentionModel, self).__init__()
        
        # Enhanced Encoder: Compress the input sequence into a lower-dimensional representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256 * sf),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_RATE_ed),
            nn.Linear(256 * sf, 128 * sf),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_RATE_ed),
            nn.Linear(128 * sf, compression_dim),  # Compress to lower dimension
            nn.ReLU(),
            nn.LayerNorm(compression_dim)  # Normalize the encoded representation
        )
        
        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(d_model=compression_dim, nhead=num_heads, dim_feedforward=512 * sf, dropout=DROPOUT_RATE)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Enhanced Decoder: Reconstruct the original input from the compressed representation
        self.decoder = nn.Sequential(
            nn.Linear(compression_dim, 128 * sf),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_RATE_ed),
            nn.Linear(128 * sf, 256 * sf),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_RATE_ed),
            nn.Linear(256 * sf, input_dim),  # Reconstruct to original dimension
            nn.Sigmoid()  # Sigmoid activation for output stabilization
        )
        
        # Feed-forward layers for the task-specific output (e.g., audio enhancement)
        self.fc = nn.Sequential(
            nn.Linear(compression_dim, 256 * sf),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(256 * sf, 256 * sf),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(256 * sf, input_dim * 2)  # Output layer
        )

    def forward(self, x):
        # Input shape: [batch_size, seq_len, input_dim]

        # Pass through the enhanced encoder
        encoded = self.encoder(x)  # Shape: [batch_size, seq_len, compression_dim]
        
        # Permute dimensions for Transformer (seq_len first)
        encoded_for_transformer = encoded.permute(1, 0, 2)  # Shape: [seq_len, batch_size, compression_dim]
        
        # Pass through Transformer Encoder
        transformer_out = self.transformer(encoded_for_transformer)  # Shape: [seq_len, batch_size, compression_dim]
        
        # Permute back to original dimensions
        transformer_out = transformer_out.permute(1, 0, 2)  # Shape: [batch_size, seq_len, compression_dim]
        
        # Pass through the enhanced decoder to reconstruct the input
        reconstructed = self.decoder(encoded)  # Shape: [batch_size, seq_len, input_dim]
        
        # Use mean pooling across sequence length for task-specific output
        context_vector = transformer_out.mean(dim=1)  # Shape: [batch_size, compression_dim]
        output = self.fc(context_vector)  # Shape: [batch_size, input_dim * 2]
        
        return reconstructed, output.view(x.size(0), 2, -1)  # Return reconstructed input and task-specific output


# Helper function to freeze parameters
def freeze_parameters(module):
    for param in module.parameters():
        param.requires_grad = False


# Training and validation
def train_and_validate(model, train_loader, val_loader, start_epoch, epochs, criterion, optimizer, device, sample_rate, checkpoint_folder, music_out_folder):
    # Move the model to the correct device
    model = model.to(device)

    step = 0  # Track the number of steps
    for epoch in range(start_epoch, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")

        # Training
        model.train()
        train_loss = 0
        for inputs, targets in tqdm(train_loader, desc="Training"):
            # Increment step count
            step += 1

            # Optionally freeze encoder-decoder weights
            if step > FREEZE_ENCODER_DECODER_AFTER:
                print("Freezing encoder and decoder weights...")
                freeze_parameters(model.encoder)
                freeze_parameters(model.decoder)
            else:
                freeze_parameters(model.transformer)
                freeze_parameters(model.fc)


            # Move inputs and targets to the correct device
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            reconstructed, outputs = model(inputs)

            # Compute loss: Reconstruction loss + Task-specific loss
            reconstruction_loss = criterion(reconstructed, inputs)
            if epoch < FREEZE_ENCODER_DECODER_AFTER:
                task_loss = 0.0
            else:
                task_loss = criterion(outputs, targets)
            loss = reconstruction_loss + task_loss

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

            # Accumulate the training loss
            train_loss += loss.item()

        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}, Reconstruction Loss: {reconstruction_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                # Move inputs and targets to the correct device
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                reconstructed, outputs = model(inputs)

                # Compute loss
                reconstruction_loss = criterion(reconstructed, inputs)
                task_loss = criterion(outputs, targets)
                loss = reconstruction_loss + task_loss

                # Accumulate the validation loss
                val_loss += loss.item()

        # Calculate average validation loss
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
            reconstructed, outputs = model(inputs)

            # Convert to NumPy and save
            input_np = inputs.cpu().numpy()[0]  # Take the first sample
            reconstructed_np = reconstructed.cpu().numpy()[0]
            output_np = outputs.cpu().numpy()[0]
            target_np = targets.cpu().numpy()[0]

            os.makedirs(music_out_folder, exist_ok=True)
            np.save(os.path.join(music_out_folder, f"input_epoch_{epoch}.npy"), input_np)
            np.save(os.path.join(music_out_folder, f"reconstructed_epoch_{epoch}.npy"), reconstructed_np)
            np.save(os.path.join(music_out_folder, f"output_epoch_{epoch}.npy"), output_np)
            np.save(os.path.join(music_out_folder, f"target_epoch_{epoch}.npy"), target_np)

            print(f"Saved input, reconstructed, output, and target as NumPy files for epoch {epoch}.")
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
    batch_size = 16 * 8
    epochs = 30000
    sample_rate = 16000
    learning_rate = 0.00002 * 0.25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_folder = "checkpoints_trans2"
    music_out_folder = "music_out_trans2"
    resume_from_checkpoint = "checkpoints_trans2/model_epoch_1290.pt"  # Change this to the checkpoint path if resuming

    # Load datasets
    train_data = np.load(os.path.join(dataset_folder, "training_set.npy"), allow_pickle=True)
    val_data = np.load(os.path.join(dataset_folder, "validation_set.npy"), allow_pickle=True)

    train_dataset = AudioDataset(train_data)
    val_dataset = AudioDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, criterion, optimizer
    input_dim = train_data[0][0].shape[-1]  # Infer input dimension from data
    model = AttentionModel(input_dim=input_dim).to(device)
    criterion = nn.MSELoss()  # Use MSELoss for reconstruction and task-specific losses
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Load checkpoint if specified
    start_epoch = 1
    if resume_from_checkpoint:
        start_epoch = load_checkpoint(resume_from_checkpoint, model, optimizer)

    # Train and validate
    train_and_validate(model, train_loader, val_loader, start_epoch, epochs, criterion, optimizer, device, sample_rate, checkpoint_folder, music_out_folder)