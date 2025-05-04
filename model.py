import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from encoder_decoder import encoder_decoder  # Importing encoder-decoder class

FREEZE_ENCODER_DECODER_AFTER = 10  # Number of steps after which encoder-decoder weights are frozen

sf = 4
do = 0.3

# Dataset class
class AudioDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_chunk, target_chunk = self.data[idx]
        return torch.tensor(input_chunk, dtype=torch.float32), torch.tensor(target_chunk, dtype=torch.float32)

class VariationalAttentionModel(nn.Module):
    def __init__(self, input_dim, num_heads=2, num_layers=1, n_channels=64, n_seq=3, sound_channels = 2, batch_size = 64, seq_len = 120000):
        super(VariationalAttentionModel, self).__init__()

        # Encoder and Decoder from encoder-decoder
        self.encoder_decoder = encoder_decoder(input_dim=sound_channels, n_channels=n_channels, n_seq=n_seq)

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=n_channels,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=0.1
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Feed-forward layers for task-specific output
        self.fc = nn.Sequential(
            nn.Linear(n_channels, 128),
            nn.ReLU(),
            nn.Dropout(p=do),
            nn.Linear(128, n_channels)
        )

    def forward(self, x):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim, seq_len].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Reconstructed tensor of shape [batch_size, input_dim, seq_len].
                - Output tensor of shape [batch_size, input_dim, seq_len].
        """

        # Verify input shape
        if len(x.shape) != 3:
            raise ValueError(f"Expected input to have 3 dimensions [batch_size, input_dim, seq_len], but got {x.shape}")

        batch_size, input_dim, seq_len = x.shape
       # x = torch.rand(batch_size, input_dim, seq_len)
        # Encoding
        encoded = self.encoder_decoder.encoder(x)  # Use encoder from encoder-decoder

        # Permute for Transformer
        x = encoded.permute(2, 0, 1)  # Shape: [output_seq_len, batch_size, n_channels]

        # Pass through Transformer Encoder
        transformer_out = self.transformer(x)  # Shape: [output_seq_len, batch_size, n_channels]

        # Permute back for Decoder
        transformer_out = transformer_out.permute(1, 2, 0)  # Shape: [batch_size, n_channels, output_seq_len]

        # Decoding
        reconstructed = self.encoder_decoder.decoder(encoded)  # Use decoder from encoder-decoder

        # Process transformer output for task-specific output
        task_specific = self.fc(transformer_out.mean(dim=2))  # Shape: [batch_size, n_channels]
        task_specific = task_specific.unsqueeze(2).expand(-1, -1, seq_len)  # Expand back to sequence length
        task_specific = self.encoder_decoder.decoder(task_specific)  # Decode to match input shape
        task_specific = task_specific.permute(0, 2, 1)  # Shape: [batch_size, seq_len, input_dim]


        return reconstructed, task_specific
