"""
model.py
---------
Defines the Convolutional Denoising Autoencoder architecture used to
remove noise from MNIST digit images.

Architecture overview:
    Encoder: Conv2d -> ReLU -> MaxPool  (twice)  -> compressed representation
    Decoder: ConvTranspose2d -> ReLU (twice) -> Sigmoid output

Input/Output shape: (batch_size, 1, 28, 28), pixel values in [0, 1]
"""

import torch
import torch.nn as nn


class DenoisingAutoencoder(nn.Module):
    """
    A convolutional autoencoder for denoising 28x28 grayscale MNIST images.
    """

    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        # ---------------- Encoder ----------------
        # Input: 1 x 28 x 28
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),  # 32 x 28 x 28
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 32 x 14 x 14

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),  # 16 x 14 x 14
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 16 x 7 x 7  -> compressed representation (784 -> 784*16/16... )
        )

        # ---------------- Decoder ----------------
        # Input to decoder: 16 x 7 x 7
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=2, stride=2),  # 32 x 14 x 14
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=2, stride=2),  # 1 x 28 x 28
            nn.Sigmoid()  # squashes output to [0, 1] to match normalized pixel range
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    # Quick sanity check: pass a dummy batch through the model and print shapes.
    model = DenoisingAutoencoder()
    dummy_input = torch.randn(8, 1, 28, 28)  # batch of 8 fake images
    output = model(dummy_input)
    print(f"Input shape : {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {num_params:,}")
