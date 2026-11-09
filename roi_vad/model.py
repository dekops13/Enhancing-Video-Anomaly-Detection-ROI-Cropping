"""32x32 grayscale convolutional autoencoder.

The architecture matches the network described in the study. Weights must be trained or supplied separately.
"""
from pathlib import Path
import torch
from torch import nn


class CNNAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Linear(128*4*4, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 128*4*4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 3, 2, 1, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x).view(x.size(0), -1)
        h = self.fc1(z)
        return self.decoder(self.fc2(h).view(-1, 128, 4, 4))


def load_checkpoint(path, device='cpu'):
    """Fail loudly on architecture mismatch rather than silently randomizing layers."""
    model = CNNAutoencoder()
    try:
        checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(Path(path), map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        checkpoint = checkpoint['model']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']
    if not isinstance(checkpoint, dict):
        raise TypeError('Expected state_dict or checkpoint with model/model_state_dict key')
    checkpoint = {k.removeprefix('module.'): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=True)
    return model.to(device).eval()
