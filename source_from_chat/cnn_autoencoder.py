# SOURCE PROVENANCE: verbatim architecture from user-supplied UCSD evaluation script in conversation.
# CHECKPOINT LOAD SUCCESS: models/best_modelped1aug.pth reported by user; checkpoint binary not available.
import torch

class CNNAutoencoder(torch.nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1,32,3,2,1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32,64,3,2,1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64,128,3,2,1), torch.nn.ReLU(inplace=True),
        )
        self.fc1 = torch.nn.Linear(128*4*4, latent_dim)
        self.fc2 = torch.nn.Linear(latent_dim, 128*4*4)
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128,64,3,2,1,1), torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(64,32,3,2,1,1), torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(32,1, 3,2,1,1), torch.nn.Sigmoid(),
        )
    def forward(self, x):
        z = self.encoder(x)
        z = z.view(z.size(0), -1)
        h = self.fc1(z)
        z2 = self.fc2(h).view(-1,128,4,4)
        return self.decoder(z2)
