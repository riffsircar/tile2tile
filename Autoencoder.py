import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F

class Flatten(nn.Module):
    def forward(self, input):
        #print(input.size())
        return input.view(input.size(0), -1)  # view(batch_size, flattened_example)
class Shape(nn.Module):
    def forward(self, input):
        print(input.shape)
        return input

class Encoder(nn.Module):
    def __init__(self, x_dim, z_dim):
        super(Encoder, self).__init__()
        
        self.encode = nn.Sequential(
            nn.Linear(x_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,z_dim)
        )
        
    def forward(self, x):
        z = self.encode(x)
        return z

class Decoder(nn.Module):
    def __init__(self, y_dim, z_dim):
        super(Decoder, self).__init__()
        
        self.decode = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, y_dim),
        )
        
    def forward(self, z):
        x = self.decode(z)
        return x

class ConvEncoder(nn.Module):
    def __init__(self, in_channels, z_dim):
        super(ConvEncoder, self).__init__()
        
        self.encode = nn.Sequential(
            # input shape: n, 6, 16, 16
            #Shape(),
            nn.Conv2d(in_channels, 512, kernel_size=4, stride=1),
            nn.BatchNorm2d(512),
            #nn.LeakyReLU(0.2),
            nn.ReLU(),
            # output shape: n, 64, 7, 7
            
            #Shape(),
            # input shape: n, 64, 7, 7
            nn.Conv2d(512, 256, kernel_size=4, stride=1),
            nn.BatchNorm2d(256),
            #nn.LeakyReLU(0.2),
            nn.ReLU(),
            # output shape: n, 128, 2, 2
            #Shape(),
            nn.Conv2d(256, z_dim, kernel_size=4, stride=2),
            nn.BatchNorm2d(z_dim),
            #nn.LeakyReLU(0.2),
            nn.ReLU(),
            # output shape: n, 128, 2, 2
            #Shape(),
            # input shape: n, 128, 2, 2
            # output shape: n, 128 * 2 * 2 = 512
        )
        
    def forward(self, x):
        z = self.encode(x)
        return z

class ConvDecoder(nn.Module):
    def __init__(self, out_channels, z_dim):
        super(ConvDecoder, self).__init__()
        
        self.decode = nn.Sequential(
            #Shape(),
            nn.ConvTranspose2d(z_dim, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #Shape(),
            nn.ConvTranspose2d(64,128,kernel_size=4,stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #Shape(),
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=1),
            #Shape(),
            nn.Upsample((15,16)),
            #Shape()
            #Shape()
            #nn.Sigmoid()
        )
        
    def forward(self, z):
        x = self.decode(z)
        return x

class Autoencoder(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim):
        super(Autoencoder, self).__init__()
        
        self.encoder = Encoder(x_dim, z_dim)
        self.decoder = Decoder(y_dim, z_dim)
        
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels, out_channels, z_dim):
        super(ConvAutoencoder, self).__init__()
        
        self.encoder = ConvEncoder(in_channels, z_dim)
        self.decoder = ConvDecoder(out_channels, z_dim)
        
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon