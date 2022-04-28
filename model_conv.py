import torch.nn as nn
import torch.optim
from torch.nn import DataParallel
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        #print(input.size())
        return input.view(input.size(0), -1)  # view(batch_size, flattened_example)

class UnFlatten(nn.Module):
    def __init__(self,h_dim):
        self.h_dim = h_dim
        super(UnFlatten,self).__init__()
    
    def forward(self, input):
        #print('Unflatten:',input.shape)
        return input.view(input.size(0), self.h_dim, 1, 1)
    

class Shape(nn.Module):
    def forward(self, input):
        print(input.shape)
        return input
    
class VAE(nn.Module):
    def __init__(self,in_dim,out_dim,z_dim=32,device=torch.device('cuda'),h_dim=512):
        super(VAE, self).__init__()
        
        self.device = device
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            # input shape: n, 6, 16, 16
            #Shape(),
            nn.Conv2d(in_dim, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            #nn.LeakyReLU(0.2),
            nn.ReLU(),
            # output shape: n, 64, 7, 7
            
            #Shape(),
            # input shape: n, 64, 7, 7
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            #nn.LeakyReLU(0.2),
            nn.ReLU(),
            # output shape: n, 128, 2, 2
            #Shape(),
            # input shape: n, 128, 2, 2
            Flatten()
            # output shape: n, 128 * 2 * 2 = 512
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)  # get means
        self.fc2 = nn.Linear(h_dim, z_dim)  # get logvars
        
        self.fc3 = nn.Linear(z_dim, h_dim)  # process the samples

        self.decoder = nn.Sequential(
            UnFlatten(h_dim),
            #Shape(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #Shape(),
            nn.ConvTranspose2d(128,64,kernel_size=4,stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #Shape(),
            nn.ConvTranspose2d(64, out_dim, kernel_size=4, stride=2),
            #Shape(),
            #Shape()
            nn.Sigmoid()
        )
        
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparametrize(mu, logvar).to(self.device)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z
    
def get_conv_model(dev, in_dim, out_dim, z_dim, lr=1e-3):
    model = VAE(in_dim, out_dim, z_dim, dev)
    model = model.to(dev).double()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    return model, opt

def load_conv_model(path, in_dim, out_dim, z_dim=32, dev=torch.device('cuda')):
    model = VAE(in_dim,out_dim,z_dim).double().to(dev)
    model.load_state_dict(torch.load(path, map_location=dev))
    return model