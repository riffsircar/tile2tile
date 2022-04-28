import torch.nn as nn
import torch.optim
from torch.nn import DataParallel
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        #print(input.shape)
        #print(input.view(input.size(0),-1).shape)
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input):
        #print(input.shape)
        #print(input.view(input.size(0), size, 1, 1).shape)
        return input.view(input.size(0), 640, 1, 1)

class Shape(nn.Module):
    def forward(self, input):
        print(input.shape)
        return input

class Encoder(nn.Module):
    def __init__(self, input_dim=256, nc=2, z_dim=32, device=torch.device('cuda')):
        super(Encoder, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_dim*nc,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,256)
        self.fc41 = nn.Linear(256,z_dim)
        self.fc42 = nn.Linear(256,z_dim)
    
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        esp = esp.to(dtype=torch.float64).to(self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc41(h), self.fc42(h)
        z = self.reparametrize(mu, logvar).to(self.device)
        return z, mu, logvar
    
    def encode(self,x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        z, mu, logvar = self.bottleneck(h3)
        return z, mu, logvar
    
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        return z, mu, logvar
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

class Decoder(nn.Module):
    def __init__(self,out_dim,nc,z_dim,device):
        super(Decoder,self).__init__()
        self.device = device
        self.fc5 = nn.Linear(z_dim,256)
        self.fc6 = nn.Linear(256,512)
        self.fc7 = nn.Linear(512,1024)
        self.fc8 = nn.Linear(1024,out_dim*nc)
    
    def decode(self,z):
        h1 = F.relu(self.fc5(z))
        h2 = F.relu(self.fc6(h1))
        h3 = F.relu(self.fc7(h2))
        h4 = self.fc8(h3)
        return F.sigmoid(h4)
    
    def forward(self,z):
        recon_x = self.decode(z)
        return recon_x

class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 2),
        )
    
    def forward(self,z):
        return self.net(z)
    
class VAE(nn.Module):
    def __init__(self, input_dim, in_nc, out_nc, z_dim=32, device=torch.device('cuda')):
        super(VAE, self).__init__()
        self.device = device
        
        self.encoder = Encoder(input_dim, in_nc,z_dim,device)
        self.decoder = Decoder(input_dim, out_nc,z_dim,device)
        
        """
        self.fc1 = nn.Linear(256*nc,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,256)
        self.fc41 = nn.Linear(256,z_dim)
        self.fc42 = nn.Linear(256,z_dim)
        self.fc5 = nn.Linear(z_dim,256)
        self.fc6 = nn.Linear(256,512)
        self.fc7 = nn.Linear(512,1024)
        self.fc8 = nn.Linear(1024,256*nc)
        
        
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        esp = esp.to(dtype=torch.float64).to(self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        #mu, logvar = self.fc1(h), self.fc2(h)
        mu, logvar = self.fc41(h), self.fc42(h)
        z = self.reparametrize(mu, logvar).to(self.device)
        return z, mu, logvar

    def encode(self, x):
        #h = self.encoder(x)
        #z, mu, logvar = self.bottleneck(h)
        
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        z, mu, logvar = self.bottleneck(h3)
        return z, mu, logvar

    def decode(self, z):
        #z = self.fc3(z)
        #z = self.decoder(z)
        #return z
    
        h1 = F.relu(self.fc5(z))
        h2 = F.relu(self.fc6(h1))
        h3 = F.relu(self.fc7(h2))
        h4 = self.fc8(h3)
        return F.sigmoid(h4)
        """
    def forward(self, x):
        #print(x,type(x))
        #z, mu, logvar = self.encode(x)
        #z = self.decode(z)
        #return z, mu, logvar
        z, mu, logvar = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar, z
    
    
    
def get_model(dev, input_dim, in_nc, out_nc, z_dim, lr=1e-3):
    model = VAE(input_dim, in_nc, out_nc, z_dim, dev)
    model = model.to(dev).double()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    return model, opt

def load_model(path, input_dim, in_nc, out_nc, nz=32, dev=torch.device('cuda')):
    model = VAE(input_dim, in_nc, out_nc, nz, dev).double()
    model.load_state_dict(torch.load(path, map_location=dev))
    return model

def get_disc(z_dim, dev=torch.device('cuda'),lr=1e-3):
    model = Discriminator(z_dim).to(dev).double()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    return model, opt

def load_disc(path,z_dim,dev=torch.device('cuda')):
    model = Discriminator(z_dim).to(dev)
    model.load_state_dict(torch.load(path, map_location=dev))
    return model
    
