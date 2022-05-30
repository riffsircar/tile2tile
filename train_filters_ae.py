import argparse, random, torch, os, math, json, sys, re
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model_lin import get_model, load_model
from Autoencoder import *
from torch.utils.data import DataLoader, Dataset, TensorDataset
from util import *


parser = argparse.ArgumentParser(description='Autoencoder')

parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--cuda', type=int, default=1, help='use of cuda (default: 1)')
parser.add_argument('--epochs', type=int, default=10000, help='number of total epochs to run (default: 100)')
parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
parser.add_argument('--z', default=32, type=int, help='gaussian size (default: 64)')
parser.add_argument('--game', default='smb', type=str)
parser.add_argument('--mt', default='fc', choices=['fc','conv'], type=str, help='autoencoder type')
args = parser.parse_args()


path = ''
folder = path + 'data/' + game_folders_nor[args.game]
#manual_seed = random.randint(1, 10000)
#random.seed(manual_seed)
## Random Seed

args.cuda = 0
args.verbose = 1
args.model_name = 'ae_' + args.mt + '_' + args.game + '_'
args.model_name += 'ld_' + str(args.z) + '_'
args.model_name += str(args.epochs)
print(args.model_name)
print('LD: ', args.z)
args.device = torch.device('cuda') if args.cuda else torch.device('cpu')
if args.cuda == 1:
   os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuID)

SEED = args.seed
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if args.cuda:
    torch.cuda.manual_seed(SEED)

levels, text = parse_folder(folder,args.game)
text = text.replace('\n','')
print(len(levels))
chars = sorted(list(set(text.strip('\n'))))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}


print(char2int)
print(sc2int)
print(int2sc)
num_tiles = len(char2int)

print('Tiles: ', num_tiles)
print('Sketch Tiles: ', num_sketch_tiles)

inputs, targets = [], []
for level in levels:
    if args.game == 'smb' and not pipe_check(level):
        continue
    tar, inp = [], []
    #translate_func = translate[GAME]
    #t_levels = translate_func(level)
    translated_level = translate_level(level,args.game)
    #print(level)
    tl_list = []
    for row in translated_level:
        tl_list.append(list(row))
    #print(tl_list)
    #sys.exit()
    for line in tl_list:
        encoded_line = [sc2int[x] for x in line]
        inp.append(encoded_line)
    #print(inp)
    inputs.append(inp)
    for line in level:
        encoded_line = [char2int[x] for x in line]
        tar.append(encoded_line)
    #print(tar)
    targets.append(tar)
    #for _ in range(len(t_levels)):
    #    targets.append(tar)
    
inputs = np.array(inputs)
targets = np.array(targets)
print(inputs.shape, targets.shape)

inputs_onehot = np.eye(num_sketch_tiles, dtype='uint8')[inputs]
inputs_onehot = np.rollaxis(inputs_onehot, 3, 1)
targets_onehot = np.eye(num_tiles, dtype='uint8')[targets]
targets_onehot = np.rollaxis(targets_onehot, 3, 1)
print(inputs_onehot.shape, targets_onehot.shape)

inputs_train = torch.from_numpy(inputs_onehot).to(dtype=torch.float64)
targets_train = torch.from_numpy(targets_onehot).to(dtype=torch.float64)
train_ds = TensorDataset(inputs_train,targets_train)
train_dl = DataLoader(train_ds, batch_size=args.batch_size,shuffle=True)

#vae, opt = get_model(device, 240, num_sketch_tiles, num_tiles, latent_dim,1e-3)
#vae, opt = get_conv_big_model(device, num_sketch_tiles, num_tiles, latent_dim)
#input_size = np.prod(train_ds[0][0].size())
input_size = num_sketch_tiles*15*16
output_size = num_tiles*15*16

if args.mt == 'fc':
    model = Autoencoder(input_size, output_size, args.z).to(args.device)
else:
    model = ConvAutoencoder(num_sketch_tiles, num_tiles, args.z).to(args.device)
opt = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(opt, step_size=2500)
#print(model)

def loss_fn(x_recon, y):
    loss = F.binary_cross_entropy_with_logits(x_recon, y, reduction='none')
    return loss.sum(-1).mean()


num_batches = len(train_dl)
model.train()
for i in range(args.epochs):
    train_loss = 0
    for batch, (x,y) in enumerate(train_dl):
        x, y = x.to(args.device), y.to(args.device)
        if args.mt == 'fc':
            x, y = x.view(x.size(0),-1), y.view(y.size(0),-1)
        #print(x.shape, y.shape)
        opt.zero_grad()
        recon_x = model(x.float())
        #print(recon_x.shape)
        #print(x.shape, recon_x.shape, y.shape)
        #sys.exit()
        #recon_x, mu, logvar, z = vae(x)
        loss = loss_fn(recon_x, y)
        train_loss += loss.item()
        loss.backward()
        opt.step()
    train_loss /= num_batches
    print('Epoch: %d\tLoss: %.2f' % (i, train_loss))
    scheduler.step()
torch.save(model.state_dict(), args.model_name + '.pth')
