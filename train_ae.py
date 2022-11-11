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
from Autoencoder import *
from torch.utils.data import DataLoader, Dataset, TensorDataset
from util import *
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Autoencoder')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--cuda', type=int, default=0, help='use of cuda (default: 0)')
parser.add_argument('--epochs', type=int, default=250, help='number of total epochs to run (default: 250)')
parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
parser.add_argument('--z', default=32, type=int, help='gaussian size (default: 32)')
parser.add_argument('--game', default='smb', type=str)
parser.add_argument('--mt', default='conv', choices=['fc','conv'], type=str, help='autoencoder type')
args = parser.parse_args()


path = ''
folder = path + 'data/' + game_folders[args.game]

args.model_name = 'ae_' + args.mt + '_' + args.game + '_'
args.model_name += 'ld_' + str(args.z) + '_'
args.model_name += str(args.epochs)
print('Model name: ', args.model_name)
print('Latent Size: ', args.z)
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
# print(len(levels))
chars = sorted(list(set(text.strip('\n'))))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}
# print(char2int)
# print(sc2int)
# print(int2sc)
num_tiles = len(char2int)

# print('Tiles: ', num_tiles)
# print('Sketch Tiles: ', num_sketch_tiles)
inputs, targets = [], []
for level in levels:
    if args.game == 'smb' and not pipe_check(level):
        continue
    if len(level[0]) != 16 or len(level) != 15:
        # print(len(level), len(level[0]))
        # print('skipping level')
        continue
    tar, inp = [], []
    translated_level = translate_level(level,args.game)
    tl_list = []
    for row in translated_level:
        tl_list.append(list(row))
    for line in tl_list:
        encoded_line = [sc2int[x] for x in line]
        inp.append(encoded_line)
    inputs.append(inp)
    for line in level:
        encoded_line = [char2int[x] for x in line]
        tar.append(encoded_line)
    targets.append(tar)
    
inputs = np.array(inputs)
targets = np.array(targets)

inputs_onehot = np.eye(num_sketch_tiles, dtype='uint8')[inputs]
inputs_onehot = np.rollaxis(inputs_onehot, 3, 1)
targets_onehot = np.eye(num_tiles, dtype='uint8')[targets]
targets_onehot = np.rollaxis(targets_onehot, 3, 1)
print(inputs_onehot.shape, targets_onehot.shape)

inputs_train = torch.from_numpy(inputs_onehot).to(dtype=torch.float64)
targets_train = torch.from_numpy(targets_onehot).to(dtype=torch.float64)
train_ds = TensorDataset(inputs_train,targets_train)
train_dl = DataLoader(train_ds, batch_size=args.batch_size,shuffle=True)

input_size = num_sketch_tiles*15*16
output_size = num_tiles*15*16

step_size = int(args.epochs/4)
print('Step size: ', step_size)
print('Input size: ', input_size)
print('Output size: ', output_size)
if args.mt == 'fc':
    model = Autoencoder(input_size, output_size, args.z).to(args.device)
else:
    model = ConvAutoencoder(num_sketch_tiles, num_tiles, args.z).to(args.device)
opt = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=0.1, patience=50, threshold=0.0001, verbose=True, min_lr=1e-8)
#print(model)

def loss_fn(x_recon, y):
    loss = F.binary_cross_entropy_with_logits(x_recon, y, reduction='none')
    return loss.sum(-1).mean()


num_batches = len(train_dl)
model.train()
for i in tqdm(range(args.epochs)):
    train_loss = 0
    for batch, (x,y) in enumerate(train_dl):
        x, y = x.to(args.device), y.to(args.device)
        if args.mt == 'fc':
            x, y = x.view(x.size(0),-1), y.view(y.size(0),-1)
        opt.zero_grad()
        recon_x = model(x.float())
        loss = loss_fn(recon_x, y)
        train_loss += loss.item()
        loss.backward()
        opt.step()
    train_loss /= num_batches
    if i % 250 == 0:
        print('Epoch: %d\tLoss: %.2f' % (i, train_loss))
    scheduler.step(train_loss)
torch.save(model.state_dict(), args.model_name + '.pth')