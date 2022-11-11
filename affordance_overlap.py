import argparse, random, numpy as np, os, sys
import torch
import torch.utils.data
from util import *
from Autoencoder import *
from scipy.special import kl_div, rel_entr
import collections, warnings
import torch.optim as optim


parser = argparse.ArgumentParser(description='Autoencoder')

parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--cuda', type=int, default=1, help='use of cuda (default: 1)')
parser.add_argument('--epochs', type=int, default=2500, help='number of total epochs to run (default: 100)')
parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
parser.add_argument('--z', default=32, type=int, help='gaussian size (default: 64)')
parser.add_argument('--game', default='smb', type=str)
parser.add_argument('--mt', default='fc', choices=['fc','conv'], type=str, help='autoencoder type')
args = parser.parse_args()

warnings.filterwarnings("ignore")
path = ''
folder = path + 'data/' + game_folders[args.game]

args.cuda = 0
args.verbose = 1
args.model_name = 'ae_full_' + args.mt + '_' + args.game + '_'
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
to_levels_original, text = parse_folder(folder,args.game)
text = text.replace('\n','')
print(len(to_levels_original))
chars = sorted(list(set(text.strip('\n'))))
sketch_chars = ['X','E','|','*','-']
int2char = dict(enumerate(chars))
int2sc = dict(enumerate(sketch_chars))
char2int = {ch: ii for ii, ch in int2char.items()}
sc2int = {ch: ii for ii, ch in int2sc.items()}
print(char2int)
print(sc2int)
print(int2sc)
num_tiles = len(char2int)
num_sketch_tiles = len(sc2int)
print('Tiles: ', num_tiles)
print('Sketch Tiles: ', num_sketch_tiles)
input_size = num_sketch_tiles*15*16
output_size = num_tiles*15*16

if args.mt == 'fc':
    model = Autoencoder(input_size, output_size, args.z).to(args.device)
else:
    model = ConvAutoencoder(num_sketch_tiles, num_tiles, args.z).to(args.device)
opt = optim.Adam(model.parameters(), lr=0.001)
model.load_state_dict(torch.load('trained_ae/' + args.model_name + '.pth',map_location=torch.device(args.device)))
model.eval()


to_levels_original_translated = []
for to_level in to_levels_original:
    to_translated = translate_level(to_level,args.game)
    to_translated = [list(l) for l in to_translated]
    to_levels_original_translated.append(to_translated)


aff_counts = {}
for tile in sc2int:
    aff_counts[tile] = {}
    for other in sc2int:
        aff_counts[tile][other] = 0
print(aff_counts)

for from_game in ['smb','ki','mm','met']:
    if from_game == args.game:
        continue
    from_folder = path + 'data/' + game_folders[from_game]
    from_levels_original, text = parse_folder(from_folder,from_game)

    from_levels_original_translated = []
    to_levels_generated_translated = []
    for from_level in from_levels_original:
        from_translated = translate_level(from_level,from_game)
        #print(from_translated, '\n')
        to_level = apply_ae(model, from_translated, num_tiles, int2char)
        #from_translated = [list(l) for l in from_translated]
        #from_levels_original_translated.append(from_translated)
        #print(translated, '\n')
        #print(to_level, '\n')
        to_level = [list(l) for l in to_level]
        to_translated = translate_level(to_level, args.game)
        #print(to_translated)
        #to_translated = [list(l) for l in to_translated]
        #to_levels_generated_translated.append(to_translated)
        for from_row, to_row in zip(from_translated, to_translated):
            #print(from_row, to_row)
            for from_tile, to_tile in zip(from_row, to_row):
                aff_counts[from_tile][to_tile] += 1
    
    print(from_game)
    print(aff_counts, '\n')
