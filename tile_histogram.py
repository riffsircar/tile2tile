import argparse, random, numpy as np, os, sys
import torch
import torch.utils.data
from util import *
from Autoencoder import *
from scipy.special import kl_div, rel_entr
import collections, warnings
import torch.optim as optim
import pickle

parser = argparse.ArgumentParser(description='Autoencoder')

parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--cuda', type=int, default=1, help='use of cuda (default: 1)')
parser.add_argument('--epochs', type=int, default=2500, help='number of total epochs to run (default: 100)')
parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
parser.add_argument('--z', default=32, type=int, help='gaussian size (default: 64)')
parser.add_argument('--game', default='smb', type=str)
parser.add_argument('--mt', default='fc', choices=['fc','conv'], type=str, help='autoencoder type')
parser.add_argument('--mod', default='ae', choices=['ae','mrf'], type=str, help='model type')
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

if args.mod == 'ae':
    if args.mt == 'fc':
        model = Autoencoder(input_size, output_size, args.z).to(args.device)
    else:
        model = ConvAutoencoder(num_sketch_tiles, num_tiles, args.z).to(args.device)
    opt = optim.Adam(model.parameters(), lr=0.001)
    model.load_state_dict(torch.load('trained_models/' + args.model_name + '.pth',map_location=torch.device(args.device)))
    model.eval()

    hist_original = {}
    to_levels_original_translated = []
    for to_level in to_levels_original:
        to_level = [''.join(l) for l in to_level]
        for row in to_level:
            for tile in row:
                if tile == '@' or tile == '-':
                    continue
                if tile not in hist_original:
                    hist_original[tile] = 0
                hist_original[tile] += 1

    import matplotlib.pyplot as plt

    print(hist_original)

    hist_generated = {}
    for from_game in ['smb','ki','mm','met']:
        hist_generated[from_game] = {}
        if from_game == args.game:
            continue
        from_folder = path + 'data/' + game_folders[from_game]
        from_levels_original, text = parse_folder(from_folder,from_game)

        from_levels_original_translated = []
        to_levels_generated_translated = []
        for from_level in from_levels_original:
            from_translated = translate_level(from_level,from_game)
            to_level = apply_ae(model, from_translated, num_tiles, int2char)
            from_translated = [list(l) for l in from_translated]
            from_levels_original_translated.append(from_translated)
            #print(from_level, '\n')
            #print(translated, '\n')
            #print(to_level, '\n')
            for row in to_level:
                for tile in row:
                    if tile == '@' or tile == '-':
                        continue
                    if tile not in hist_generated[from_game]:
                        hist_generated[from_game][tile] = 0
                    hist_generated[from_game][tile] += 1
            # to_level = [list(l) for l in to_level]
            # to_translated = translate_level(to_level, args.game)
            # to_translated = [list(l) for l in to_translated]
            # to_levels_generated_translated.append(to_translated)
            #print(to_level)
    for game in hist_generated:
        print(game, hist_generated[game])
elif args.mod == 'mrf':
# MRF
    mrf_ns = '4'
    smb_mrf = pickle.load(open('mrf_' + mrf_ns + '_' + 'smb.pickle','rb'))
    ki_mrf = pickle.load(open('mrf_' + mrf_ns + '_' + 'ki.pickle','rb'))
    mm_mrf = pickle.load(open('mrf_' + mrf_ns + '_' + 'mm.pickle','rb'))
    met_mrf = pickle.load(open('mrf_' + mrf_ns + '_' + 'met.pickle','rb'))
    mrfs = {'smb':smb_mrf, 'ki':ki_mrf, 'mm':mm_mrf, 'met':met_mrf}

    hist_original_mrf = {}
    hist_generated_mrf = {}
    for from_game in ['smb','ki','mm','met']:
        if from_game == args.game:
            continue
        hist_generated_mrf[from_game] = {}
        for level_file in os.listdir('VGLC/' + from_game + '/'):
            from_level = open('VGLC/' + from_game + '/' + level_file).read().splitlines()
            for row in from_level:
                for tile in row:
                    if tile == '@' or tile == '-':
                        continue
                    if tile not in hist_original_mrf:
                        hist_original_mrf[tile] = 0
                    hist_original_mrf[tile] += 1

            from_translated = translate_level(from_level,from_game)
            out_level = apply_mrf(from_translated,mrfs[args.game],int(mrf_ns))
            #print('\n'.join(out_level))
            for row in out_level:
                for tile in row:
                    if tile == '@' or tile == '-':
                        continue
                    if tile not in hist_generated_mrf[from_game]:
                        hist_generated_mrf[from_game][tile] = 0
                    hist_generated_mrf[from_game][tile] += 1
            to_level = [''.join(l) for l in out_level]
            #print('\n'.join(to_level))

    print('MRF')
    plt.bar(x=list(hist_original_mrf.keys()), height=hist_original_mrf.values())
    plt.show()
    print(hist_original_mrf)
    for game in hist_generated_mrf:
        print(game, hist_generated_mrf[game])
        plt.bar(list(hist_generated_mrf[game].keys()), height=hist_generated_mrf[game].values())
        plt.show()
