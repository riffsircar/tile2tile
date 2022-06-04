import argparse, random, numpy as np, os, sys
import torch
import torch.utils.data
from util import *
from Autoencoder import *
from scipy.special import kl_div, rel_entr
import collections, warnings
import torch.optim as optim
import pickle
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.stats import gof

parser = argparse.ArgumentParser(description='Autoencoder')

parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--cuda', type=int, default=1, help='use of cuda (default: 1)')
parser.add_argument('--epochs', type=int, default=250, help='number of total epochs to run (default: 100)')
parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
parser.add_argument('--z', default=32, type=int, help='gaussian size (default: 64)')
parser.add_argument('--game', default='smb', type=str)
parser.add_argument('--mt', default='conv', choices=['fc','conv'], type=str, help='autoencoder type')
parser.add_argument('--mod', default='ae', choices=['ae','mrf'], type=str, help='model type')
parser.add_argument('--ns', default=4, choices=[4,8], type=int, help='MRF net')
args = parser.parse_args()

warnings.filterwarnings("ignore")
path = ''
folder = path + 'data/' + game_folders_nor[args.game]

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
#int2char = dict(enumerate(chars))
int2char = int2chars[args.game] if args.game in ['mm','met'] else dict(enumerate(chars))
int2sc = dict(enumerate(sketch_chars))
char2int = {ch: ii for ii, ch in int2char.items()}
sc2int = {ch: ii for ii, ch in int2sc.items()}
print('int2char: ', int2char)
print('char2int: ', char2int)
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

    print(hist_original)
    sum_values = sum(hist_original.values())
    hist_original_norm = {}
    for key in hist_original:
        hist_original_norm[key] = hist_original[key]/sum_values
    print(hist_original_norm)
    #hist_original_norm_sorted = collections.OrderedDict(sorted(hist_original_norm.items()))
    hist_original_norm_sorted = dict(sorted(hist_original_norm.items()))
    #plt.hist(hist_original_norm)
    #print(hist_original)
    #plt.hist(hist_original)
    plt.bar(hist_original_norm_sorted.keys(), hist_original_norm_sorted.values())
    plt.ylabel('Occurence Percentage')
    plt.xlabel('Tiles')
    plt.xticks(fontsize=15)
    plt.savefig(f'out_tile_hist/tile_hist_{args.mod}_orig_{args.game}.png')
    plt.clf()

    hist_generated = {}
    for from_game in ['smb','ki','mm','met']: #,'mm','met']:
        if from_game == args.game:
            continue
        hist_generated[from_game] = {}
        for tile in hist_original:
            hist_generated[from_game][tile] = 0
        from_folder = path + 'data/' + game_folders_nor[from_game]
        from_levels_original, text = parse_folder(from_folder,from_game)

        from_levels_original_translated = []
        to_levels_generated_translated = []
        for from_level in from_levels_original:
            from_translated = translate_level(from_level,from_game)
            #to_level = apply_ae(model, from_translated, num_tiles, int2char, args.mt)
            to_level = apply_ae(model, from_translated, num_tiles, args.game, args.mt)
            from_translated = [list(l) for l in from_translated]
            from_levels_original_translated.append(from_translated)
            for row in to_level:
                for tile in row:
                    if tile == '@' or tile == '-':
                        continue
                    if tile not in hist_generated[from_game]:
                        hist_generated[from_game][tile] = 0
                    hist_generated[from_game][tile] += 1
        
    
        #out_file = open(f'tile_hist_ae_{from_game}.csv','w')
    
elif args.mod == 'mrf':
    smb_mrf = pickle.load(open('mrf_' + str(args.ns) + '_' + 'smb.pickle','rb'))
    ki_mrf = pickle.load(open('mrf_' + str(args.ns) + '_' + 'ki.pickle','rb'))
    mm_mrf = pickle.load(open('mrf_' + str(args.ns) + '_' + 'mm.pickle','rb'))
    met_mrf = pickle.load(open('mrf_' + str(args.ns) + '_' + 'met.pickle','rb'))
    mrfs = {'smb':smb_mrf, 'ki':ki_mrf, 'mm':mm_mrf, 'met':met_mrf}

    hist_original = {}
    hist_generated = {}

    to_levels = []
    for level_file in os.listdir('VGLC/' + args.game + '/'):
        level = open('VGLC/' + args.game + '/' + level_file).read().splitlines()
        level = preprocess_level(level,args.game)
        for row in level:
            for tile in row:
                if tile == '@' or tile == '-':
                    continue
                if tile not in hist_original:
                    hist_original[tile] = 0
                hist_original[tile] += 1
        to_levels.append(level)

    for from_game in ['smb','ki','mm','met']:
        print(from_game)
        if from_game == args.game:
            continue
        hist_generated[from_game] = {}
        from_levels = []
        for level_file in os.listdir('VGLC/' + from_game + '/'):
            from_level = open('VGLC/' + from_game + '/' + level_file).read().splitlines()
            from_level = preprocess_level(from_level,from_game)
            for tile in hist_original:
                hist_generated[from_game][tile] = 0
            from_translated = translate_level(from_level,from_game,'mrf')
            out_level = apply_mrf(from_translated,mrfs[args.game],args.game,args.ns)
            from_levels.append(out_level)
        print(len(from_levels), len(to_levels))
        #from_sample = random.sample(from_levels, len(to_levels))
        from_sample = from_levels
        for from_level in from_sample:
            for row in from_level:
                for tile in row:
                    if tile == '@' or tile == '-':
                        continue
                    if tile not in hist_generated[from_game]:
                        hist_generated[from_game][tile] = 0
                    hist_generated[from_game][tile] += 1
        #to_level = [''.join(l) for l in out_level]
        #print('\n'.join(to_level))

    print('MRF')
    #plt.bar(x=list(hist_original_mrf.keys()), height=hist_original_mrf.values())
    #plt.show()
    print(hist_original)
    hist_original_norm = {}
    sum_values = sum(hist_original.values())
    for key in hist_original:
        hist_original_norm[key] = hist_original[key]/sum_values
    print(hist_original_norm)
    #plt.hist(hist_original_norm)
    #print(hist_original)
    #plt.hist(hist_original)
    hist_original_norm_sorted = dict(sorted(hist_original_norm.items()))
    plt.bar(hist_original_norm_sorted.keys(), hist_original_norm_sorted.values())
    plt.ylabel('Occurence Percentage')
    plt.xlabel('Tiles')
    plt.savefig(f'out_tile_hist/tile_hist_orig_{args.mod}_{args.game}.png')
    plt.xticks(fontsize=15)
    plt.clf()
    # for from_game in hist_generated:
    #     print(from_game, hist_generated[from_game])
    #     plt.bar(list(hist_generated[from_game].keys()), height=hist_generated[from_game].values())
    #     plt.show()

if args.mod == 'ae':
    out_file = open(f'out_tile_hist/tile_hist_{args.mod}_{args.mt}_{args.z}_{args.epochs}_{args.game}.csv','w')
else:
    out_file = open(f'out_tile_hist/tile_hist_{args.mod}_{args.ns}_{args.game}.csv','w')
out_file.write(',Chi-Square P-Val\n')
print(hist_original)
x_del = 0
for from_game, hist_generated_from_game in hist_generated.items():
    #stats.chisquare(f_obs=hist)
    #out_file.write(f'{from_game}-{args.game},')
    
    print(from_game, hist_generated_from_game)
    
    sum_values = sum(hist_generated_from_game.values())
    hist_norm = {}
    for key in hist_generated_from_game:
        hist_norm[key] = hist_generated_from_game[key]/sum_values
    print(hist_norm)
    #plt.hist(hist_norm)
    
    hist_norm_sorted = dict(sorted(hist_norm.items()))
    x = np.arange(len(hist_norm_sorted))
    #plt.bar(hist_norm_sorted.keys(), hist_norm_sorted.values(), alpha=0.5, label=f'{from_game}_{args.game}')
    plt.bar(x+x_del, hist_norm_sorted.values(), 0.3, label=f'{from_game}-{args.game}'.upper())
    #plt.bar(hist_norm_sorted, alpha=0.25, label=f'{from_game}_{args.game}', x=x+x_del)
    x_del += 0.3
    #plt.hist(hist_norm_sorted.values(), list(hist_norm_sorted.keys()), alpha=0.5, label=f'{from_game}_{args.game}')
    plt.ylabel('Occurence Percentage')
    plt.xlabel('Tiles')
    
    hist_original_list, hist_gen_from_game_list = [], []
    for tile in hist_original:
        hist_original_list.append(hist_original[tile])
        hist_gen_from_game_list.append(hist_generated_from_game[tile])
    print(hist_gen_from_game_list)
    print(hist_original_list)
    hgfgl = np.array(hist_gen_from_game_list)
    hol = np.array(hist_original_list)
    hgfgl = hgfgl/hgfgl.sum()
    hol = hol/hol.sum()
    #chisq = stats.chisquare(hgfgl, hol)
    print(sum(hist_gen_from_game_list), sum(hgfgl))
    print(sum(hist_original_list), sum(hol))
    #chisq = stats.chisquare(hgfgl, hol)
    # chisq = stats.chisquare(hist_gen_from_game_list,hist_original_list)
    # print(chisq)
    # pval = chisq[1]
    # #print('GOF:', gof.chisquare(hgfgl, f_exp=hol))
    # out_file.write(f'{from_game}-{args.game},{pval}\n')
    if args.mod == 'mrf':
        hist_file = open(f'out_tile_hist/{args.mod}_{args.ns}_{from_game}_{args.game}.csv','w')
    else:
        hist_file = open(f'out_tile_hist/{args.mod}_{args.mt}_{args.z}_{from_game}_{args.game}.csv','w')
    hist_file.write('OG,TF\n')
    #for ho, hg in zip(hol,hist_gen_from_game_list):
    for ho, hg in zip(hist_original_list,hist_gen_from_game_list):
        hist_file.write(f'{str(ho)},{str(hg)}\n')
    hist_file.close()
plt.legend(fontsize=15)
plt.xticks(np.arange(len(hist_norm_sorted))+0.3, hist_norm_sorted.keys(), fontsize=15)
if args.mod == 'ae':
    plt.savefig(f'out_tile_hist/tile_hist_ae_{args.mt}_{args.z}_{args.epochs}_{args.game}.png')
else:
    plt.savefig(f'out_tile_hist/tile_hist_mrf_{args.ns}_{args.game}.png')
out_file.close()