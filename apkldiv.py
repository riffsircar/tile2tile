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
parser.add_argument('--epochs', type=int, default=500, help='number of total epochs to run (default: 100)')
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
model.load_state_dict(torch.load('trained_models/' + args.model_name + '.pth',map_location=torch.device(args.device)))
model.eval()
#print(model)

def get_pattern_dict(levels, ps):
    patterns = collections.defaultdict(int)
    for level in levels:
        level = [''.join(l) for l in level]
        for r in range(0,len(level)-ps):
            for c in range(0, len(level[0])-ps):
                out = []
                for line in level[r:r+ps]:
                    out.append(line[c:c+ps])
                outstr = ''.join(out)
                if outstr not in patterns:
                    patterns[outstr] = 1
                else:
                    patterns[outstr] += 1
    return patterns


def compute_pattern_prob(pattern_count, num_patterns, epsilon=1e-7):
    return (pattern_count + epsilon) / ((num_patterns + epsilon) * (1 + epsilon))

def compute_kldiv(p, q, pattern_sizes):
    klds = []
    for ps in pattern_sizes:
        p_pats = get_pattern_dict(p,ps)
        q_pats = get_pattern_dict(q,ps)

        num_p = sum(p_pats.values())
        num_q = sum(q_pats.values())
        p_probs, q_probs = [], []
        for pat, count in p_pats.items():
            p_prob = compute_pattern_prob(count, num_p)
            q_prob = compute_pattern_prob(q_pats[pat], num_q)
            p_probs.append(p_prob)
            q_probs.append(q_prob)
        #kld = kl_div(p_probs, q_probs)
        rel = rel_entr(p_probs, q_probs)
        #kl_divergence += p_prob * math.log(p_prob / q_prob) + (1 - hparams.weight) * q_prob * math.log(q_prob / p_prob)
        #print(kld.shape)
        #print(ps, '\t', np.sum(kld), '\t', np.sum(rel))
        klds.append(np.sum(rel))
    return np.mean(klds)

to_levels_original_translated = []
for to_level in to_levels_original:
    to_translated = translate_level(to_level,args.game)
    to_translated = [list(l) for l in to_translated]
    to_levels_original_translated.append(to_translated)

out_file = open(f'apkldiv_{args.game}_{args.mt}_{args.epochs}_{args.z}.csv','w')
out_file.write('Source,TF Target vs OG Source,TF Target vs OG Target,OG Target vs OG Source\n')
# COMPUTE APKLD between afford(generated) vs afford(from) and afford(generated) vs afford(to) - former should be lower??
for from_game in ['smb','ki','mm','met']:
    if from_game == args.game:
        continue
    from_folder = path + 'data/' + game_folders[from_game]
    from_levels_original, text = parse_folder(from_folder,from_game)

    from_levels_original_translated = []
    to_levels_generated_translated = []
    for from_level in from_levels_original:

        # translate source tiles to source affs
        from_translated = translate_level(from_level,from_game)
        from_translated = [list(l) for l in from_translated]
        from_levels_original_translated.append(from_translated) # store source levels in aff format

        # apply autoencoder to transfer style from source aff to target tiles
        to_level = apply_ae(model, from_translated, num_tiles, int2char, args.mt)
        
        # translate target tiles to target affs
        to_level = [list(l) for l in to_level]
        to_translated = translate_level(to_level, args.game)
        to_translated = [list(l) for l in to_translated]
        to_levels_generated_translated.append(to_translated) # store target levels in aff format
    

    patterns = [2,3,4]
    # content/affordance loss between generated targets and original sources
    kldiv_from = compute_kldiv(to_levels_generated_translated, from_levels_original_translated, patterns)

    # content/affordance loss between generated targets and original targets
    kldiv_to = compute_kldiv(to_levels_generated_translated, to_levels_original_translated, patterns)

    # content loss between original levels of the 2 games
    kldiv_from_to = compute_kldiv(to_levels_original_translated, from_levels_original_translated, patterns) 

    print(f"{from_game} to {args.game}: ")
    print(kldiv_from, kldiv_to, kldiv_from_to, '\n')
    out_file.write(f'{from_game},{kldiv_from:.2f},{kldiv_to:.2f},{kldiv_from_to:.2f}\n')
out_file.close()