import argparse, random, numpy as np, os, sys
import torch
import torch.utils.data
from util import *
from Autoencoder import *
import warnings
import torch.optim as optim
from agent import *
import pickle

parser = argparse.ArgumentParser(description='Playability evaluation')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--cuda', type=int, default=0, help='use of cuda (default: 0)')
parser.add_argument('--epochs', type=int, default=250, help='number of total epochs to run (default: 250)')
parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
parser.add_argument('--z', default=32, type=int, help='gaussian size (default: 32)')
parser.add_argument('--game', default='smb', type=str)
parser.add_argument('--mt', default='conv', choices=['fc','conv'], type=str, help='autoencoder type')
parser.add_argument('--mod', default='ae', choices=['ae','mrf'], type=str, help='model type')
parser.add_argument('--ns', default=4, choices=[4,8], type=int, help='MRF network size (default: 4)')
args = parser.parse_args()

warnings.filterwarnings("ignore")
path = ''
folder = path + 'data/' + game_folders[args.game]

args.model_name = 'ae_full_' + args.mt + '_' + args.game + '_'
args.model_name += 'ld_' + str(args.z) + '_'
args.model_name += str(args.epochs)
print('Model Name: ', args.model_name)
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
to_levels_original, text = parse_folder(folder,args.game)
text = text.replace('\n','')
# print(len(to_levels_original))
chars = sorted(list(set(text.strip('\n'))))
sketch_chars = ['X','E','|','*','-']
int2char = dict(enumerate(chars))
int2sc = dict(enumerate(sketch_chars))
char2int = {ch: ii for ii, ch in int2char.items()}
sc2int = {ch: ii for ii, ch in int2sc.items()}
# print(char2int)
# print(sc2int)
# print(int2sc)
num_tiles = len(char2int)
num_sketch_tiles = len(sc2int)
# print('Tiles: ', num_tiles)
# print('Sketch Tiles: ', num_sketch_tiles)
input_size = num_sketch_tiles*15*16
output_size = num_tiles*15*16

with open(path + 'affordances/aff_aff.json') as f:
	aff_aff = json.load(f)
with open(path + 'affordances/aff_smb.json') as f:
	aff_smb = json.load(f)
with open(path + 'affordances/aff_ki.json') as f:
	aff_ki = json.load(f)
with open(path + 'affordances/aff_mm.json') as f:
	aff_mm = json.load(f)
with open(path + 'affordances/aff_met.json') as f:
	aff_met = json.load(f)

with open(path + 'jumps/jumps_smb.json') as f:
	jumps_smb = json.load(f)['jumps']
with open(path + 'jumps/jumps_ki.json') as f:
	jumps_ki = json.load(f)['jumps']
with open(path + 'jumps/jumps_mm.json') as f:
	jumps_mm = json.load(f)['jumps']
with open(path + 'jumps/jumps_met.json') as f:
	jumps_met = json.load(f)['jumps']

affs = {'aff':aff_aff, 'smb':aff_smb, 'ki': aff_ki, 'mm':aff_mm, 'met':aff_met}
jumps = {'smb':jumps_smb, 'ki': jumps_ki, 'mm':jumps_mm, 'met':jumps_met}

def extract_chunks(level):
	chunks = []
	for h_offset in range(0,len(level)-15+1,15): #,dims[0]):
		for w_offset in range(0,len(level[0])-(16-1),16): #,dims[1]):
			write = True
			out = []
			for line in level[h_offset:h_offset+15]:
				out.append(line[w_offset:w_offset+16])
			if any('@' in line for line in out):
				continue
			chunks.append(out)
			try:
				out_string = '\n'.join(out)
			except:
				print(out_string)
				print(out)
				print(type(out), len(out), len(out[0]))
				sys.exit()
			# count = 0
			# for line in out:
			# 	count += line.count('#')
			# if count >= (15*16):
			# 	blocky += 1
			# 	write = False
			#if count >= (.5 * 15*16):
			#	print('\n'.join(out),'\n')
	return chunks

if args.mod == 'ae':
	if args.mt == 'fc':
		model = Autoencoder(input_size, output_size, args.z).to(args.device)
	else:
		model = ConvAutoencoder(num_sketch_tiles, num_tiles, args.z).to(args.device)
	opt = optim.Adam(model.parameters(), lr=0.001)
	model.load_state_dict(torch.load('trained_ae/' + args.model_name + '.pth',map_location=torch.device(args.device)))
	model.eval()
	for from_game in ['smb','ki','mm','met']: 
		if from_game == args.game:
			continue
		from_folder = path + 'data/' + game_folders_nor[from_game]
		from_levels_original, text = parse_folder(from_folder,from_game)

		h, v, e = 0, 0, 0
		for from_level in from_levels_original:
			from_translated = translate_level(from_level,from_game)
			to_level = apply_ae(model, from_translated, num_tiles, args.game, args.mt)
			to_aff = affs[args.game]
			to_jumps = jumps[args.game]
			h_path, h_goals = find_path(to_aff,to_jumps,to_level,'h')
			v_path, v_goals = find_path(to_aff,to_jumps,to_level,'v')
			if h_path:
				h += 1
			if v_path:
				v += 1
			if h_path or v_path:
				e += 1
		num_levels = len(from_levels_original)
		print(args.game)
		print(from_game, num_levels, (h*100)/num_levels, (v*100)/num_levels, (e*100)/num_levels)
else:
	# mrf_ns = '8'
	smb_mrf = pickle.load(open(f'trained_mrf/mrf_{args.ns}_smb.pickle','rb'))
	ki_mrf = pickle.load(open(f'trained_mrf/mrf_{args.ns}_ki.pickle','rb'))
	mm_mrf = pickle.load(open(f'trained_mrf/mrf_{args.ns}_mm.pickle','rb'))
	met_mrf = pickle.load(open(f'trained_mrf/mrf_{args.ns}_met.pickle','rb'))
	mrfs = {'smb':smb_mrf, 'ki':ki_mrf, 'mm':mm_mrf, 'met':met_mrf}


	for from_game in ['smb','ki','mm','met']:
		if from_game == args.game:
			continue
		print(from_game)
		chunks = []
		for level_file in os.listdir('VGLC/' + from_game + '/'):
			from_level = open('VGLC/' + from_game + '/' + level_file).read().splitlines()
			from_translated = translate_level(from_level,from_game,'mrf')
			out_level = apply_mrf(from_translated,mrfs[args.game],args.game, int(args.ns))
			#print('\n'.join(out_level))
			chunks.extend(extract_chunks(out_level))
			to_level = [''.join(l) for l in out_level]
			# print('\n'.join(to_level))
			# to_aff = affs[args.game]
			# to_jumps = jumps[args.game]
			# h_path, h_goals = find_path(to_aff,to_jumps,to_level,'h')
			# v_path, v_goals = find_path(to_aff,to_jumps,to_level,'v')
			# print(h_path)
			# print(v_path)
		print(len(chunks))
		h, v, e = 0, 0, 0
		for chunk in chunks:
			chunk = [''.join(c) for c in chunk]
			to_aff = affs[args.game]
			to_jumps = jumps[args.game]
			h_path, h_goals = find_path(to_aff,to_jumps,chunk,'h')
			v_path, v_goals = find_path(to_aff,to_jumps,chunk,'v')
			if h_path:
				h += 1
			if v_path:
				v += 1
			if h_path or v_path:
				e += 1
		num_levels = len(chunks)
		print(args.game)
		print(from_game, num_levels, (h*100)/num_levels, (v*100)/num_levels, (e*100)/num_levels)

