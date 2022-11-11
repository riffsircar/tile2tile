import torch, os, sys, random, warnings, argparse
import torch.optim as optim
import torch.utils.data
import numpy as np
from model_lin import *
from model_conv import *
from Autoencoder import *
from util import *

parser = argparse.ArgumentParser(description='Perform style transfer with autoencoder')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--cuda', type=int, default=0, help='use of cuda (default: 0)')
parser.add_argument('--epochs', type=int, default=250, help='number of total epochs to run (default: 250)')
parser.add_argument('--z', default=256, type=int, help='gaussian size (default: 256)')
parser.add_argument('--target', default='smb', type=str)
parser.add_argument('--mt', default='conv', choices=['fc','conv'], type=str, help='autoencoder type')
args = parser.parse_args()

warnings.filterwarnings("ignore")
path = ''
folder = path + 'data/' + game_folders[args.target]
#manual_seed = random.randint(1, 10000)
#random.seed(manual_seed)
## Random Seed

args.model_name = 'ae_full_' + args.mt + '_' + args.target + '_'
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

levels, text = parse_folder(folder,args.target)
text = text.replace('\n','')
# print(len(levels))
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

if args.mt == 'fc':
    model = Autoencoder(input_size, output_size, args.z).to(args.device)
else:
    model = ConvAutoencoder(num_sketch_tiles, num_tiles, args.z).to(args.device)
opt = optim.Adam(model.parameters(), lr=0.001)
model.load_state_dict(torch.load('trained_models/' + args.model_name + '.pth',map_location=torch.device(args.device)))
model.eval()

out_file = open('outs/outs_to_' + args.target + '_' + args.mt + '_' + str(args.z) + '_' + str(args.epochs) + '.txt', 'w')
for from_game in ['smb','ki','mm','met']:
    if from_game == args.target:
        continue
    print('Source: ', from_game)
    from_folder = path + 'data/' + game_folders_nor[from_game]
    from_levels, _ = parse_folder(from_folder,from_game)
    #from_samples = random.sample(from_levels, 50)
    from_samples = from_levels
    for i, sample in enumerate(from_samples):
        out_file.write(f"{i}. {from_game}-to-{args.target}: ")
        out_file.write('\n')
        sample = [''.join(row) for row in sample]
        out_file.write('\n'.join(sample))
        translated = translate_level(sample,from_game)
        out_file.write('\n\n')
        level = apply_ae(model, translated, num_tiles, args.target, args.mt)
        from_to = []
        for fr, tr in zip(sample,level):
            from_to.append(fr+'@'+tr)
        from_to_img = get_image_from_segment(from_to)
        from_to_img.save(f'output_img/{args.mt}-{from_game}-to-{args.target}-{args.epochs}-{args.z}-{i}.png')
        out_file.write('\n'.join(level))
        out_file.write('\n\n\n')
out_file.close()
# for i in range(6):
#     print(i)
#     print('SMB: ')
#     z = get_z_from_file('','filter_' + str(i) + '.txt','smb')
#     level = get_level_from_z(z,'smb')
#     get_image_from_level(level,'smb_' + str(i),'smb')
#     print('\n')
#     print('KI: ')
#     z = get_z_from_file('','filter_' + str(i) + '.txt','ki')
#     level = get_level_from_z(z,'ki')
#     get_image_from_level(level,'ki_' + str(i),'ki')
#     print('\n')
#     print('MM:')
#     z = get_z_from_file('','filter_' + str(i) + '.txt','mm')
#     level = get_level_from_z(z,'mm')
#     get_image_from_level(level,'mm_' + str(i),'mm')
# #print(level,'\n')

# #sys.exit()
# #""" 

# print('SMB-to-KI')
# level = translate_file(smb_folder,'smb_chunk_100.txt','smb')
# #get_image_from_level(level,'smb_chunk_10','smb')
# #print('Translated: \n', '\n'.join(level))
# z = get_z_from_level(level,'ki')
# level = get_level_from_z(z,'ki')
# get_image_from_level(level,'filter_ki_from_smb_chunk_100','ki')
# print('\n')

# print('KI-to-SMB')
# level = translate_file(ki_folder,'ki_chunk_100.txt','ki')
# #get_image_from_level(level,'ki_chunk_10','ki')
# #print('Translated: \n', '\n'.join(level))
# z = get_z_from_level(level,'smb')
# level = get_level_from_z(z,'smb')
# get_image_from_level(level,'smb_from_ki_chunk_100','smb')
# print('\n')

# print('SMB-to-MM')
# level = translate_file(smb_folder,'smb_chunk_10.txt','smb')
# #print('Translated: \n', '\n'.join(level))
# z = get_z_from_level(level,'mm')
# level = get_level_from_z(z,'mm')
# get_image_from_level(level,'mm_from_smb_chunk_10','mm')
# print('\n')

# print('MM-to-SMB')
# level = translate_file(mm_folder,'mm_chunk_100.txt','mm')
# #get_image_from_level(level,'mm_chunk_2000','mm')
# #print('Translated: \n', '\n'.join(level))
# z = get_z_from_level(level,'smb')
# level = get_level_from_z(z,'smb')
# get_image_from_level(level,'smb_from_mm_chunk_100','smb')
# print('\n')

# print('KI-to-MM')
# level = translate_file(ki_folder,'ki_chunk_10.txt','ki')
# #print('Translated: \n', '\n'.join(level))
# z = get_z_from_level(level,'mm')
# level = get_level_from_z(z,'mm')
# get_image_from_level(level,'mm_from_ki_chunk_10','mm')
# print('\n')

# print('MM-to-KI')
# level = translate_file(mm_folder,'mm_chunk_100.txt','mm')
# #print('Translated: \n', '\n'.join(level))
# z = get_z_from_level(level,'ki')
# level = get_level_from_z(z,'ki')
# get_image_from_level(level,'ki_from_mm_chunk_100','ki')
# print('\n')

