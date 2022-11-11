import pickle, os, argparse, warnings
from util import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Perform style transfer with Markov random field')
parser.add_argument('--target', default='smb', choices=['smb','ki, mm, met'], type=str)
parser.add_argument('--ns', default=4, choices=[4,8], type=int, help='MRF network size (default: 4)')

args = parser.parse_args()

mrf_model = pickle.load(open(f'trained_mrf/mrf_{args.ns}_{args.target}.pickle','rb'))
for game in ['smb','ki','mm','met']:
    if game == args.target:
        continue
    for level_file in os.listdir('VGLC/' + game + '/'):
        level = open('VGLC/' + game + '/' + level_file).read().splitlines()
        translated = translate_level(level,game,'mrf')
        out_level = apply_mrf(translated,mrf_model,args.target,int(args.ns))
        img = get_image_from_segment(out_level)
        img.save(f'outputs/{level_file}_to_{args.target}_{args.ns}.png')