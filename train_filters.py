import os, sys
from util import *

affs = {'SMB': 'aff_smb.json', 'KI':'aff_ki.json', 'MM': 'aff_mm.json', 'Met':'aff_met.json'}
for game in ['SMB','KI','MM','Met']:
    with open(affs[game]) as f:
        aff = json.load(f)
    for level_file in os.listdir('VGLC/' + game + '/'):
        level = open('VGLC/' + game + '/' + level_file).read().splitlines()
        print(level,'\n')
        level = translate_level(level,game)
        print(level)
        sys.exit()
    sys.exit()