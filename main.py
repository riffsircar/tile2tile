import pickle, os
from util import *


# To-SMB
smb_mrf = pickle.load(open('mrf_smb.pickle','rb'))
for game in ['ki','mm','met']:
    for level_file in os.listdir('VGLC/' + game + '/'):
        level = open('VGLC/' + game + '/' + level_file).read().splitlines()
        translated = translate_level(level,game)
        out_level = apply_mrf(translated,smb_mrf)
        img = get_image_from_segment(out_level)
        img.save('outputs/' + level_file + '_to_smb.png')

# To-KI
ki_mrf = pickle.load(open('mrf_ki.pickle','rb'))
for game in ['smb','mm','met']:
    for level_file in os.listdir('VGLC/' + game + '/'):
        level = open('VGLC/' + game + '/' + level_file).read().splitlines()
        translated = translate_level(level,game)
        out_level = apply_mrf(translated,ki_mrf)
        img = get_image_from_segment(out_level)
        img.save('outputs/' + level_file + '_to_ki.png')

# To-MM
mm_mrf = pickle.load(open('mrf_mm.pickle','rb'))
for game in ['smb', 'ki', 'met']:
    for level_file in os.listdir('VGLC/' + game + '/'):
        level = open('VGLC/' + game + '/' + level_file).read().splitlines()
        translated = translate_level(level,game)
        out_level = apply_mrf(translated,mm_mrf)
        img = get_image_from_segment(out_level)
        img.save('outputs/' + level_file + '_to_mm.png')

# To-Met
met_mrf = pickle.load(open('mrf_met.pickle','rb'))
for game in ['smb','ki','mm']:
    for level_file in os.listdir('VGLC/' + game + '/'):
        level = open('VGLC/' + game + '/' + level_file).read().splitlines()
        translated = translate_level(level,game)
        out_level = apply_mrf(translated,met_mrf)
        print(out_level)
        img = get_image_from_segment(out_level)
        img.save('outputs/' + level_file + '_to_met.png')
