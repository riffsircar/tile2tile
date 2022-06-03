import pickle, os
from util import *

mrf_ns = sys.argv[1]

# To-SMB
smb_mrf = pickle.load(open('mrf_' + mrf_ns + '_' + 'smb.pickle','rb'))
for game in ['ki','mm','met']:
    for level_file in os.listdir('VGLC/' + game + '/'):
        level = open('VGLC/' + game + '/' + level_file).read().splitlines()
        translated = translate_level(level,game)
        out_level = apply_mrf(translated,smb_mrf,'smb',int(mrf_ns))
        img = get_image_from_segment(out_level)
        img.save('outputs/' + level_file + '_to_smb_' + mrf_ns + '.png')

# To-KI
ki_mrf = pickle.load(open('mrf_' + mrf_ns + '_' + 'ki.pickle','rb'))
for game in ['smb','mm','met']:
    for level_file in os.listdir('VGLC/' + game + '/'):
        level = open('VGLC/' + game + '/' + level_file).read().splitlines()
        translated = translate_level(level,game)
        out_level = apply_mrf(translated,ki_mrf,'ki',int(mrf_ns))
        img = get_image_from_segment(out_level)
        img.save('outputs/' + level_file + '_to_ki_' + mrf_ns + '.png')

# To-MM
mm_mrf = pickle.load(open('mrf_' + mrf_ns + '_' + 'mm.pickle','rb'))
for game in ['smb', 'ki', 'met']:
    for level_file in os.listdir('VGLC/' + game + '/'):
        level = open('VGLC/' + game + '/' + level_file).read().splitlines()
        translated = translate_level(level,game)
        out_level = apply_mrf(translated,mm_mrf,'mm',int(mrf_ns))
        img = get_image_from_segment(out_level)
        img.save('outputs/' + level_file + '_to_mm_' + mrf_ns + '.png')

# To-Met
met_mrf = pickle.load(open('mrf_' + mrf_ns + '_' + 'met.pickle','rb'))
for game in ['smb','ki','mm']:
    for level_file in os.listdir('VGLC/' + game + '/'):
        level = open('VGLC/' + game + '/' + level_file).read().splitlines()
        translated = translate_level(level,game)
        out_level = apply_mrf(translated,met_mrf,'met',int(mrf_ns))
        img = get_image_from_segment(out_level)
        img.save('outputs/' + level_file + '_to_met_' + mrf_ns + '.png')
