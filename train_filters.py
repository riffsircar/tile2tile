import os, sys, pickle
from util import *

#affs = {'SMB': 'aff_smb.json', 'KI':'aff_ki.json', 'MM': 'aff_mm.json', 'Met':'aff_met.json'}

for game in ['smb','ki','mm','met']:
    print(game)
    levels, translated_levels = [], []
    markov_counts, markov_probs = {}, {}
    # with open(affs[game]) as f:
    #     aff = json.load(f)
    for level_file in os.listdir('VGLC/' + game + '/'):
        level = open('VGLC/' + game + '/' + level_file).read().splitlines()
        level = preprocess_level(level,game)
        translated_level = translate_level(level,game)
        levels.append(level)
        translated_levels.append(translated_level)

    # translation has to predict tile given affordance neighborhood
    for level, translated in zip(levels, translated_levels):
        for row in range(1, len(translated)-1):
            for col in range(1,len(translated[0])-1):
                #print(row, col)
                north = translated[row-1][col]
                south = translated[row+1][col]
                east = translated[row][col-1]
                west = translated[row][col+1]
                
                tile = level[row][col]
                
                context = north+south+east+west
                if context not in markov_counts:
                    markov_counts[context] = {}
                if tile not in markov_counts[context]:
                    markov_counts[context][tile] = 0
                markov_counts[context][tile] += 1
        #print(len(level),len(level[0]),len(translated),len(translated[0]))
    print(markov_counts)

    for context in markov_counts:
        markov_probs[context] = {}
        freq_sum = sum(markov_counts[context].values())
        for tile, count in markov_counts[context].items():
            markov_probs[context][tile] = count/freq_sum
    print('\n',markov_probs,'\n','\n')
    
    pickle.dump(markov_probs, open('mrf_' + game + '.pickle', 'wb'))