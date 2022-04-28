import os, re, sys, json, copy
import numpy as np
import torch
from PIL import Image

sketch_chars = ['X','E','|','*','-']
num_tiles = 39  # afford - 10, non-afford - 39
#{'#': 0, '*': 1, '+': 2, '-': 3, '<': 4, '>': 5, '?': 6, 'B': 7, 'C': 8, 'D': 9, 'E': 10, 'G': 11, 'H': 12, 'L': 13, 'M': 14, 'Q': 15, 'S': 16, 'T': 17, 'U': 18, 'W': 19, 'X': 20, '[': 21, ']': 22, '^': 23, 'b': 24, 'd': 25, 'e': 26, 'g': 27, 'h': 28, 'l': 29, 'm': 30, 'o': 31, 'r': 32, 's': 33, 't': 34, 'u': 35, 'w': 36, 'x': 37, '|': 38}

int2char = {0: '#', 1: '*', 2: '+', 3: '-', 4: '<', 5: '>', 6: '?', 7: 'B', 8: 'C', 9: 'D', 10: 'E', 11: 'G', 12: 'H', 13: 'L', 14: 'M', 15: 'Q', 16: 'S', 17: 'T', 18: 'U', 19: 'W', 20: 'X', 21: '[', 22: ']', 23: '^', 24: 'b', 25: 'd', 26: 'e', 27: 'g', 28: 'h', 29: 'l', 30: 'm', 31: 'o', 32: 'r', 33: 's', 34: 't', 35: 'u', 36: 'w', 37: 'x', 38: '|'}
images = {
    'P': Image.open('tiles/P.png'),
    '-': Image.open('tiles/-.png'),
    # SMB
    '<': Image.open('tiles/SMB_PTL.png'),
    '>': Image.open('tiles/SMB_PTR.png'),
    '[': Image.open('tiles/SMB_[.png'),
    ']': Image.open('tiles/SMB_].png'),
    '?': Image.open('tiles/SMB_Q.png'),
    'B': Image.open('tiles/SMB_B.png'),
    'b': Image.open('tiles/SMB_bb.png'),
    'G': Image.open('tiles/SMB_G.png'),
    'Q': Image.open('tiles/SMB_Q.png'),
    'S': Image.open('tiles/SMB_S.png'),
    'X': Image.open('tiles/SMB_X.png'),
    'o': Image.open('tiles/SMB_o.png'),
    'E': Image.open('tiles/SMB_E.png'),

    # KI
    '#': Image.open('tiles/KI_#.png'),  # solid
    'D': Image.open('tiles/KI_D.png'),
    'H': Image.open('tiles/KI_H.png'),
    'M': Image.open('tiles/KI_M.png'),
    'T': Image.open('tiles/KI_T.png'),

    # MM
    "*": Image.open('tiles/MM_star.png'),
    '+': Image.open('tiles/MM_+.png'),
    'C': Image.open('tiles/MM_C.png'),
    'L': Image.open('tiles/MM_ll.png'),
    'U': Image.open('tiles/MM_U.png'),
    'W': Image.open('tiles/MM_w.png'),
    'h': Image.open('tiles/MM_H.png'),
    'l': Image.open('tiles/MM_ll.png'),
    'm': Image.open('tiles/MM_M.png'),
    's': Image.open('tiles/MM_S.png'),
    't': Image.open('tiles/MM_T.png'),
    'w': Image.open('tiles/MM_w.png'),
    'x': Image.open('tiles/MM_X.png'),
    '|': Image.open('tiles/MM_L.png'),

    # Met
    '^': Image.open('tiles/Met_^2.png'),
    'd': Image.open('tiles/Met_D.png'),
    'e': Image.open('tiles/Met_E.png'),
    'g': Image.open('tiles/Met_G.png'),
    'r': Image.open('tiles/Met_R.png'),
    'u': Image.open('tiles/Met_U.png')
}

all_images = {16: images}

def get_blend_affordances():
    path = ''
    with open(path + 'aff_aff.json') as f:
        aff_aff = json.load(f)
    with open(path + 'aff_smb.json') as f:
        aff_smb = json.load(f)
        #print('smb: ', aff_smb, '\n')
    with open(path + 'aff_ki.json') as f:
        aff_ki = json.load(f)
        #print('ki: ', aff_ki, '\n')
    with open(path + 'aff_mm.json') as f:
        aff_mm = json.load(f)
        #print('mm: ', aff_mm, '\n')
    with open(path + 'aff_met.json') as f:
        aff_met = json.load(f)
        #print('met: ', aff_met, '\n')

    aff_blend = copy.deepcopy(aff_smb)
    for agent in [aff_ki, aff_mm, aff_met]:
        for key, val in agent.items():
            aff_blend[key].extend(val)
    return aff_blend

def translate_level(level, game):
    translate_func = translate[game]
    for i, row in enumerate(level):
        level[i] = translate_func(row)
    return level

def affordify(line,aff):
  a_line = ''
  for c in line:
      if c in aff['solid']:
          a_line += 'X'
      elif c in aff['breakable']:
          a_line += 'B'
      elif c in aff['hazard']:
          a_line += 'H'
      elif c in aff['enemies']:
          a_line += 'E'
      elif c in aff['collectable']:
          a_line += '*'
      elif c in aff['weapon']:
          a_line += 'W'
      elif c in aff['moving']:
          a_line += 'M'
      elif c in aff['door']:
          a_line += 'D'
      elif c in aff['climbable']:
          a_line += '|'
      else:
          a_line += '-'
  return a_line

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def parse_folder(folder,game, affordances=None, afford=False):
    levels, text = [], ''
    files = os.listdir(folder)
    files[:] = (value for value in files if value != '.')
    files = natural_sort(files)
    for file in files:
        if file.startswith('.'):
            continue
        with open(os.path.join(folder,file),'r') as infile:
            level = []
            for i, line in enumerate(infile):
                line = line.rstrip()
                if game == 'mm' or game == 'ng':
                    line = line.replace('P','-')
                if afford:
                  line = affordify(line, affordances[game])
                else:
                    if i == 14 and game == 'smb':
                        line = line.replace('X','G')
                    if game == 'mm':
                        line = line.replace('#','x')
                        line = line.replace('B','s')
                        line = line.replace('H','h')
                        line = line.replace('M','m')
                    elif game == 'met':
                        line = line.replace('#','g')
                        line = line.replace('+','u')
                        line = line.replace('B','r')
                        line = line.replace('D','d')
                        line = line.replace('E','e')
                text += line
                level.append(list(line.rstrip()))
            levels.append(level)
    return levels, text

def pipe_check(level):
    temp = ''
    for l in level:
        temp += ''.join(l)
    if '[' in temp and ']' not in temp:
        return False
    if ']' in temp and '[' not in temp:
        return False
    if '<' in temp and '>' not in temp:
        return False
    if '>' in temp and '<' not in temp:
        return False
    return True

"""
def write_segment_to_file(segment,name, folder):
    outfile = open(out_folders[GAME] + '/' + name + '.txt','w')
    for row in segment:
        outfile.write(row + '\n')
    outfile.close()

def get_image_from_segment(segment,name):
    img = Image.new('RGB',(16*16, 16*16))
    for row, seq in enumerate(segment):
        for col, tile in enumerate(seq):
            img.paste(images[tile],(col*16,row*16))
    img.save(out_folders[GAME] + '/' + name + '.png')
"""

def get_z_from_file(model,folder,f,char2int):
    print('\nInput:')
    chunk = open(folder + f, 'r').read().splitlines()
    chunk = [line.replace('\r\n','') for line in chunk]
    out = []
    for line in chunk:
        print(line)
        line_list = list(line)
        line_list_map = [char2int[x] for x in line_list]
        out.append(line_list_map)
    out = np.asarray(out)
    out_onehot = np.eye(num_tiles, dtype='uint8')[out]
    out_onehot = np.rollaxis(out_onehot, 2, 0)

    out_onehot = out_onehot[None, :, :]

    data = torch.DoubleTensor(out_onehot)
    data = data.view(data.size(0),-1)
    z, _, _ = model.encoder.encode(data)
    return z

def get_z_from_segment(model, segment, char2int):
    out = []
    for l in segment:
        l = list(l)
        l_map = [char2int[x] for x in l]
        out.append(l_map)
    out = np.asarray(out)
    out_onehot = np.eye(num_tiles, dtype='uint8')[out]
    out_onehot = np.rollaxis(out_onehot, 2, 0)
    out_onehot = out_onehot[None, :, :]
    out = torch.DoubleTensor(out_onehot)
    #out = out.to(device)
    #out_lin = out.view(out.size(0),-1)
    #z, _, _ = model.encoder.encode(out)
    out = out.view(out.size(0),-1)
    z, _, _ = model.encoder.encode(out)
    return z
    

def get_segment_from_file(folder,f):
    chunk = open(folder + f, 'r').read().splitlines()
    chunk = [line.replace('\r\n','') for line in chunk]
    return chunk
    out = []
    for line in chunk:
        line_list = list(line)
        #line_list_map = [char2int[x] for x in line_list]
        out.append(line_list)
    return out

def get_segment_from_z(model,z,int2char):
    level = model.decoder.decode(z)
    level = level.reshape(level.size(0),num_tiles,15,16)
    im = level.data.cpu().numpy()
    im = np.argmax(im, axis=1).squeeze(0)
    level = np.zeros(im.shape)
    level = []
    for i in im:
        level.append(''.join([int2char[t] for t in i]))
    return level

def get_segment_from_zc(model,z,c,int2char):
    level = model.decoder.decode(z,c)
    level = level.reshape(level.size(0),num_tiles,15,16)
    im = level.data.cpu().numpy()
    im = np.argmax(im, axis=1).squeeze(0)
    level = np.zeros(im.shape)
    level = []
    for i in im:
        level.append(''.join([int2char[t] for t in i]))
    return level

def display_generation_games(generated, int2char):
    for g in generated:
      g = g.reshape(num_tiles,15,16).argmax(axis=0)
      level = np.zeros(g.shape)
      level = []
      for i in g:
          level.append(''.join([int2char[t] for t in i]))
      print('\n'.join(level),'\n')

def get_image_from_segment(segment, tilesize=16):
    dim = (15,16)
    img = Image.new('RGB',(dim[1]*tilesize, dim[0]*tilesize))
    images_to_use = all_images[tilesize]
    for row, seq in enumerate(segment):
        for col, tile in enumerate(seq):
            img.paste(images_to_use[tile],(col*16,row*16))
    return img


def translate_ki(level):
    t_level = []
    for line in level:
        t_line = ''
        for c in line:
            if c in '#MT':
                t_line += 'X'
            elif c in 'H':
                t_line += 'E'
            elif c in 'D':
                t_line += '|'
            elif c in 'P':
                t_line += '-'
            else:
                t_line += c
        t_level.append(t_line)
    return [t_level]



def translate_smb(level):
    t_level = []
    for line in level:
        t_line = ''
        for c in line:
            if c in 'X<>[]S':
                t_line += 'X'
            elif c in 'o?Q':
                t_line += '*'
            elif c in 'Bb':
                t_line += 'E'
            else:
                t_line += c
        t_level.append(t_line)
    return [t_level]

def translate_mm(level):
    t_level = []
    for line in level:
        t_line = ''
        for c in line:
            if c in '#BM':
                t_line += 'X'
            elif c in 'CHt':
                t_line += 'E'
            elif c in 'D|':
                t_line += '|'
            elif c in '*+LUWlw':
                t_line += '*'
            elif c in 'P':
                t_line += '-'
            else:
                t_line += c
        t_level.append(t_line)
    return [t_level]

translate = {'SMB':translate_smb, 'KI':translate_ki, 'MM':translate_mm}