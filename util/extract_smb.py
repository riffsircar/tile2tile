import sys
import os
import random
import json

path = '../VGLC/SMB/'
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

levels = []
i = 0
dims = (15,16)
failed = 0
for file in os.listdir(path):
	print(file)
	data = open(path + file,'r').read().splitlines()
	data = [line.replace('\r\n','') for line in data]
	out = []
	for offset in range(0,len(data[0])-dims[1]+1):#,dims[1]):
		temp_out = []
		for line in data:
			temp_out.append(line[offset:offset+dims[1]])
		out.append(temp_out)
	
	for (_,line) in enumerate(out):
		if pipe_check(line):
			outfile = open('../game_data/smb_chunks/smb_chunk_' + str(i) + '.txt','w')
			temp = []
			for j,d in enumerate(line):
				outfile.write(d)
				if j < len(line)-1:
					outfile.write('\n')

			outfile.close()
			i += 1
		else:
			failed += 1
			print('Pipe check failed!')
			print('\n'.join(line))
			print('\n')
print(failed)
