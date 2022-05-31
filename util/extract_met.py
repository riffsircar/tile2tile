import os, sys, random

dims = (15,16)
chunks = set()
blocky = 0
path = '../VGLC/Met/'
for file in os.listdir(path):
	i = 0
	level = file[file.index('_')+1]
	#if level != '3':
	#		continue
	print(file)
	data = open(path + file,'r').read().splitlines()
	data = [line.replace('\r\n','') for line in data]
	for h_offset in range(0,len(data)-dims[0]+1,1): #dims[0]):
		#out = []
		#print("H: ", h_offset)
		for w_offset in range(0,len(data[0])-(dims[1]-1),1): #dims[1]):
			write = True
			#print("W: ", w_offset)
			out = []
			for line in data[h_offset:h_offset+dims[0]]:
				out.append(line[w_offset:w_offset+dims[1]])
			if any('@' in line for line in out):
				write = False
				continue
			#print('\n'.join(out))
			count = 0
			for line in out:
				count += line.count('#')
			#(15*16):
			if count >= 240:
				blocky += 1
				write = False
				#print('\n'.join(out),'\n')
				continue
				#print('\n'.join(out),'\n')

			if write:
				out_string = '\n'.join(out)
				outfile = open('../data/met_chunks/' + file[:-4] + '_chunk_' + str(i) + '.txt','w')
				for j, line in enumerate(out):
					outfile.write(line)
					if j < len(out)-1:
						outfile.write('\n')
				outfile.close()
				i += 1

print(blocky)