import os, sys, random

dims = (11,16)
dirs = ['DR','DL','UR','UL']
chunks = set()
blocky = 0
for file in os.listdir('lode_runner'):
	i = 0
	print(file)
	# translation: '.' -> '-', '-' -> 'R', '#' -> 'L'
	level = file[file.index(' ')+1]
	data = open('lode_runner/' + file,'r').read().splitlines()
	data = [line.replace('\r\n','') for line in data]
	for h_offset in range(0,len(data)-dims[0]+1,dims[0]):
		#out = []
		#print("H: ", h_offset)
		for w_offset in range(0,len(data[0])-(dims[1]-1),dims[1]):
			write = True
			#print("W: ", w_offset)
			out = []
			for line in data[h_offset:h_offset+dims[0]]:
				line = line.replace('-','R')
				line = line.replace('.','-')
				line = line.replace('#','L')
				#print(line)
				out.append(line[w_offset:w_offset+dims[1]])
						
			count = 0
			
			
			
			out_string = '\n'.join(out)
			#if not out_string in chunks:
				#chunks.add(out_string)
			outfile = open('lode_runner_chunks/' + file[:-4] + '_chunk_' + str(i) + '_' + dirs[i] + '.txt','w')
			for j, line in enumerate(out):
				outfile.write(line)
				if j < len(out)-1:
					outfile.write('\n')
			outfile.close()
			i += 1

