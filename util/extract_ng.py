import os, sys, random

dims = (15,16)
chunks = set()
blocky = 0
for file in os.listdir('NG'):
	i = 0
	print(file)
	l1 = file[file.index('_')+1:]
	level = l1[:l1.index('.')]
	data = open('NG/' + file,'r').read().splitlines()
	data = [line.replace('\r\n','') for line in data]
	for h_offset in range(0,len(data)-dims[0]+1):#,dims[0]):
		#out = []
		#print("H: ", h_offset)
		for w_offset in range(0,len(data[0])-(dims[1]-1)):#,dims[1]):
			write = True
			#print("W: ", w_offset)
			out = []
			for line in data[h_offset:h_offset+dims[0]]:
				#print(line)
				out.append(line[w_offset:w_offset+dims[1]])
			#print(out)
			if any('@' in line for line in out):
				write = False
			
			count = 0
			for line in out:
				count += line.count('#')
			if count >= (15*16):
				blocky += 1
				write = False
			#if count >= (.5 * 15*16):
			#	print('\n'.join(out),'\n')

			if write:
				out_string = '\n'.join(out)
				#if not out_string in chunks:
					#chunks.add(out_string)
				outfile = open('ng_chunks/ng_' + level + '_chunk_' + str(i) + '.txt','w')
				for j, line in enumerate(out):
					#line = line.replace('P','-') # remove path
					outfile.write(line)
					if j < len(out)-1:
						outfile.write('\n')
				outfile.close()
				i += 1

print(blocky)