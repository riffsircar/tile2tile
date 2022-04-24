import sys, os

levels = []
i = 0
dims = (15,16)

path = '../VGLC/KI/'
for file in os.listdir(path):
	print(file)
	data = open(path + file,'r').read().splitlines()
	data = [line.replace('\r\n','') for line in data]
	data.reverse()
	#print(data,len(data),len(data[0]))
	#sys.exit()
	out = []
	for offset in range(0,len(data)-dims[0]+1): #,dims[0]):
		temp_data = []
		for line in data[offset:offset+dims[0]]:
			temp_data.append(line)
		temp_data.reverse()
		#sys.exit()
		out.append(temp_data)

	#out.reverse()
	#print(out[0])
	#sys.exit()

	for (_, line) in enumerate(out):
		"""
		for j, l in enumerate(line):
			if 'P' in l:
				line[j] = l.replace('P','Z')
				break
		for k, l in enumerate(reversed(line)):
			if 'P' in l:
				line[len(line)-k-1] = l.replace('P','Y')
				break
		#print(line)
		"""
		outfile = open('../game_data/ki_chunks/ki_chunk_' + str(i) + '.txt', 'w')
		temp = []
		for j,d in enumerate(line):
			outfile.write(d)
			if j < len(line)-1:
				outfile.write('\n')
		outfile.close()
		i += 1
	#sys.exit()