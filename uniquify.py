import sys, os, random, json

folder = 'data/met_chunks_3/'
chunks = set()
for i, file in enumerate(os.listdir(folder)):
	#if i % 1000 == 0:
	#	print(i, '\t', len(chunks))
	level = file[file.index('_')+1]
	data = open(folder + file,'r').read().splitlines()
	data = [line.replace('\r\n','') for line in data]
	#print(data)
	data = '\n'.join(data)
	data = data.strip()
	if data not in chunks:
		chunks.add(data)
		outfile = open('data/met_chunks_unique_3/met_chunk_' + str(i) + '.txt','w')
		for line in data:
			outfile.write(line)
		outfile.close()
	#chunks.add(data.join('\n'))

print(len(chunks))
sys.exit()
print('Writing chunks')
for i, chunk in enumerate(chunks):
	if i % 1000 == 0:
		print(i)
	outfile = open('met_chunks/unique_met_chunks/met_chunk_' + str(i) + '.txt','w')
	for line in chunk:
		outfile.write(line)
	outfile.close()