import os, sys, random

def flip_room_y(room):
	return [line for line in reversed(room)]

def flip_room_x(room):
	room_t = [''.join(s) for s in zip(*room)]
	flipped = flip_room_y(room_t)
	room_tt = [''.join(s) for s in zip(*flipped)]
	return room_tt

def flip_room_both(room):
	return flip_room_x(flip_room_y(room))

dims = (11,16)
rooms = set()
for file in os.listdir('zelda_fixed'):
	i = 0
	print(file)
	level = file[file.index('_')+1]
	data = open('zelda_fixed/' + file,'r').read().splitlines()
	data = [line.replace('\r\n','') for line in data]
	for h_offset in range(0,len(data)-dims[0]+1,dims[0]):
		#out = []
		#print("H: ", h_offset)
		for w_offset in range(0,len(data[0])-(dims[1]-1),dims[1]):
			write = True
			#print("W: ", w_offset)
			out = []
			for line in data[h_offset:h_offset+dims[0]]:
				#print(line)
				out.append(line[w_offset:w_offset+dims[1]])
			#print(out)
			if any('-' in line for line in out):
				write = False

			if write:
				outs = [out, flip_room_x(out), flip_room_y(out), flip_room_both(out)]
				out_strings = ['\n'.join(out) for out in outs]
				for k, out_string in enumerate(out_strings):
					#print(out_string)
					if not out_string in rooms:
						rooms.add(out_string)
						outfile = open('zelda_rooms_new/' + file[4:-10] + '_room_' + str(i) + '.txt','w')
						for j, line in enumerate(outs[k]):
							outfile.write(line)
							if j < len(outs[k])-1:
								outfile.write('\n')
						outfile.close()
						i += 1

print(len(rooms))