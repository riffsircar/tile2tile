import pathfinding, sys
from heapq import heappush, heappop

def makeIsSolid(solids):
	def isSolid(tile):
		return tile in solids
	return isSolid

def makeIsPassable(passables, hazards):
	def isPassable(tile):
		return tile in passables and tile not in hazards
	return isPassable

def makeIsHazard(hazards):
	def isHazard(tile):
		return tile in hazards
	return isHazard

def makeIsClimbable(climbables):
	def isClimbable(tile):
		return tile in climbables
	return isClimbable

def makeIsValid(solids,passables,hazards,climbables):
	def isValid(tile):
		return (tile in solids) or (tile in passables) or (tile in hazards) or (tile in climbables)
	return isValid

#Make sure we are getting the proper neighbors and that all checks are happending appropriately
def makeGetNeighbors(jumps,levelStr,visited,isSolid,isPassable,isClimbable,isHazard,isValid,level_wrap=False):
	maxX = len(levelStr[0])
	maxY = len(levelStr)-1
	jumpDiffs = []
	for jump in jumps:
		jumpDiff = [jump[0]]
		for ii in range(1,len(jump)):
			jumpDiff.append((jump[ii][0]-jump[ii-1][0],jump[ii][1]-jump[ii-1][1]))
		jumpDiffs.append(jumpDiff)
	jumps = jumpDiffs

	def getNeighbors(pos):
		# print('pos: ', pos)
		dist = pos[0]-pos[2] 
		pos = pos[1] 
		visited.add((pos[0],pos[1])) 
		below = (pos[0],pos[1]+1) 
		# print('dist: ', dist)
		# print('pos: ', pos)
		# print('below: ', below)
		# print('jumps: ', jumps)

		neighbors = []
		#if the player falls to the bottom of the level
		if below[1] > maxY or isHazard(levelStr[below[1]][below[0]]):
			return []
		if pos[2] != -1:
			ii = pos[3] +1
			jump = pos[2]

			if ii < len(jumps[jump]):
				jump_pos_0 = pos[0]+pos[4]*jumps[jump][ii][0]
				jump_pos_1 = pos[1]+jumps[jump][ii][1]
				if level_wrap:
					if  (pos[1]+jumps[jump][ii][1] >= 0) and (not isSolid(levelStr[pos[1]+jumps[jump][ii][1]][(pos[0]+pos[4]*jumps[jump][ii][0])%maxX]) or isPassable((levelStr[pos[1]+jumps[jump][ii][1]][(pos[0]+pos[4]*jumps[jump][ii][0])%maxX]))):
						neighbors.append([dist+1,((pos[0]+pos[4]*jumps[jump][ii][0])%maxX,pos[1]+jumps[jump][ii][1],jump,ii,pos[4])])
						#print("mid jump")
					if 	pos[1]+jumps[jump][ii][1] < 0 and (not isSolid(levelStr[pos[1]+jumps[jump][ii][1]][(pos[0]+pos[4]*jumps[jump][ii][0])%maxX]) or isPassable(levelStr[pos[1]+jumps[jump][ii][1]][(pos[0]+pos[4]*jumps[jump][ii][0])%maxX])):
						neighbors.append([dist+1,((pos[0]+pos[4]*jumps[jump][ii][0])%maxX,0,jump,ii,pos[4])])
						#print("mid fall")
				else:
					#if  pos[1]+jumps[jump][ii][1] >= 0 and (pos[0]+pos[4]*jumps[jump][ii][0]) < maxX and (pos[0]+pos[4]*jumps[jump][ii][0]) >= 0 and (not isSolid(levelStr[pos[1]+jumps[jump][ii][1]][(pos[0]+pos[4]*jumps[jump][ii][0])]) or isPassable((levelStr[pos[1]+jumps[jump][ii][1]][(pos[0]+pos[4]*jumps[jump][ii][0])]))):
					#if  jump_pos_1 >= 0 and (jump_pos_0) < maxX and (jump_pos_0) >= 0 and (not isSolid(levelStr[jump_pos_1][jump_pos_0]) or isPassable((levelStr[jump_pos_1][jump_pos_0]))):
					if  jump_pos_1 >= 0 and (jump_pos_0) < maxX and (jump_pos_0) >= 0 and (isValid(levelStr[jump_pos_1][jump_pos_0]) and (not isSolid(levelStr[jump_pos_1][jump_pos_0]) or isPassable(levelStr[jump_pos_1][jump_pos_0]))):
						neighbors.append([dist+1,((pos[0]+pos[4]*jumps[jump][ii][0]),jump_pos_1,jump,ii,pos[4])])
						#print("mid jump")
					#if 	pos[1]+jumps[jump][ii][1] < 0 and (pos[0]+pos[4]*jumps[jump][ii][0]) < maxX and (pos[0]+pos[4]*jumps[jump][ii][0]) >= 0 and (not isSolid(levelStr[pos[1]+jumps[jump][ii][1]][pos[0]+pos[4]*jumps[jump][ii][0]]) or isPassable(levelStr[pos[1]+jumps[jump][ii][1]][(pos[0]+pos[4]*jumps[jump][ii][0])])):
					if 	jump_pos_1 < 0 and (jump_pos_0) < maxX and (jump_pos_0) >= 0 and (isValid(levelStr[jump_pos_1][jump_pos_0]) and (not isSolid(levelStr[jump_pos_1][jump_pos_0]) or isPassable(levelStr[jump_pos_1][jump_pos_0]))):
						neighbors.append([dist+1,(jump_pos_0,0,jump,ii,pos[4])])
						#print("mid fall")
				
		if isValid(levelStr[below[1]][below[0]]) and isSolid(levelStr[below[1]][below[0]]) and not isHazard(levelStr[below[1]][below[0]]):
			if level_wrap:
				if not isSolid(levelStr[pos[1]][(pos[0]+1)%maxX]):
					neighbors.append([dist+1,((pos[0]+1)%maxX,pos[1],-1)])
					#print("move right")
				if not isSolid(levelStr[pos[1]][(pos[0]-1)%maxX]):
					neighbors.append([dist+1,((pos[0]-1)%maxX,pos[1],-1)])
					#print("move left")

				for jump in range(len(jumps)):
					ii = 0
					if pos[1] >= 0 and (not isSolid(levelStr[pos[1]+jumps[jump][ii][1]][(pos[0]+jumps[jump][ii][0])%maxX]) or isPassable(levelStr[pos[1]+jumps[jump][ii][1]][(pos[0]+jumps[jump][ii][0])%maxX])):
						neighbors.append([dist+ii+1,((pos[0]+jumps[jump][ii][0])%maxX,pos[1]+jumps[jump][ii][1],jump,ii,1)])
						#print("start jump right")

					if pos[1] >= 0 and (not isSolid(levelStr[pos[1]+jumps[jump][ii][1]][(pos[0]-jumps[jump][ii][0])%maxX]) or isPassable(levelStr[pos[1]+jumps[jump][ii][1]][(pos[0]-jumps[jump][ii][0])%maxX])):
						neighbors.append([dist+ii+1,((pos[0]-jumps[jump][ii][0])%maxX,pos[1]+jumps[jump][ii][1],jump,ii,-1)])
						#print("start jump left")
			else:
				#if pos[0]+1 < maxX and not isSolid(levelStr[pos[1]][(pos[0]+1)]):
				if pos[0]+1 < maxX and not isSolid(levelStr[pos[1]][(pos[0]+1)]) and isValid(levelStr[pos[1]][(pos[0]+1)]):
					neighbors.append([dist+1,((pos[0]+1),pos[1],-1)])
					#print("move right")
				#if pos[0]-1 >= 0 and not isSolid(levelStr[pos[1]][(pos[0]-1)]):
				if pos[0]-1 >= 0 and not isSolid(levelStr[pos[1]][(pos[0]-1)]) and isValid(levelStr[pos[1]][(pos[0]-1)]):
					neighbors.append([dist+1,((pos[0]-1),pos[1],-1)])
					#print("move left")

				for jump in range(len(jumps)):
					ii = 0
					#print(pos[1]+jumps[jump][ii][1], pos[0]-jumps[jump][ii][0], len(levelStr), len(levelStr[0]))
					#print(levelStr[pos[1]+jumps[jump][ii][1]])
					#print(isSolid(levelStr[pos[1]+jumps[jump][ii][1]][pos[0]-jumps[jump][ii][0]]))
					#print(isPassable(levelStr[pos[1]+jumps[jump][ii][1]][(pos[0]-jumps[jump][ii][0])]))
					jump_pos_plus_0 = pos[0]+jumps[jump][ii][0]
					jump_pos_minus_0 = pos[0]-jumps[jump][ii][0]
					jump_pos_plus_1 = pos[1]+jumps[jump][ii][1]
					#if (pos[1] >= 0 and (jump_pos_plus_0) < maxX and (jump_pos_plus_0) >=0) and (not isSolid(levelStr[jump_pos_plus_1][jump_pos_plus_0]) or isPassable(levelStr[jump_pos_plus_1][jump_pos_plus_0])):
					if (pos[1] >= 0 and (jump_pos_plus_0) < maxX and (jump_pos_plus_0) >=0) and (isValid(levelStr[jump_pos_plus_1][jump_pos_plus_0]) and (not isSolid(levelStr[jump_pos_plus_1][jump_pos_plus_0]) or isPassable(levelStr[jump_pos_plus_1][jump_pos_plus_0]))):
						neighbors.append([dist+ii+1,(jump_pos_plus_0,jump_pos_plus_1,jump,ii,1)])
						#print("start jump right")

					#if (pos[1] >= 0 and (jump_pos_minus_0) < maxX and (jump_pos_minus_0) >= 0) and (not isSolid(levelStr[jump_pos_plus_1][jump_pos_minus_0]) or isPassable(levelStr[jump_pos_plus_1][jump_pos_minus_0])):
					if (pos[1] >= 0 and (jump_pos_minus_0) < maxX and (jump_pos_minus_0) >= 0) and (isValid(levelStr[jump_pos_plus_1][jump_pos_minus_0]) and (not isSolid(levelStr[jump_pos_plus_1][jump_pos_minus_0]) or isPassable(levelStr[jump_pos_plus_1][jump_pos_minus_0]))):
						neighbors.append([dist+ii+1,(jump_pos_minus_0,jump_pos_plus_1,jump,ii,-1)])

		if isValid(levelStr[below[1]][below[0]]) and (not isSolid(levelStr[below[1]][below[0]]) or isPassable(levelStr[below[1]][below[0]])):
			neighbors.append([dist+1,(pos[0],pos[1]+1,-1)])
			if level_wrap:
				if pos[1]+1 <= maxY:
					if not isSolid(levelStr[pos[1]+1][(pos[0]+1)%maxX]) or isPassable(levelStr[pos[1]+1][(pos[0]+1)%maxX]):
						neighbors.append([dist+1.4,((pos[0]+1)%maxX,pos[1]+1,-1)])
						#print("falling right")
					if not isSolid(levelStr[pos[1]+1][(pos[0]-1)%maxX]) or isPassable(levelStr[pos[1]+1][(pos[0]-1)%maxX]):
						neighbors.append([dist+1.4,((pos[0]-1)%maxX,pos[1]+1,-1)])
						#print("falling left")
					if not isSolid(levelStr[pos[1]+1][pos[0]]) or isPassable(levelStr[pos[1]+1][pos[0]]):
						neighbors.append([dist+1,(pos[0],pos[1]+1,-1)])
						#falling straight down
				if pos[1]+2 <= maxY:
					if (not isSolid(levelStr[pos[1]+2][(pos[0]+1)%maxX]) or isPassable(levelStr[pos[1]+2][(pos[0]+1)%maxX])) and (not isSolid(levelStr[pos[1]+1][(pos[0]+1)%maxX]) or isPassable(levelStr[pos[1]+1][(pos[0]+1)%maxX])):
						neighbors.append([dist+2,((pos[0]+1)%maxX,pos[1]+2,-1)])
						#print("falling right fast")
					if (not isSolid(levelStr[pos[1]+2][(pos[0]-1)%maxX]) or isPassable(levelStr[pos[1]+2][(pos[0]-1)%maxX])) and (not isSolid(levelStr[pos[1]+1][(pos[0]-1)%maxX]) or isPassable(levelStr[pos[1]+1][(pos[0]-1)%maxX])):
						neighbors.append([dist+2,((pos[0]-1)%maxX,pos[1]+2,-1)])
						#print("falling left fast")
				#	if (not isSolid(levelStr[pos[1]+2][pos[0]]) or isPassable(levelStr[pos[1]+2][pos[0]])) and (not isSolid(levelStr[pos[1]+1][pos[0]]) or isPassable(levelStr[pos[1]+1][pos[0]])):
				#		neighbors.append([dist+2,(pos[0],pos[1]+2,-1)])
						#falling straight down fast
			else:
				if pos[1]+1 <= maxY:
					#if pos[0]+1 < maxX and (not isSolid(levelStr[pos[1]+1][pos[0]+1]) or isPassable(levelStr[pos[1]+1][pos[0]+1])):
					if pos[0]+1 < maxX and (isValid(levelStr[pos[1]+1][pos[0]+1]) and (not isSolid(levelStr[pos[1]+1][pos[0]+1]) or isPassable(levelStr[pos[1]+1][pos[0]+1]))):
						neighbors.append([dist+1.4,(pos[0]+1,pos[1]+1,-1)])
						#print("falling right")
					#if pos[0]-1 >= 0 and (not isSolid(levelStr[pos[1]+1][pos[0]-1]) or isPassable(levelStr[pos[1]+1][pos[0]-1])):
					if pos[0]-1 >= 0 and (isValid(levelStr[pos[1]+1][pos[0]-1]) and (not isSolid(levelStr[pos[1]+1][pos[0]-1]) or isPassable(levelStr[pos[1]+1][pos[0]-1]))):
						neighbors.append([dist+1.4,(pos[0]-1,pos[1]+1,-1)])
						#print("falling left")
					#if not isSolid(levelStr[pos[1]+1][pos[0]]) or isPassable(levelStr[pos[1]+1][pos[0]]):
					if isValid(levelStr[pos[1]+1][pos[0]]) and (not isSolid(levelStr[pos[1]+1][pos[0]]) or isPassable(levelStr[pos[1]+1][pos[0]])):
						neighbors.append([dist+1,(pos[0],pos[1]+1,-1)])
						#falling straight down
				if pos[1]+2 <= maxY:
					#if pos[0]+1 < maxX and (not isSolid(levelStr[pos[1]+2][pos[0]+1]) or isPassable(levelStr[pos[1]+2][pos[0]+1])) and (not isSolid(levelStr[pos[1]+1][pos[0]+1]) or isPassable(levelStr[pos[1]+1][pos[0]+1])):
					if pos[0]+1 < maxX and (isValid(levelStr[pos[1]+2][pos[0]+1]) and (not isSolid(levelStr[pos[1]+2][pos[0]+1]) or isPassable(levelStr[pos[1]+2][pos[0]+1]))) and (isValid(levelStr[pos[1]+1][pos[0]+1]) and (not isSolid(levelStr[pos[1]+1][pos[0]+1]) or isPassable(levelStr[pos[1]+1][pos[0]+1]))):
						neighbors.append([dist+2,(pos[0]+1,pos[1]+2,-1)])
						#print("falling right fast")
					#if pos[0]-1 >= 0 and (not isSolid(levelStr[pos[1]+2][pos[0]-1]) or isPassable(levelStr[pos[1]+2][pos[0]-1])) and (not isSolid(levelStr[pos[1]+1][pos[0]-1]) or isPassable(levelStr[pos[1]+1][pos[0]-1])):
					if pos[0]-1 >= 0 and (isValid(levelStr[pos[1]+2][pos[0]-1]) and (not isSolid(levelStr[pos[1]+2][pos[0]-1]) or isPassable(levelStr[pos[1]+2][pos[0]-1]))) and (isValid(levelStr[pos[1]+1][pos[0]-1]) and (not isSolid(levelStr[pos[1]+1][pos[0]-1]) or isPassable(levelStr[pos[1]+1][pos[0]-1]))):
						neighbors.append([dist+2,(pos[0]-1,pos[1]+2,-1)])
						#print("falling left fast")
					#if (not isSolid(levelStr[pos[1]+2][pos[0]]) or isPassable(levelStr[pos[1]+2][pos[0]])) and (not isSolid(levelStr[pos[1]+1][pos[0]]) or isPassable(levelStr[pos[1]+1][pos[0]])):
					#	neighbors.append([dist+2,(pos[0],pos[1]+2,-1)])
						#falling straight down fast
		
		#if currently on a climbable tile, see if we can climb in any direction
		if isClimbable(levelStr[pos[1]][pos[0]]) and isValid(levelStr[pos[1]][pos[0]]):
			#up
			#if pos[1]+1 <=maxY and (isClimbable(levelStr[below[1]][below[0]]) or not isSolid(levelStr[below[1]][below[0]]) or isPassable(levelStr[below[1]][below[0]])):
			if pos[1]+1 <=maxY and (isValid(levelStr[below[1]][below[0]]) and (isClimbable(levelStr[below[1]][below[0]]) or not isSolid(levelStr[below[1]][below[0]]) or isPassable(levelStr[below[1]][below[0]]))):
				neighbors.append([dist+1, (pos[0], pos[1]+1,-1)])
			#down
			#if pos[1]-1 >= 0 and (isClimbable(levelStr[pos[1]-1][pos[0]]) or not isSolid(levelStr[pos[1]-1][pos[0]]) or isPassable(levelStr[pos[1]-1][pos[0]])):
			if pos[1]-1 >= 0 and (isValid(levelStr[pos[1]-1][pos[0]]) and (isClimbable(levelStr[pos[1]-1][pos[0]]) or not isSolid(levelStr[pos[1]-1][pos[0]]) or isPassable(levelStr[pos[1]-1][pos[0]]))):
				neighbors.append([dist+1, (pos[0], pos[1]-1,-1)])

			if level_wrap:
				#left
				if isClimbable(levelStr[pos[1]][(pos[0]-1)%maxX]) or (not isSolid(levelStr[pos[1]][(pos[0]-1)%maxX]) or isPassable(levelStr[pos[1]][(pos[0]-1)%maxX])):
					neighbors.append([dist+1, ((pos[0]-1)%maxX, pos[1],-1)])
				#right
				if isClimbable(levelStr[pos[1]][(pos[0]+1)%maxX]) or (not isSolid(levelStr[pos[1]][(pos[0]+1)%maxX]) or isPassable(levelStr[pos[1]][(pos[0]+1)%maxX])):
					neighbors.append([dist+1, ((pos[0]+1)%maxX, pos[1],-1)])
			else:
				#left
				#if pos[0]-1 >= 0 and (isClimbable(levelStr[pos[1]][pos[0]-1]) or not isSolid(levelStr[pos[1]][pos[0]-1]) or isPassable(levelStr[pos[1]][pos[0]-1])):
				if pos[0]-1 >= 0 and (isValid(levelStr[pos[1]][pos[0]-1]) and (isClimbable(levelStr[pos[1]][pos[0]-1]) or not isSolid(levelStr[pos[1]][pos[0]-1]) or isPassable(levelStr[pos[1]][pos[0]-1]))):
					neighbors.append([dist+1, (pos[0]-1, pos[1],-1)])
				#right
				#if pos[0]+1 < maxX and (isClimbable(levelStr[pos[1]][pos[0]+1]) or not isSolid(levelStr[pos[1]][pos[0]+1]) or isPassable(levelStr[pos[1]][pos[0]+1])):
				if pos[0]+1 < maxX and (isValid(levelStr[pos[1]][pos[0]+1]) and (isClimbable(levelStr[pos[1]][pos[0]+1]) or not isSolid(levelStr[pos[1]][pos[0]+1]) or isPassable(levelStr[pos[1]][pos[0]+1]))):
					neighbors.append([dist+1, (pos[0]+1, pos[1],-1)])


		return neighbors
	return getNeighbors

def goalReached(current_width, current_height, goal_width, goal_height):
	#If no goal specified
	if goal_height == None and goal_width == None:
		print("No goal specified; so we reached it!")
		return True
	#If only a horizontal goal
	elif goal_height == None and goal_width != None:
		#print(str(current_width)+ " "+str(goal_width))
		if current_width == goal_width:
			#print(current_width, goal_width)
			print("horizontal goal reached!")
			return True
		else:
			return False
	#If only a vertical goal
	elif goal_height != None and goal_width == None:
		if current_height == goal_height:
			print("vertical goal reached!")
			return True
		else:
			return False
	#If an exact positional goal
	elif goal_height != None and goal_width != None:
		if current_height == goal_height and current_width == goal_width:
			print("positional goal reached!")
			return True
		else:
			return False

def tileDistance(X_1, Y_1, X_2, Y_2):
	if X_2 == None:
		return abs(Y_2 - Y_1)
	elif Y_2 == None:
		return abs(X_2 - X_1)
	else:
		return abs(X_2 - X_1) + abs(Y_2 - Y_1)

# def find_goals_mario(level, start_pos, solids):
# 	goals = set()
# 	for ic in range(15, 2, -1):
# 		if ic <= start_pos[0] + 1:
# 			return None
		
# 		found_solid = False
# 		for ir in range(len(level) - 1, -1, -1):
# 			if level[ir][ic] in solids:
# 				found_solid = True
# 			else:
# 				if found_solid:
# 					goals.add((ic, ir))
# 		if len(goals) != 0:
# 			return goals
# 	return None

def find_start_mario(level, solids, passables):
	col = -1
	row = None
	while row == None:
		col += 1
		if col >= (len(level[0]))/2: # if start goes beyond half the segment, then fail
			return None
		for y in range(len(level)-2, 0, -1):
			if level[y][col] in passables and level[y + 1][col] in solids:
				row = y
				return (col, row, -1)	

def find_goals_mario(level, start, solids, passables):
	col = 16
	row = None
	goals = set()
	while row == None:
		col -= 1
		if col <= start[0] + 1:
			return None
		if col == -1:
			return None
		for row in range(len(level)-2, 0, -1):
			if level[row][col] in passables:
				goals.add((col,row))
		if len(goals) != 0:
			return goals
	return None

def find_start_icarus(level, solids, passables):
	for ir in range(len(level) - 2, 0, -1):
		for ic in range(len(level[0])):
			if level[ir][ic] in passables and level[ir + 1][ic] in solids:
				return (ic, ir, -1)
	return None

def find_goals_icarus(level, start_pos, solids, passables):
	goals = set()
	#print('sp: ', start_pos)
	#for ir in range(len(level) - 1 - index, len(level) - 1):
	#for ir in range(0, len(level) - 2):
	for ir in range(0, len(level)//2):
		if ir >= start_pos[1] - 1:
			return None
		
		for ic in range(len(level[0])):
			#if not level[ir][ic] in solids:
			if level[ir][ic] in passables and level[ir+1][ic] in solids:
				goals.add((ic, ir))
		if len(goals) != 0:
			return goals
	return None

# def find_goals_icarus(level, start, solids):
# 	goals = set()
# 	#print('sp: ', start_pos)
# 	#for ir in range(len(level) - 1 - index, len(level) - 1):
# 	#for ir in range(5, len(level) - 1):
# 	for ir in range(0, 6):  # check first 5 rows for goals, else return no goal
# 		if ir >= start[1] - 1:
# 			return None
		
# 		for ic in range(len(level[0])):
# 			if not level[ir][ic] in solids:
# 				goals.add((ic, ir))
# 		if len(goals) != 0:
# 			return goals
	
# 	# goalX_v = None
# 	# goalY_v = 0
# 	# while goalX_v == None:
# 	# 	goalY_v += 1
# 	# 	#if goalY_v == 14:
# 	# 	if goalY_v == 5 or goalY_v > startY_v:
# 	# 		break
# 	# 	for x in range(1, maxX):
# 	# 		if not isSolid(levelStr[goalY_v][x]) and isSolid(levelStr[goalY_v + 1][x]):
# 	# 			goalX_v = x
# 	# 			break
# 	return None



def find_path(affordances, jumps,levelStr, dir):
	visited = set()
	solids, passables, climbables, hazards = affordances['solid'], affordances['passable'], affordances['climbable'], affordances['hazard']
	isSolid = makeIsSolid(solids)
	isPassable = makeIsPassable(passables, hazards)
	isHazard = makeIsHazard(hazards)
	isClimbable = makeIsClimbable(climbables)
	isValid = makeIsValid(solids, passables, hazards, climbables)
	getNeighborsUnwrapped = makeGetNeighbors(jumps,levelStr,visited,isSolid,isPassable,isClimbable,isHazard,isValid,False)
	#getNeighborsWrapped = makeGetNeighbors(jumps,levelStr,visited,isSolid,isPassable,isClimbable,isHazard,False)

	heuristic = lambda pos: 0.0 # TODO: distance to goal locations?
	#start_vert = find_start_icarus(levelStr, solids, passables)
	#start_hor = find_start_mario(levelStr, solids, passables)
	#starts = [(0,0,-1), start_vert]
	#starts = [start_hor, start_vert]
	#starts = [start_hor] if dir == 'h' else [start_vert]
	#goals_h = find_goals_mario(levelStr, (0,0), solids)
	if dir == 'h':
		start = find_start_mario(levelStr, solids, passables)
		goals = None if start is None else find_goals_mario(levelStr, start, solids, passables)
	else:
		start = find_start_icarus(levelStr, solids, passables)
		goals = None if start is None else find_goals_icarus(levelStr, start, solids, passables)
	#goals_h = None if start_hor is None else find_goals_mario(levelStr, start_hor, solids, passables)
	#goals_v = None if start_vert is None else find_goals_icarus(levelStr, start_vert, solids, passables)
	#goals = goals_h if dir == 'h' else goals_v
	DEBUG_DISPLAY = False
	dist = {}
	prev = {}
	paths = []
	getNeighbors = getNeighborsUnwrapped
	#for i, start in enumerate(starts):
	#goals = goals_h if i == 0 else goals_v
	if goals is None:
		return None, goals
		#paths.append(None)
		#continue
	
	dist[start] = 0
	prev[start] = None
	heap = [(dist[start], start, 0)]
	end_node = None

	# if DEBUG_DISPLAY:
	# 	explored = set()
	# 	path = set()

	# 	def displayLevel():
	# 		print()
	# 		for yy, row in enumerate(levelStr):
	# 			for xx, tile in enumerate(row):
	# 				if (xx, yy) in path:
	# 					sys.stdout.write('!')
	# 				elif (xx, yy) in explored:
	# 					sys.stdout.write('.')
	# 				else:
	# 					sys.stdout.write(tile)
	# 			sys.stdout.write('\n')

	#if i == 0 else getNeighborsWrapped
	while heap:
		node = heappop(heap)

		# if DEBUG_DISPLAY:
		# 	explored.add((node[1][0], node[1][1]))
			#displayLevel()
			
		for next_node in getNeighbors(node):
			#if i == 1:
			#	print('next: ', next_node)
			next_node[0] += heuristic(next_node[1])
			next_node.append(heuristic(next_node[1]))
			if next_node[1] not in dist or next_node[0] < dist[next_node[1]]:
				dist[next_node[1]] = next_node[0]
				prev[next_node[1]] = node[1]
				heappush(heap, next_node)

				next_pos = (next_node[1][0], next_node[1][1])

				if next_pos in goals:

					# if DEBUG_DISPLAY:
					# 	full_path = []
					# 	path_node = next_node[1]

					# 	while path_node != None:
					# 		path.add((path_node[0], path_node[1]))
					# 		full_path.append(path_node)
							
					# 		if path_node == next_node[1]:
					# 			path_node = node[1]
					# 		else:
					# 			path_node = prev[path_node]

					# 	print('path', list(reversed(full_path)))
					# displayLevel()
			
					end_node = next_node[1]
					break
				
		if end_node != None:
			break

	# if DEBUG_DISPLAY:
	# 	displayLevel()

	if end_node == None:
		#paths.append(None)
		return None, goals
	else:
		path = []
		curr_node = end_node
		while curr_node != None:
			path.append(curr_node)
			curr_node = prev[curr_node]
		#paths.append(list(reversed(path)))
	#print('Start_h:' , start_hor, '\tGoals_h: ', goals_h)
	#print('Start_v:', start_vert, '\tGoals_v: ', goals_v)
	#print(paths)
	#return paths[0],paths[1], goals_h, goals_v
	#return paths[0], goals
	return list(reversed(path)), goals

def findPaths(affordances, jumps,levelStr,is_vertical=False):
	visited = set()
	solids, passables, climbables, hazards = affordances['solid'], affordances['passable'], affordances['climbable'], affordances['hazard']
	isSolid = makeIsSolid(solids)
	isPassable = makeIsPassable(passables, hazards)
	isHazard = makeIsHazard(hazards)
	isClimbable = makeIsClimbable(climbables)
	getNeighbors = makeGetNeighbors(jumps,levelStr,visited,isSolid,isPassable,isClimbable,isHazard)
	getNeighborsWrapped = makeGetNeighbors(jumps,levelStr,visited,isSolid,isPassable,isClimbable,isHazard,True)
	#maxX = len(levelStr[0])-1
	maxX = len(levelStr[0]) - 1
	maxY = len(levelStr) - 1

	"""	start_ic = find_start_icarus(levelStr, solids)
	goals_ic = find_goals_icarus(levelStr, (start_ic[0], start_ic[1]), solids)
	goals_sm = find_goals_mario(levelStr, (0,0), solids)
	print('\n','\n'.join(levelStr))
	print("ICS: ", start_ic)
	print("IC: ", goals_ic)
	print("SM: ", goals_sm)
	return
	"""
	# vertical start and goal
	startX_v = None
	startY_v = maxY
	while startX_v == None:
		startY_v -= 1
		if startY_v < 1:
			break
		for x in range(1, maxX):
			#print(x, startY_v)
			if not isSolid(levelStr[startY_v][x]) and isSolid(levelStr[startY_v + 1][x]):
				startX_v = x
				break

	goalX_v = None
	goalY_v = 0
	while goalX_v == None:
		goalY_v += 1
		#if goalY_v == 14:
		if goalY_v == 5 or goalY_v > startY_v:
			break
		for x in range(1, maxX):
			if not isSolid(levelStr[goalY_v][x]) and isSolid(levelStr[goalY_v + 1][x]):
				goalX_v = x
				break

	# horizontal start and goal
	startX_h = -1
	startY_h = None
	while startY_h == None:
		startX_h += 1
		if startX_h == 16:
			break
		for y in range(maxY-1, 0, -1):
			#print(y, startX)
			if not isSolid(levelStr[y][startX_h]) and isSolid(levelStr[y + 1][startX_h]):
				startY_h = y
				break

	goalX_h = maxX+1
	goalY_h = None
	while goalY_h == None:
		goalX_h -= 1
		if goalX_h < 0 or goalX_h < startX_h:
			break
		for y in range(maxY-1, 0, -1):
			if not isSolid(levelStr[y][goalX_h]) and isSolid(levelStr[y + 1][goalX_h]):
				goalY_h = y
				break

	#level[startY][startX] = '{'
	#level[goalY][goalX] = '}'
	
	#print('S',startX_v, startY_v, ' G', goalX_v, goalY_v)
	print('S',startX_h, startY_h, ' G', goalX_h, goalY_h)
	#print('\n'.join(levelStr))
	#paths = pathfinding.astar_shortest_path( (2,2,-1), lambda pos: pos[0] == maxX, getNeighbors, subOptimal, lambda pos: 0)#lambda pos: abs(maxX-pos[0]))
	if None in [startX_h, startY_h, goalX_h, goalY_v, startX_v, startY_v, goalX_v, goalY_v]:
		#print('start goal error')
		#print(startX, startY, goalX, goalY)
		return None, None, 0, 0

	sx, sy = startX_h, startY_h
	gx, gy = goalX_h, goalY_h
	path_h, node_h = pathfinding.astar_shortest_path( (sx,sy,-1), lambda pos: pos[0] == gx and pos[1] == gy, getNeighbors, lambda pos: 0)#lambda pos: abs(maxX-pos[0]))

	sx, sy = startX_v, startY_v
	gx, gy = goalX_v, goalY_v
	path_v, node_v = pathfinding.astar_shortest_path( (sx,sy,-1), lambda pos: pos[0] == gx and pos[1] == gy, getNeighborsWrapped, lambda pos: 0)#lambda pos: abs(maxX-pos[0]))
	
	dist_h, dist_v = node_h[1][0], node_v[1][0]
	if not path_h and not path_v:
		ph_out = None, None, dist_h, dist_v
	if not path_h:
		ph_out = None
	else:
		ph_out = [ (p[0],p[1]) for p in path_h]
	
	if not path_v:
		pv_out = None
	else:
		pv_out = [ (p[0],p[1]) for p in path_v]
	return ph_out, pv_out, dist_h, dist_v

def findPathsFull(affordances, levelStr, game):
	visited = set()
	solids, passables, climbables, hazards, jumps = affordances['solid'], affordances['passable'], affordances['climbable'], affordances['hazard'], affordances['jumps']
	isSolid = makeIsSolid(solids)
	isPassable = makeIsPassable(passables, hazards)
	isHazard = makeIsHazard(hazards)
	isClimbable = makeIsClimbable(climbables)

	#maxX = len(levelStr[0])-1
	maxX = len(levelStr[0]) - 1
	maxY = len(levelStr) - 1
	starts_h, goals_h = set(), set()
	starts_v, goals_v = set(), set()
	sg = set()
	if game == 'ki' or game == 'mm' or game == 'ng':
		startX = None
		startY = maxY-1
		#while startX == None and startY > 0:
		while startX == None and startY > maxY-1:
			startY -= 1
			for x in range(0, maxX):
				if not isSolid(levelStr[startY][x]) and isSolid(levelStr[startY + 1][x]):
					startX = x
					starts_v.add((startX,startY))
					if game == 'mm':  # MM can go both ways vertically
						goals_v.add((startX,startY))
					break
		
		startX = None
		startY = maxY-1
		#while startX == None and startY > 0:
		while startX == None and startY > maxY-1:
			startY -= 1
			for x in range(maxX-1, 0, -1):
				if not isSolid(levelStr[startY][x]) and isSolid(levelStr[startY + 1][x]):
					startX = x
					starts_v.add((startX,startY))
					if game == 'mm':
						goals_v.add((startX,startY))
					break

		goalX = None
		goalY = -1
		#while goalX == None and goalY < maxY-1:
		while goalX == None and goalY < 0:
			goalY += 1
			for x in range(0, maxX):
				if not isSolid(levelStr[goalY][x]) and isSolid(levelStr[goalY + 1][x]):
					goalX = x
					goals_v.add((goalX,goalY))
					if game == 'mm':
						starts_v.add((goalX,goalY))
					break

		goalX = None
		goalY = -1
		#while goalX == None and goalY < maxY-1:
		while goalX == None and goalY < 0:	
			goalY += 1
			for x in range(maxX-1, 0, -1):
				if not isSolid(levelStr[goalY][x]) and isSolid(levelStr[goalY + 1][x]):
					goalX = x
					goals_v.add((goalX,goalY))
					if game == 'mm':
						starts_v.add((goalX,goalY))
					break
	if game == 'smb' or game == 'mm' or game == 'cv' or game == 'ng':
		startX = -1
		startY_bottom, startY_top = None, None
		#while startY_bottom == None and startX < maxX:
		while startY_bottom == None and startX < 0:
			startX += 1
			for y in range(maxY-1, 0, -1):
				if not isSolid(levelStr[y][startX]) and isSolid(levelStr[y + 1][startX]):
					startY_bottom = y
					starts_h.add((startX,startY_bottom))
					break
		
		startX = -1
		#while startY_top == None and startX < maxX:
		while startY_bottom == None and startX < 0:
			startX += 1
			for y in range(0, maxY-1):
				if not isSolid(levelStr[y][startX]) and isSolid(levelStr[y + 1][startX]):
					startY_top = y
					starts_h.add((startX,startY_top))
					break
		
		goalX = maxX+1
		goalY_bottom, goalY_top = None, None
		#while goalY_bottom == None and goalX > 0:
		while goalY_bottom == None and goalX > maxX:
			goalX -= 1
			for y in range(maxY-1, 0, -1):
				if not isSolid(levelStr[y][goalX]) and isSolid(levelStr[y + 1][goalX]):
					goalY_bottom = y
					goals_h.add((goalX, goalY_bottom))
					break

		goalX = maxX+1
		#while goalY_top == None and goalX > 0:
		while goalY_bottom == None and goalX > maxX:
			goalX -= 1
			for y in range(0, maxY-1):
				if not isSolid(levelStr[y][goalX]) and isSolid(levelStr[y + 1][goalX]):
					goalY_top = y
					goals_h.add((goalX, goalY_top))
					break
		
	#print('starts_h: ',starts_h)
	#print('goals_h: ',goals_h)
	#print('starts_v: ',starts_v)
	#print('goals_v: ',goals_v)
	for start in starts_h:
		for goal in goals_h:
			sg.add((start,goal))
	
	for start in starts_v:
		for goal in goals_v:
			sg.add((start,goal))

		"""
		startX_p = -1
		startY_p = None
		while startY_p == None and startX_p < maxX:
			startX_p += 1
			for y in range(maxY-1, 0, -1):
				#print(y,startX_p)
				if levelStr[y][startX_p] == 'P':
					startY_p = y
					#starts.append((startX_p,startY_p))
					break
		goalX_p = maxX+1
		goalY_p = None
		while goalY_p == None and goalX_p > 0:
			goalX_p -= 1
			for y in range(maxY-1, 0, -1):
				if levelStr[y][goalX_p] == 'P':
					goalY_p = y
					#goals.append((goalX_p,goalY_p))
					break
		"""
		#if None not in [startX_p,startY_p,goalX_p,goalY_p]:
		#	sg.add(((startX_p,startY_p),(goalX_p,goalY_p)))

	#level[startY][startX] = '{'
	#level[goalY][goalX] = '}'
	#print('start: ', startX, startY)
	#print('goal: ', goalX, goalY)
	#if None in [startX, startY, goalX, goalY]:
	#	print('start goal error')
	#	print(startX, startY, goalX, goalY)
	#	return None, 0

	if len(sg) == 0:
	#if len(starts) == 0:
		return None, 0
	getNeighbors = makeGetNeighbors(jumps,levelStr,visited,isSolid,isPassable,isClimbable,isHazard,game)
	#paths = pathfinding.astar_shortest_path( (start_X, start_Y,-1), lambda pos: pos[0] == maxX, getNeighbors, subOptimal, lambda pos: 0)
	paths, best_dist, best_path = [], 0, None
	for start, goal in sg:
	#for start in starts:
		#print(start)
		#print('SG: ',start,goal)
		startX, startY = start
		goalX, goalY = goal
		path, node = pathfinding.astar_shortest_path( (startX, startY,-1), lambda pos: pos[0] == goalX and pos[1] == goalY, getNeighbors, lambda pos: 0)
		#path, node = pathfinding.astar_shortest_path( (startX, startY,-1), lambda pos: pos[1] == 0 if is_vertical else pos[0] == maxX, getNeighbors, lambda pos: 0)
		#print('Path: ',path)
		#print('Start: ', start, ' Goal: ', goal)
		if path:
			#print('Path node: ', node)
			first, last = path[0], path[-1]
			#print('First:',first,' Last:',last)
			if last[0] == goalX and last[1] == goalY:
				dist = 16
		else:
			#print('SG:',start,goal)
			#print('Dist: 0')
			final_x, final_y = node[1][0], node[1][1]
			dist_x = abs(final_x - startX)
			dist_y = abs(final_y - startY)
			dist = max(dist_x, dist_y)
			#print('Start:', start)
			#print('Goal: ', goal)
			#print('No path dist: ', dist, '\tBest: ', best_dist)
			#print('No path node: ', node)
		#dist = node[1][0]+1
		if dist > best_dist:
			best_dist = dist
			best_path = path
	#print(node)
	
	best_dist /= 16
	#print('Best dist in play: ', best_dist)
	if not best_path:
		#for i,(start,goal) in enumerate(sg):
		#	print(i,'Start: ', start, '\tGoal: ', goal)
		#print('\n')
		#print('no best path Starts:',starts)
		return None, best_dist
	return [ (p[0],p[1]) for p in best_path], best_dist