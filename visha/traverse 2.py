import numpy as np
import cv2
img = cv2.imread(arena_2.png)
Y ,X = shape.img[:2]
blocksize = 50
grid_Y = Y/blocksize
grid_X = X/blocksize
grid = np.zeros((grid_X,grid_Y),np.int32)
grid[:] = -1

neigh = []
prev_neigh = []
prev_neigh.append([cent[0],cent[1]])
grid_ind = 0

def start_grid(bot_loc):
    start = np.zeros((1,2),np.int32)
    start[0][0] = curr_loc[0]/blocksize
    start[0][1] = curr_loc[1]/blocksize
    return start

def neighbours(curr_loc):
    adj = [[1,1],[-1,-1]],[1,-1],[-1,1],[0,1],[1,0],[-1,0],[0,-1]]
    count = 1
    for i in adj:
        N_x = curr_loc[0]+i[0]
        N_y = curr_loc[1]+i[1]
        if (N_x <= grid_X and N_y <= grid_Y):
            if (grid[N_x][N_y] == -1):
                neigh.append([N_x,N_y])
                count = count + 1
    return neigh
while 1:
    for i in prev_neigh:
        neighbours(i)
        prev = prev_neigh.pop(0)
        grid_ind = grid[prev[0]][prev[1]] + 1
        l = len(neigh)
        for j in range(l):
            curr_block = neigh.pop(0):

            if (curr_block == goal):
                grid[curr_block[0]][curr_block[1] = grid_ind
                prev_neigh.append(curr_block)
                break

            grid[curr_block[0]][curr_bloc k[1]] = grid_ind
            prev_neigh.append(curr_block)

