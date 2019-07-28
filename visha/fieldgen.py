import numpy as np
import cv2
img = cv2.imread(arena_2.png)
Y ,X = shape.img[:2]
blocksize = 50
grid_Y = Y/blocksize
grid_X = X/blocksize
grid = np.zeros((grid_X,grid_Y),np.int32)
grid[:] = -1

neigh = np.zeros((8,2),np.int32
sec_neigh = np.zeros((8,2),np.int32)

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
        neigh[count][0] = N_x
        neigh[count][1] = N_y
        count = count + 1
    return neigh

def sec_neighbours(loc):
    adj = [[1, 1], [-1, -1]], [1, -1], [-1, 1], [0, 1], [1, 0], [-1, 0], [0, -1]]
    count = 1
    for i in adj:
    N_x = curr_loc[0] + i[0]
    N_y = curr_loc[1] + i[1]
    sec_neigh[count][0] = N_x
    sec_neigh[count][1] = N_y
    count = count + 1
    return sec_neigh

def obstacle(obst):
    for j in obst:
        grid[j[0]/blocksize][j[1]/blocksize] = -2
grid_ind = 2
curr_soc = neighbours(start_grid(marker_loc()))
for curr_block in curr_soc:
    if (curr_block == goal):
        grid[curr_block[0]][curr_block[1] == grid_ind
        break
for curr_block in curr_soc:
    if (grid[curr_block[0]][curr_block[1]] == -1):
        grid[curr_block[0]][curr_block[1]] == 1
while 1:
    next_soc = neighbours()
    for i in neigh:
        curr_soc = sec_neighbbours(i)
        for curr_block in curr_soc:
            if (curr_block == goal):
                grid[curr_block[0]][curr_block[1] == grid_ind
                break
        for curr_block in curr_soc:
             if (grid[curr_block[0]][curr_block[1]] == -1):
                grid[curr_block[0]][curr_block[1]] == grid_ind

