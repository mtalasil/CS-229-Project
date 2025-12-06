import numpy as np
import heapq
from src.point import Point

# just a quick wrapper around list of occ_grids (bc making an nd grid class is cumbersome and not nessesary)
class ST_grid():
    def __init__(self, frames):
        self.grid = frames
        self.T = len(frames)
        self.h = frames[0].h
        self.w = frames[0].w

    def idx2point(self, idx):
        z = idx // (self.w * self.h)
        plane_idx = idx % (self.w * self.h)
        y = plane_idx // self.w
        x = plane_idx % self.w 
        return Point([z, y, x])
    
    def point2idx(self, p):
        return p[2] + p[1] * self.w + p[0] * self.w * self.h
    
    # idx in current occ grid
    def local_idx(self, idx):
        return idx % (self.w * self.h)
    
    # returns edge weight between 2 adjacent nodes
    def get_weight(self, n1, n2):
        p1 = self.idx2point(n1)
        p2 = self.idx2point(n2)
        dist = p1.dist(p2)
        if dist > np.sqrt(3) + 1e-6:
            raise ValueError("Points must be neighbours")
        return dist
    
    def is_obstacle(self, idx):
        p = self.idx2point(idx)
        loc_idx = self.local_idx(idx)
        if self.grid[p[0]].is_obstacle(loc_idx):
            return True
        return False
    
    def is_unsafe(self, idx):
        p = self.idx2point(idx)
        loc_idx = self.local_idx(idx)
        prev_grid = self.grid[p[0]]
        unsafe = prev_grid.near_obstacle(loc_idx)
        return unsafe

    # returns only future neighbours (including curr xy in t+1)
    def get_neighbours(self, idx):
        # just call get neighbours on occ_grid one timstep ahead at same xy position
        idx_t1 = idx + self.w * self.h # idx of node in same xy position but one timestep ahead
        loc_idx_t1 = self.local_idx(idx_t1)
        if idx_t1 >= self.w * self.h * self.T:
            return [] #if at last timestep, it has no neighbours so return empty list, bc neighbours are always in future
        p_t1 = self.idx2point(idx_t1)
        nbrs = [idx_t1]

        timestep = p_t1[0]
        base_idx_tl = (self.w * self.h) * p_t1[0]
        occ_grid_t1 = self.grid[timestep]
        for n in occ_grid_t1.get_neighbours(loc_idx_t1):
            nbrs.append(n + base_idx_tl)
        return nbrs
        # returns list of neighbours