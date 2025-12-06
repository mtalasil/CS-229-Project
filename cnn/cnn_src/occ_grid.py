import numpy as np
import heapq
from src.point import Point

# occ grid class
# we identify nodes by IDX and implicitly calc y and x from idx using h and w
class Occ_grid():
    offsets = [
        Point([-1, -1]),
        Point([-1, 0]),
        Point([-1, 1]),
        Point([0, 1]),
        Point([1, 1]),
        Point([1, 0]),
        Point([1, -1]),
        Point([0, -1])
    ]

    def __init__(self, h, w):
        self.h = h
        self.w = w
        self.grid = np.zeros((h, w))

    def idx2point(self, idx):
        y = idx // self.w
        x = idx % self.w
        return Point([y, x])
    
    def point2idx(self, p):
        return p[1] + p[0]*self.w
    
    def is_obstacle(self, idx):
        p = self.idx2point(idx)
        if self.grid[p[0], p[1]] == 1:
            return True
        return False
    
    def near_obstacle(self, idx):
        nbrs = self.get_neighbours(idx)
        if self.is_obstacle(idx):
            return True
        for n in nbrs:
            if self.is_obstacle(n):
                return True
        return False
    
    # returns edge weight between 2 adjacent nodes  -
    def get_weight(self, n1, n2):
        p1 = self.idx2point(n1)
        p2 = self.idx2point(n2)
        dist = p1.dist(p2)
        if dist > np.sqrt(2) + 1e-6:
            raise ValueError("Points must be neighbours")
        return dist
        # diff = p1-p2
        # dy = abs(diff[0])
        # dx = abs(diff[1])
        # if dy > 1.1 or dx > 1.1:
        #     raise ValueError("Points must be neighbours")
        # elif (dy + dx) - 1 < 1e-6:
        #     return 1.0
        # return np.sqrt(2.0)

    def get_neighbours(self, idx):
        p = self.idx2point(idx)
        nbrs = []
        for o in self.offsets:
            nbr = p + o
            if nbr[0] >= 0 and nbr[0] < self.h and nbr[1] >=0 and nbr[1] < self.w:
                nbrs.append(self.point2idx(nbr))
        return nbrs