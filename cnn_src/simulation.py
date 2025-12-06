from src.occ_grid import Occ_grid
from src.st_grid import ST_grid
import numpy as np
from src.point import Point
from collections import deque
# simulates a dynamic environment

class Simulation():
    def __init__(self, h, w):
        self.h = h
        self.w = w

    # as (y, x) points, where index corresponds to ordinal:
    # 1 2 3
    # 8 0 4
    # 7 6 5
    offsets = [
        Point([0, 0]),
        Point([-1, -1]),
        Point([-1, 0]),
        Point([-1, 1]),
        Point([0, 1]),
        Point([1, 1]),
        Point([1, 0]),
        Point([1, -1]),
        Point([0, -1])
    ]
    rotation_probs = [0.01, 0.01, 0.01, 0.07, 0.8, 0.07, 0.01, 0.01, 0.01]
    #rotation_probs = [0.005, 0.005, 0.005, 0.005, 0.96, 0.005, 0.005, 0.005, 0.005]
    stationary_probs = [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    # 0th element is placeholder since we dont rotate for middle (zero position)
    rotations = [-1, 5, 6, 7, 0, 1, 2, 3, 4]

    class Obstacle():
        def __init__(self, y, x, h, w, is_dynamic=False):
            self.h = h
            self.w = w
            self.is_dynamic = is_dynamic
            # note y, x defines the top left corner of the obstacle
            self.pose = Point([y, x])
            self.prev_action = 0

    def action2offset(self, a):
        return self.offsets[a]

    # TODO: make prob dist of actions about prev action a
    def gen_dist(self, a):
        if a == 0:
            return self.stationary_probs.copy()
        else:
            probs = self.rotation_probs.copy()
            outer_probs = deque(probs[1:])
            outer_probs.rotate(self.rotations[a])
            probs[1:] = list(outer_probs)
            return probs
            
    def collides(self, o):
        # check 2 corners since rectangle
        start_x = o.pose[1]
        start_y = o.pose[0]
        end_x = o.pose[1] + o.w - 1
        end_y = o.pose[0] + o.h - 1
        if start_x < 0 or end_x >= self.w or start_y < 0 or end_y >= self.h:
            return True
        return False

    # does nothing to obstacle state if not dynamics
    def transition(self, o):
        if o.is_dynamic:
            probs = self.gen_dist(o.prev_action)
            action = np.random.choice(len(probs), p=probs)
            offset = self.action2offset(action)
            o.pose = o.pose + offset
            while self.collides(o):
                o.pose = o.pose - offset
                action = np.random.choice(len(probs), p=probs)
                offset = self.action2offset(action)
                o.pose = o.pose + offset
            o.prev_action = action

    # returns st_grid
    def simulate(self, time_steps=100, max_object_size=10, num_obstacles=10):
        st_grid = []
        obs_width_bounds = (1, max_object_size)
        obs_height_bounds = (1, max_object_size)
        obstacles = self.initialize_obstacles(num_obstacles, obs_height_bounds, obs_width_bounds)
        for t in range(time_steps):
            o_grid = Occ_grid(self.h, self.w)
            self.place_obstacles(o_grid, obstacles)
            self.update_obstacles(obstacles)
            st_grid.append(o_grid)
        return st_grid

    def update_obstacles(self, obstacles):
        for o in obstacles:
            self.transition(o)

    @staticmethod
    def place_obstacles(grid, obstacles):
        for o in obstacles:
            start_x = o.pose[1]
            start_y = o.pose[0]
            end_x = o.pose[1] + o.w
            end_y = o.pose[0] + o.h
            grid.grid[start_y:end_y, start_x:end_x] = 1
            
    def initialize_obstacles(self, n, hb, wb):
        obstacles = []
        for _ in range(n):
            w = np.random.randint(wb[0], wb[1]+1)
            h = np.random.randint(hb[0], hb[1]+1)
            x = np.random.randint(self.w)
            y = np.random.randint(self.h)
            is_dynamic = np.random.rand() < 0.9
            obs = Simulation.Obstacle(y, x, h, w, is_dynamic)
            while self.collides(obs):
                x = np.random.randint(self.w)
                y = np.random.randint(self.h)
                obs = Simulation.Obstacle(y, x, h, w, is_dynamic)
            obstacles.append(obs)
        return obstacles


