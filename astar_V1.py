import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class AStar(object):

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo         # state space lower bound 
        self.statespace_hi = statespace_hi         # state space upper bound 
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space 
        self.x_offset = x_init                     
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # visited states
        self.open_set = set()      # states not visited

        self.est_cost_through = {}  # estimated cost from start to goal passing through state 
        self.cost_to_arrive = {}    # cost-to-arrive at state from start 
        self.came_from = {}         # each state's parent for path reconstruction

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # final path as a list of states

    def is_free(self, x): # collision checker
        return self.occupancy.is_free(x)

    def distance(self, x1, x2): # L2 norm euclidian distance between two states
        return np.linalg.norm(np.array(x2) - np.array(x1))

    def snap_to_grid(self, x): # closest point to x on discrete grid
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x): # retrieves free neighbors of state x
        neighbors = []
        x1, x2 = x
        potential_neighbors = [
        self.snap_to_grid((x1, x2 + self.resolution)),
        self.snap_to_grid((x1, x2 - self.resolution)),
        self.snap_to_grid((x1 + self.resolution, x2)),
        self.snap_to_grid((x1 - self.resolution, x2)),
        self.snap_to_grid((x1 + self.resolution, x2 + self.resolution)),
        self.snap_to_grid((x1 + self.resolution, x2 - self.resolution)),
        self.snap_to_grid((x1 - self.resolution, x2 + self.resolution)),
        self.snap_to_grid((x1 - self.resolution, x2 - self.resolution))
    ]
        for neigh in potential_neighbors:
            if (self.is_free(neigh) and
                self.statespace_lo[0] <= neigh[0] <= self.statespace_hi[0] and
                self.statespace_lo[1] <= neigh[1] <= self.statespace_hi[1]):
                neighbors.append(neigh)

        return neighbors

    def find_best_est_cost_through(self): # best estimated cost from start to goal
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self): # reconstructing path from initial to goal
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def plot_path(self, fig_num=0, show_init_label=True, save_path=None):
        if not self.path:
            return

        ax = self.occupancy.plot(fig_num)

        for (x, y) in self.path:
            rect = patches.Rectangle((x, y), 1, 1, facecolor='green', edgecolor='none', alpha=0.6)
            ax.add_patch(rect)

        start_rect = patches.Rectangle(self.x_init, 1, 1, facecolor='blue', edgecolor='black')
        goal_rect = patches.Rectangle(self.x_goal, 1, 1, facecolor='red', edgecolor='black')
        ax.add_patch(start_rect)
        ax.add_patch(goal_rect)

        # Optional labels
        if show_init_label:
            ax.text(self.x_init[0] + 0.1, self.x_init[1] + 0.1, "S", fontsize=8, color="white")
            ax.text(self.x_goal[0] + 0.1, self.x_goal[1] + 0.1, "G", fontsize=8, color="white")

        plt.axis([0, self.occupancy.width, 0, self.occupancy.height])
        ax.set_aspect('equal')
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=200)
            plt.close()


    def solve(self): # solves planning problem using A*
        self.open_set = {self.x_init}
        self.closed_set = set()
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        while len(self.open_set) > 0:
            x_current = self.find_best_est_cost_through()
            if x_current == self.x_goal:
                self.path = self.reconstruct_path()
                return True
            self.open_set.remove(x_current)
            self.closed_set.add(x_current)
            neighbors = self.get_neighbors(x_current)
            for x_neigh in neighbors:
                if x_neigh in self.closed_set: continue
                tentative_cost_to_arrive = self.cost_to_arrive[x_current] + self.distance(x_current,x_neigh)
                if x_neigh not in self.open_set:
                    self.open_set.add(x_neigh)
                elif tentative_cost_to_arrive > self.cost_to_arrive[x_neigh]:
                    continue
                self.came_from[x_neigh] = x_current
                self.cost_to_arrive[x_neigh] = tentative_cost_to_arrive
                self.est_cost_through[x_neigh] = tentative_cost_to_arrive + self.distance(x_neigh,self.x_goal)
        return False

class DetOccupancyGrid2D(object): # 2D state spage grid object with rectangular obstacles

    def __init__(self, width, height, obstacles):
        self.width = width # grid size width
        self.height = height # grid size height
        self.obstacles = obstacles # obstacles grid; 1.0 if taken

    def is_free(self, x): # collision checker
        xi, yi = int(x[0]), int(x[1])  # x=col, y=row
        if 0 <= xi < self.width and 0 <= yi < self.height:
            return self.obstacles[yi, xi] == 0  # <-- swap indices


    def plot(self, fig_num=0):  # plots grid and obstacles
        fig = plt.figure(fig_num, figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.imshow(self.obstacles.T, origin='lower', cmap='gray_r', extent=[0, self.width, 0, self.height])

        # Add gridlines for each cell
        ax.set_xticks(np.arange(0, self.width + 1, 1))
        ax.set_yticks(np.arange(0, self.height + 1, 1))
        ax.grid(which="both", color="lightgray", linewidth=0.5)

        # Make each square cell visible
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')

        return ax
