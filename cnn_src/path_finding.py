import numpy as np
import heapq 
from src.point import Point

def stidx2spoint(idx, stgrid):
    p = stgrid.idx2point(idx)
    return Point([p[1], p[2]])

def backtrace_st(goal, start, prev_nodes, stgrid):
    path = []
    curr = goal
    path.append(stidx2spoint(curr, stgrid))
    while curr != start:
        curr = prev_nodes[curr]
        if curr == -1:
            return []
        path.append(stidx2spoint(curr, stgrid))
    return path[::-1]

def h_cost_st(curr, goal, stgrid):
    p_curr = stgrid.idx2point(stgrid.local_idx(curr))
    p_goal = stgrid.idx2point(goal)
    return p_curr.dist(p_goal)

def backtrace(goal, start, prev_nodes, occ_grid):
    path = []
    curr = goal
    path.append(occ_grid.idx2point(curr))
    while curr != start:
        curr = prev_nodes[curr]
        if curr == -1:
            return []
        path.append(occ_grid.idx2point(curr))
    return path[::-1]

def h_cost(curr, goal, occ_grid):
    p_curr = occ_grid.idx2point(curr)
    p_goal = occ_grid.idx2point(goal)
    return p_curr.dist(p_goal)

#start and goal are local indices (have no time component)
def st_a_star(stgrid, start, goal):
    T = stgrid.T
    h = stgrid.h
    w = stgrid.w
    shortest_distances = np.full(h*w*T, -1.0)
    visited = np.zeros(h*w*T, dtype=bool)
    prev_nodes = np.full(h*w*T, -1, dtype=int)
    updated_f_costs = np.full(h*w*T, -1.0)
    pq = []
    heapq.heappush(pq, (0, start))
    shortest_distances[start] = 0
    updated_f_costs[start] = 0
    goal_st_idx = -1
    while pq:
        curr = heapq.heappop(pq)
        curr_idx = curr[1]
        if abs(updated_f_costs[curr_idx] - curr[0]) > 1e-6:
            continue
        visited[curr_idx] = True
        if stgrid.local_idx(curr_idx) == goal:
            goal_st_idx = curr_idx
            break
        nbrs = stgrid.get_neighbours(curr_idx)
        for n in nbrs:
            if not visited[n] and not stgrid.is_obstacle(n):
                new_g_cost = shortest_distances[curr_idx] + stgrid.get_weight(curr_idx, n)
                if shortest_distances[n] == -1 or new_g_cost < shortest_distances[n]:
                    shortest_distances[n] = new_g_cost
                    prev_nodes[n] = curr_idx
                    f_cost = shortest_distances[n] + h_cost_st(n, goal, stgrid)
                    heapq.heappush(pq, (f_cost, n))
                    updated_f_costs[n] = f_cost
    # done search
    if goal_st_idx == -1:
        print("no path")
        return ([], False)
    path = backtrace_st(goal_st_idx, start, prev_nodes, stgrid)
    return (path, True)

def a_star(occ_grid, start, goal):
    h = occ_grid.h
    w = occ_grid.w
    shortest_distances = np.full(h*w, -1.0)
    visited = np.zeros(h*w, dtype=bool)
    prev_nodes = np.full(h*w, -1, dtype=int)
    updated_f_costs = np.full(h*w, -1.0)
    pq = []
    heapq.heappush(pq, (0, start))
    shortest_distances[start] = 0
    updated_f_costs[start] = 0
    path_found = False
    while pq:
        curr = heapq.heappop(pq)
        curr_idx = curr[1]
        if abs(updated_f_costs[curr_idx] - curr[0]) > 1e-6:
            continue
        visited[curr_idx] = True
        if curr_idx == goal:
            path_found = True
            break
        nbrs = occ_grid.get_neighbours(curr_idx)
        for n in nbrs:
            if not visited[n] and not occ_grid.is_obstacle(n):
                new_g_cost = shortest_distances[curr_idx] + occ_grid.get_weight(curr_idx, n)
                if shortest_distances[n] == -1 or new_g_cost < shortest_distances[n]:
                    shortest_distances[n] = new_g_cost
                    prev_nodes[n] = curr_idx
                    f_cost = shortest_distances[n] + h_cost(n, goal, occ_grid)
                    heapq.heappush(pq, (f_cost, n))
                    updated_f_costs[n] = f_cost
    # done search
    if not path_found == -1:
        #print("no path")
        return ([], False)
    path = backtrace(goal, start, prev_nodes, occ_grid)
    return (path, True)



    