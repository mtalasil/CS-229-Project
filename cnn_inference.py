import numpy as np
import torch.nn as nn
import torch
from src import simulation
from src import st_grid
from src import path_finding
from src.occ_grid import Occ_grid
from src.point import Point
from generate_data import viz_sim
from generate_data import viz_sim_2p
from src.valueCNN import ValueCNN
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from generate_data_val_fn import plot_field

action_costs = [
    #(Point([0, 0]), 0),
    (Point([-1, -1]), np.sqrt(2)),
    (Point([-1, 0]), 1),
    (Point([-1, 1]), np.sqrt(2)),
    (Point([0, 1]), 1),
    (Point([1, 1]), np.sqrt(2)),
    (Point([1, 0]), 1),
    (Point([1, -1]), np.sqrt(2)),
    (Point([0, -1]), 1)
]

# technically not correct bc free space in curr timestep could be taken in next timestep
# TODO: fix this bug described above later
def valid_pose(pose, occ_grid):
    if pose[0] < 0 or pose[0] >= occ_grid.h or pose[1] < 0 or pose[1] >= occ_grid.w:
        return False
    idx = occ_grid.point2idx(pose)
    if occ_grid.near_obstacle(idx):
        return False
    return True

def plot_field(J, h, w, goal_p, curr_p, three_d=False):
    if three_d:
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(x, y)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            goal_p[1], goal_p[0], J[goal_p[0], goal_p[1]],     # x, y, and the value z = V(y,x)
            color='red',
            s=30,
            marker='o',
        )
        ax.scatter(
            curr_p[1], curr_p[0], J[curr_p[0], curr_p[1]],     # x, y, and the value z = V(y,x)
            color='green',
            s=30,
            marker='o',
        )
        ax.plot_surface(X, Y, J, rstride=1, cstride=1, linewidth=0)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('J(x,y)')
        plt.show()
    else:
        plt.imshow(J, cmap='gray')
        plt.colorbar()
        plt.scatter(goal_p[1], goal_p[0], c='red', s=50)
        plt.scatter(curr_p[1], curr_p[0], c='green', s=50)
        plt.show()

def heuristic(p1, p2):
    return p1.dist(p2)

def run_inference(stgrid, model, start, goal, device):
    occ_grids = stgrid.grid
    time_steps = len(occ_grids)
    h = stgrid.h
    w = stgrid.w
    goal_p = occ_grids[0].idx2point(goal)
    start_p = occ_grids[0].idx2point(start)
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    goal_dx = (goal_p[1] - xs) / w
    goal_dy = (goal_p[0] - ys) / h
    curr_pose = start_p
    path = [start_p]
    path_found = False
    norm_const = 1/(time_steps * np.sqrt(2))
    heuristic_bonus_base = 1.0
    heuristic_bonus = heuristic_bonus_base
    #### start timer!
    total_path_cost = 0
    start_t = time.perf_counter()
    for t in range(time_steps):
        if t == 0:
            curr_stack = np.stack([occ_grids[t].grid, occ_grids[t].grid, occ_grids[t].grid, goal_dy, goal_dx], axis=0)
        if t == 1:
            curr_stack = np.stack([occ_grids[t-1].grid, occ_grids[t-1].grid, occ_grids[t].grid, goal_dy, goal_dx], axis=0)
        else:
            curr_stack = np.stack([occ_grids[t-2].grid, occ_grids[t-1].grid, occ_grids[t].grid, goal_dy, goal_dx], axis=0)
        curr_state = torch.from_numpy(curr_stack).float().unsqueeze(0).to(device)
        with torch.no_grad():
            J_raw = model(curr_state)
            J = J_raw.squeeze().cpu().numpy()
            #plot_field(J, h, w, goal_p, curr_pose, three_d=True)
            min_cost = np.inf
            best_next_pose = None
            best_action_cost = None
            for a, cost in action_costs:
                pot_next_pose = curr_pose + a
                pot_np_idx = occ_grids[t].point2idx(pot_next_pose)
                if pot_next_pose[0] < 0 or pot_next_pose[0] >= h or pot_next_pose[1] < 0 or pot_next_pose[1] >= w or occ_grids[t].near_obstacle(pot_np_idx):
                    continue
                #print(f"ctg: {J[pot_next_pose[0], pot_next_pose[1]]}")
                #print(f"heuristic: {norm_const * heuristic_bonus * heuristic(curr_pose, goal_p)}")
                next_cost = norm_const * cost + J[pot_next_pose[0], pot_next_pose[1]] + norm_const * heuristic_bonus * heuristic(pot_next_pose, goal_p) # + immediate cost if you want but you MUST normalize by some constant or it will dominate (ex time_steps * sqrt(2) -> which is longest possible path given time horizon)
                if next_cost < min_cost:
                    min_cost = next_cost
                    best_next_pose = pot_next_pose
                    best_action_cost = cost
            if best_next_pose is None:
                #print("No path from here!")
                break
            curr_pose = best_next_pose
            total_path_cost += best_action_cost
            # if len(path) > 3 and path[-2] == curr_pose and path[-4] == curr_pose: # if oscillating (bc in some local min), then increase heuristic temporarily and reset once out of local min
            #     heuristic_bonus = max(heuristic_bonus*2, 20*heuristic_bonus)
            # else:
            #     heuristic_bonus = heuristic_bonus_base
            if occ_grids[t].is_obstacle(occ_grids[t].point2idx(curr_pose)):
                print("inside obstacle!!")
            path.append(curr_pose)
            if curr_pose == goal_p:
                path_found = True
                break
            # get values of 9 squares (including curr square?) around you and take action with lowest value
            # then add new pose to path
            # then check if new pose is goal pose - if so set path_found = True and break from loop
    end_t = time.perf_counter()
    #### end timer!
    #if path_found:
        #print("path found!")
    #else:
        #print("no path found :(")
    return (path, total_path_cost, end_t-start_t, path_found)

def run_astar_inference(stgrid, start, goal):
    occ_grids = stgrid.grid
    time_steps = len(occ_grids)
    h = stgrid.h
    w = stgrid.w
    goal_p = occ_grids[0].idx2point(goal)
    start_p = occ_grids[0].idx2point(start)
    path = [start_p]
    found_path = False
    start_t = time.perf_counter()
    curr_pose = start_p
    curr_pose_idx = start
    total_path_cost = 0
    for t in range(time_steps):
        #no_path = True
        # if not occ_grids[t].is_obstacle(goal):
        #     curr_path, valid = path_finding.a_star(occ_grids[t], curr_pose_idx, goal)
        #     if valid:
        #         no_path = False
        #         path.append(curr_path[1])
        #         curr_pose = curr_path[1]
        #         curr_pose_idx = occ_grids[t].point2idx(curr_pose)
        # if no_path:
        best_action_cost = None
        curr_path, valid = path_finding.a_star(occ_grids[t], curr_pose_idx, goal)
        if valid:
            path.append(curr_path[1])
            curr_pose = curr_path[1]
            curr_pose_idx = occ_grids[t].point2idx(curr_pose)
            best_action_cost = action_costs[path[-1] - path[-2]]
        else:
            best_next_pose = None
            min_dist = np.inf
            for a, cost in action_costs:
                pot_next_pose = curr_pose + a
                pot_next_pose_idx = occ_grids[t].point2idx(pot_next_pose)
                if pot_next_pose[0] < 0 or pot_next_pose[0] >= h or pot_next_pose[1] < 0 or pot_next_pose[1] >= w or occ_grids[t].near_obstacle(pot_next_pose_idx):
                    continue
                if pot_next_pose.dist(goal_p) < min_dist:
                    min_dist = pot_next_pose.dist(goal_p)
                    best_next_pose = pot_next_pose
                    best_action_cost = cost
            if best_next_pose is None:
                #print("no action :(")
                break
            path.append(best_next_pose)
            curr_pose = best_next_pose
            curr_pose_idx = occ_grids[t].point2idx(best_next_pose)
        total_path_cost += best_action_cost
        if curr_pose == goal_p:
            found_path = True
            #print("Found path!")
            break
    end_t = time.perf_counter()
    return (path, total_path_cost, end_t-start_t, found_path)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_sim(sim, model, device, time_steps, plot=False):
    h = sim.h
    w = sim.w
    max_object_size = np.random.randint(1, h//4)
    num_obstacles = np.random.randint(0, h//2)
    occ_grids = sim.simulate(time_steps=time_steps, max_object_size=max_object_size, num_obstacles=num_obstacles)
    stgrid = st_grid.ST_grid(occ_grids)
    num_spatial_nodes = h*w
    start = np.random.randint(0, num_spatial_nodes)
    goal = np.random.randint(0, num_spatial_nodes)
    while start == goal or occ_grids[0].near_obstacle(start) or occ_grids[0].is_obstacle(goal):
        start = np.random.randint(0, num_spatial_nodes)
        goal = np.random.randint(0, num_spatial_nodes)
    goal_p = occ_grids[0].idx2point(goal)
    start_p = occ_grids[0].idx2point(start)
    cnn_stats = run_inference(stgrid, model, start, goal, device)
    astar_stats = run_astar_inference(stgrid, start, goal)
    (cnn_path, cnn_total_path_cost, cnn_time, cnn_found_path) = cnn_stats
    (astar_path, astar_total_path_cost, astar_time, astar_found_path) = astar_stats

    if plot:
        print("--------------------")
        if cnn_found_path:
            print("CNN PATH FOUND!")
        else:
            print("NO CNN PATH FOUND!")

        if astar_found_path:
            print("ASTAR PATH FOUND!")
        else:
            print("NO ASTAR PATH FOUND!")

        print("CNN")
        print(f"time: {cnn_time}")
        print(f"path cost: {cnn_total_path_cost}")
        print("A*")
        print(f"time: {astar_time}")
        print(f"path cost: {astar_total_path_cost}")
        viz_sim_2p(stgrid, start_p, goal_p, cnn_path, astar_path)
    #viz_sim(stgrid, start_p, goal_p, astar_path)
    return cnn_stats, astar_stats

def consolidate_stats(stats):
    cnn_shorter_path_count = 0 # number of time cnn has lower cost
    cnn_shorter_time_count = 0 # number of times cnn has lower time
    cnn_exclusive_no_path_count = 0 # number of times when cnn doesnt find a path but A* does
    astar_exclusive_no_path_count = 0 # number of times when A* doesnt find a path but CNN does
    cnn_to_astar_cost_ratio_sum = 0 # sum of cost ratios
    cnn_to_astar_time_ratio_sum = 0 # sum of time ratios
    both_path_found_count = 0 # where both find a path (otherwise cant compare stats)
    path_found_count = 0 # where at least one finds a path - dont care about cases where both find no path bc likely an unsolvable case so dont consider for stats (bc impossible to find path anyway)
    # ALL STATS ARE FOR WHEN BOTH METHODS FIND A PATH (except no_path_count stats)
    for row in stats:
        (_, cnn_path_cost, cnn_time, cnn_found_path) = row[0]
        (_, astar_path_cost, astar_time, astar_found_path) = row[1]
        if cnn_found_path and astar_found_path:
            both_path_found_count += 1
            if cnn_path_cost <= astar_path_cost:
                cnn_shorter_path_count += 1
            if cnn_time <= astar_time:
                cnn_shorter_time_count += 1
            cnn_to_astar_cost_ratio_sum += cnn_path_cost / astar_path_cost
            cnn_to_astar_time_ratio_sum += cnn_time / astar_time
        if cnn_found_path or astar_found_path:
            path_found_count += 1
        if not cnn_found_path and astar_found_path:
            cnn_exclusive_no_path_count += 1
        if not astar_found_path and cnn_found_path:
            astar_exclusive_no_path_count += 1
        
    avg_cnn_to_astar_cost_ratio = cnn_to_astar_cost_ratio_sum / both_path_found_count
    avg_cnn_to_astar_time_ratio = cnn_to_astar_time_ratio_sum / both_path_found_count
    cnn_shorter_path_pct = cnn_shorter_path_count / both_path_found_count
    cnn_shorter_time_count_pct = cnn_shorter_time_count / both_path_found_count
    if path_found_count > 0:
        cnn_exclusive_no_path_pct = cnn_exclusive_no_path_count / path_found_count
        astar_exclusive_no_path_pct = astar_exclusive_no_path_count / path_found_count
    else:
        cnn_exclusive_no_path_pct = -1
        astar_exclusive_no_path_pct = -1

    print(f"avg_cnn_to_astar_cost_ratio: {avg_cnn_to_astar_cost_ratio}")
    print(f"avg_cnn_to_astar_time_ratio: {avg_cnn_to_astar_time_ratio}")
    print(f"cnn_shorter_path_pct: {cnn_shorter_path_pct}")
    print(f"cnn_shorter_time_count_pct: {cnn_shorter_time_count_pct}")
    print(f"cnn_exclusive_no_path_pct: {cnn_exclusive_no_path_pct}")
    print(f"astar_exclusive_no_path_pct: {astar_exclusive_no_path_pct}")

def main():
    np.random.seed(7)
    # test seed is 6
    device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    h = 64
    w = 64
    time_steps = 150
    sim = simulation.Simulation(h, w)

    IN_CHANNELS = 5
    base_ch = 8
    model = ValueCNN(IN_CHANNELS, base_ch)
    model.load_state_dict(torch.load(f"unet_base{base_ch}_cnn_model_epoch_6.pt", map_location=device))
    model.to(device)
    model.eval()
    print(count_parameters(model))

    num_sims = 0
    plot = True
    stats = []
    for s in range(num_sims):
        print(f"on sim {s}")
        cnn_stats, astar_stats = run_sim(sim, model, device, time_steps, plot=plot)
        stats.append((cnn_stats, astar_stats))
    #consolidate_stats(stats)
    
    #TODO: make unet smaller (by 4x?? or even more!?) and see if it can still have decent accuracy and be fast on cpu!
    # Also technically for A*, if goal is covered dont do A* (wastes time)
    # worst case, with new A*, see if NN can beat new A* on gpu (if cpu really bad even for small unet)

if __name__ == "__main__":
    main()