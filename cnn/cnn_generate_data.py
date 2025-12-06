import numpy as np
import heapq
from src import occ_grid
from src import simulation
from src import path_finding
from src import st_grid
import random
import matplotlib.pyplot as plt
import cv2
import time
import copy
from src.point import Point


import torch.nn as nn
import torch
from src.policyCNN import PolicyCNN

import cv2
import copy

action_map = {
    Point([0, 0]): 0,
    Point([-1, -1]): 1,
    Point([-1, 0]): 2,
    Point([-1, 1]): 3,
    Point([0, 1]): 4,
    Point([1, 1]): 5,
    Point([1, 0]): 6,
    Point([1, -1]): 7,
    Point([0, -1]): 8
}
def viz_sim_2p(stgrid, start_p, goal_p, path1, path2, scale=8):
    frames_cpy = copy.deepcopy(stgrid.grid)

    start_y, start_x = start_p
    goal_y,  goal_x  = goal_p
    i = 0
    for frame in frames_cpy:
        if i >= max(len(path1), len(path2)):
            break
        f = frame.grid  # shape: (H, W) or (H, W, 3)

        # --- Convert to BGR if grayscale -------------------------
        if f.ndim == 2:
            # grayscale -> BGR
            f_disp = cv2.cvtColor(f.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
        elif f.ndim == 3 and f.shape[2] == 3:
            # RGB -> BGR
            f_disp = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError("Unsupported grid format")

        # --- Draw start (blue) and goal (red) --------------------
        # Ensure coordinates are inside bounds

        for j in range(i):
            p1_idx = min(j, len(path1)-1)
            p2_idx = min(j, len(path2)-1)
            f_disp[path1[p1_idx][0], path1[p1_idx][1]] = (0, 255, 0)
            f_disp[path2[p2_idx][0], path2[p2_idx][1]] = (255, 0, 255)

        H, W = f_disp.shape[:2]
        if 0 <= start_y < H and 0 <= start_x < W:
            f_disp[start_y, start_x] = (255, 0, 0)   # Blue for start

        if 0 <= goal_y < H and 0 <= goal_x < W:
            f_disp[goal_y, goal_x] = (0, 0, 255)     # Red for goal

        # --- Upscale for visualization --------------------------
        f_disp = cv2.resize(
            f_disp,
            (W * scale, H * scale),
            interpolation=cv2.INTER_NEAREST
        )
        i += 1
        cv2.imshow("Video", f_disp)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def viz_sim(stgrid, start_p, goal_p, path, scale=8):
    frames_cpy = copy.deepcopy(stgrid.grid)

    start_y, start_x = start_p
    goal_y,  goal_x  = goal_p
    i = 0
    for frame in frames_cpy:
        if i >= len(path):
            break
        f = frame.grid  # shape: (H, W) or (H, W, 3)

        # --- Convert to BGR if grayscale -------------------------
        if f.ndim == 2:
            # grayscale -> BGR
            f_disp = cv2.cvtColor(f.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
        elif f.ndim == 3 and f.shape[2] == 3:
            # RGB -> BGR
            f_disp = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError("Unsupported grid format")

        # --- Draw start (blue) and goal (red) --------------------
        # Ensure coordinates are inside bounds
        H, W = f_disp.shape[:2]
        if 0 <= start_y < H and 0 <= start_x < W:
            f_disp[start_y, start_x] = (255, 0, 0)   # Blue for start

        if 0 <= goal_y < H and 0 <= goal_x < W:
            f_disp[goal_y, goal_x] = (0, 0, 255)     # Red for goal
        #for j in range(i):
        j = i
        f_disp[path[j][0], path[j][1]] = (0, 255, 0)

        # --- Upscale for visualization --------------------------
        f_disp = cv2.resize(
            f_disp,
            (W * scale, H * scale),
            interpolation=cv2.INTER_NEAREST
        )
        i += 1
        cv2.imshow("Video", f_disp)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def collect_sim_data(simulation, num_sims, time_steps, h, w):
    stacks = []
    num_spatial_nodes = h * w
    for sim in range(num_sims):
        print(f"On sim {sim}...")
        max_object_size = np.random.randint(1, h//4)
        num_obstacles = np.random.randint(0, h//3)
        occ_grids = simulation.simulate(time_steps=time_steps, max_object_size=max_object_size, num_obstacles=num_obstacles)
        #stgrid = st_grid.ST_grid(occ_grids)
        start = np.random.randint(0, num_spatial_nodes)
        # MAKE SURE START IS 2 AWAY FROM OBSTACLES TO START
        goal = np.random.randint(0, num_spatial_nodes)
        goal_p = occ_grids[0].idx2point(goal)
        while start == goal or occ_grids[0].near_obstacle(start) or occ_grids[0].is_obstacle(goal):
            start = np.random.randint(0, num_spatial_nodes)
            goal = np.random.randint(0, num_spatial_nodes)
        #path, valid = path_finding.st_a_star(stgrid, start, goal)
        # if valid:
        #     print("Valid path")
        #     start_p = occ_grids[0].idx2point(start)
        #     goal_p = occ_grids[0].idx2point(goal)
        #     viz_sim(stgrid, start_p, goal_p, path)
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        goal_dx = (goal_p[0] - xs) / w
        goal_dy = (goal_p[1] - ys) / h
        curr_pose_idx = start
        for occ_grid in occ_grids:
            if curr_pose_idx == goal:
                break
            static_path, valid = path_finding.a_star(occ_grid, curr_pose_idx, goal)
            if valid: 
                action_offset = static_path[1] - static_path[0]
                action = action_map[action_offset]
                agent_mask = np.zeros((h, w))
                agent_mask[static_path[0][0], static_path[0][1]] = 1
                curr_stack = np.stack([occ_grid.grid, agent_mask, goal_dx, goal_dy], axis=0)
                stacks.append((curr_stack, action))
                curr_pose_idx = occ_grid.point2idx(static_path[1])
    return stacks

def generate_data_same_sims(sim, time_steps, height, width, num_sims=100):
    # generate training data
    print("Generating data...")
    data = collect_sim_data(sim, num_sims, time_steps, height, width)
    random.shuffle(data)
    inputs, labels = zip(*data)
    inputs_arr = np.stack(inputs, axis=0)
    labels_arr = np.array(labels)
    split_idx = int(0.8 * len(data))
    np.savez_compressed(f"train_dataset_{split_idx}_exs.npz", inputs=inputs_arr[0:split_idx], labels=labels_arr[0:split_idx])
    np.savez_compressed(f"val_dataset_{len(data)-split_idx}_exs.npz", inputs=inputs_arr[split_idx:], labels=labels_arr[split_idx:])

    # # generate val data
    # print("Generating val data...")
    # val_data = collect_sim_data(sim, num_sims_val, time_steps, height, width)
    # random.shuffle(val_data)
    # inputs_val, labels_val = zip(*val_data)
    # inputs_val_arr = np.stack(inputs_val, axis=0)
    # labels_val_arr = np.array(labels_val)

# def generate_data_diff_sims(sim, time_steps, height, width, num_sims=100):
#     # generate training data
#     print("Generating train data...")
#     val_data = collect_sim_data(sim, num_sims_val, time_steps, height, width)
#     random.shuffle(val_data)
#     inputs_val, labels_val = zip(*val_data)
#     inputs_val_arr = np.stack(inputs_val, axis=0)
#     labels_val_arr = np.array(labels_val)
#     np.savez_compressed(f"val_dataset_{len(data)-split_idx}_exs.npz", inputs=inputs_arr[split_idx:], labels=labels_arr[split_idx:])

#     # generate val data
#     print("Generating val data...")
#     val_data = collect_sim_data(sim, num_sims_val, time_steps, height, width)
#     random.shuffle(val_data)
#     inputs_val, labels_val = zip(*val_data)
#     inputs_val_arr = np.stack(inputs_val, axis=0)
#     labels_val_arr = np.array(labels_val)
#     np.savez_compressed(f"val_dataset_{len(data)-split_idx}_exs.npz", inputs=inputs_arr[split_idx:], labels=labels_arr[split_idx:])
    

def test_sim(sim, time_steps, height, width):
    max_object_size = np.random.randint(1, height//4)
    num_obstacles = np.random.randint(0, height//2)
    occ_grids = sim.simulate(time_steps=time_steps, max_object_size=max_object_size, num_obstacles=num_obstacles)
    stgrid = st_grid.ST_grid(occ_grids)
    num_spatial_nodes = height * width
    start = np.random.randint(0, num_spatial_nodes)
    goal = np.random.randint(0, num_spatial_nodes)
    while start == goal:
        goal = np.random.randint(0, num_spatial_nodes)
    path, valid = path_finding.st_a_star(stgrid, start, goal)
    if valid:
        print("Valid path")
        start_p = occ_grids[0].idx2point(start)
        goal_p = occ_grids[0].idx2point(goal)
        viz_sim(stgrid, start_p, goal_p, path)

def generate_policy_data(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    h = 64
    w = 64
    time_steps = 100
    num_spatial_nodes = h*w
    sim = simulation.Simulation(h, w)

    IN_CHANNELS = 5
    NUM_ACTIONS = 9
    model = PolicyCNN(IN_CHANNELS, NUM_ACTIONS)
    model.load_state_dict(torch.load(model_name, map_location=device))
    model.to(device)
    model.eval()

def main():
    np.random.seed(0)
    height = 64
    width = 64
    time_steps = 100
    sim = simulation.Simulation(height, width)
    # for i in range(2):
    #     test_sim(sim, time_steps, height, width)
    generate_data_same_sims(sim, time_steps, height, width, num_sims=500)


if __name__ == "__main__":
    main()