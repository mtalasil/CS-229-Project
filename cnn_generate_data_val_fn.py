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

action_costs = [
    (Point([0, 0]), 0),
    (Point([-1, -1]), np.sqrt(2)),
    (Point([-1, 0]), 1),
    (Point([-1, 1]), np.sqrt(2)),
    (Point([0, 1]), 1),
    (Point([1, 1]), np.sqrt(2)),
    (Point([1, 0]), 1),
    (Point([1, -1]), np.sqrt(2)),
    (Point([0, -1]), 1)
]

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

def normalize_value_fns(value_fns):
    T = value_fns.shape[0]
    masks =  np.zeros_like(value_fns, dtype=np.bool)
    norm_values = np.zeros_like(value_fns)
    for t in range(T):
        masks[t] = np.isfinite(value_fns[t])
        if np.sum(masks[t]) == 0:
            continue
        norm_const = np.max(value_fns[t][masks[t]])
        if abs(norm_const) < 1e-6:
            continue
        norm_values[t][masks[t]] = value_fns[t][masks[t]] / norm_const
        norm_values[t][~masks[t]] = 1.0
    return norm_values, masks

def calc_value_fns(stgrid, goal):
    h = stgrid.h
    w = stgrid.w
    T = len(stgrid.grid)
    J = np.full((T, h, w), np.inf, dtype=np.float32)
    for t in range(T):
        if stgrid.grid[t].grid[goal[0], goal[1]] == 0:
            J[t, goal[0], goal[1]] = 0.0

    for t in range(T-2, -1, -1):
        for y in range(h):
            for x in range(w):
                if stgrid.grid[t].grid[y, x] == 1:
                    continue
                min_cost = np.inf
                for a, cost in action_costs:
                    next_pose = a + Point([y, x])
                    n_y = next_pose[0]
                    n_x = next_pose[1]
                    if n_x < 0 or n_x >= w or n_y < 0 or n_y >= h or stgrid.grid[t+1].grid[n_y, n_x] == 1:
                        continue
                    next_pose_value = J[t+1, next_pose[0], next_pose[1]]
                    min_cost = min(min_cost, cost +  next_pose_value)
                J[t, y, x] = min_cost
    return J

def find_full_vf_timestep(occ_grids, vf_masks):
    T = len(occ_grids)
    for t in range(T-1, -1, -1):
        occ_mask = occ_grids[t].grid == 0
        if np.all(vf_masks[t][occ_mask]):
            return t
    return -1

def collect_sim_data(simulation, num_sims, time_steps, h, w):
    stacks = []
    num_spatial_nodes = h * w
    for sim in range(num_sims):
        print(f"On sim {sim}...")
        max_object_size = np.random.randint(1, h//4)
        num_obstacles = np.random.randint(0, h//3)
        occ_grids = simulation.simulate(time_steps=time_steps, max_object_size=max_object_size, num_obstacles=num_obstacles)
        stgrid = st_grid.ST_grid(occ_grids)
        start = np.random.randint(0, num_spatial_nodes)
  
        goal = np.random.randint(0, num_spatial_nodes)
        while start == goal or occ_grids[0].near_obstacle(start) or occ_grids[0].is_obstacle(goal):
            start = np.random.randint(0, num_spatial_nodes)
            goal = np.random.randint(0, num_spatial_nodes)
        goal_p = occ_grids[0].idx2point(goal)
        start_p = occ_grids[0].idx2point(start)
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        goal_dx = (goal_p[1] - xs) / w
        goal_dy = (goal_p[0] - ys) / h

        # run backwards DP to generate list of value functions
        value_fns = calc_value_fns(stgrid, goal_p)
        norm_vfs, vf_masks = normalize_value_fns(value_fns)
        valid_t = find_full_vf_timestep(occ_grids, vf_masks)
        plot_field(norm_vfs[0], h, w, goal_p, start_p, three_d=False)

        if valid_t < 2:
            continue
        for t in range(2, valid_t+1):
            curr_stack = np.stack([occ_grids[t-2].grid, occ_grids[t-1].grid, occ_grids[t].grid, goal_dy, goal_dx], axis=0)
            stacks.append((curr_stack, norm_vfs[t], vf_masks[t]))
            plot_field(norm_vfs[t], h, w, goal_p, start_p, three_d=True)
    return stacks


def generate_data_same_sims(sim, time_steps, height, width, f_name, num_sims=100):
    # generate training data
    print("Generating data...")
    data = collect_sim_data(sim, num_sims, time_steps, height, width)
    random.shuffle(data)
    inputs, labels, masks = zip(*data)
    inputs_arr = np.stack(inputs, axis=0)
    labels_arr = np.array(labels)
    masks_arr = np.stack(masks, axis=0)

    train_inputs = inputs_arr
    train_labels = labels_arr
    train_masks  = masks_arr

    # val_inputs = inputs_arr[split_idx:]
    # val_labels = labels_arr[split_idx:]
    # val_masks  = masks_arr[split_idx:]

    save_sharded(f"{f_name}_{len(data)}_exs", train_inputs, train_labels, train_masks)
    #save_sharded(f"sharded_data/val_dataset_{len(data)-split_idx}_exs",   val_inputs,   val_labels,   val_masks)

    #np.savez_compressed(f"train_dataset_{split_idx}_exs.npz", inputs=, labels=labels_arr[0:split_idx], masks=masks_arr[0:split_idx])
    #np.savez_compressed(f"val_dataset_{len(data)-split_idx}_exs.npz", inputs=inputs_arr[split_idx:], labels=labels_arr[split_idx:], masks=masks_arr[split_idx:])

def save_sharded(prefix, inputs, labels, masks, shard_size=10000):
    N = inputs.shape[0]
    num_shards = (N + shard_size - 1) // shard_size

    for i in range(num_shards):
        s = i * shard_size
        e = min((i + 1) * shard_size, N)
        
        fname = f"{prefix}_shard_{i:02d}.npz"
        print(f"Saving {fname} with {e - s} examples...")

        np.savez_compressed(
            fname,
            inputs=inputs[s:e],
            labels=labels[s:e],
            masks=masks[s:e],
        )

def save_sharded_w(prefix, inputs, labels, masks, weights, shard_size=10000):
    N = inputs.shape[0]
    num_shards = (N + shard_size - 1) // shard_size

    for i in range(num_shards):
        s = i * shard_size
        e = min((i + 1) * shard_size, N)
        
        fname = f"{prefix}_shard_{i:02d}.npz"
        print(f"Saving {fname} with {e - s} examples...")

        np.savez_compressed(
            fname,
            inputs=inputs[s:e],
            labels=labels[s:e],
            masks=masks[s:e],
            weights=weights[s:e],
        )

def main():
    np.random.seed(2)
    # training data: seed 2
    # val data: seed 3
    height = 64
    width = 64
    time_steps = 100
    sim = simulation.Simulation(height, width)
    # for i in range(2):
    #     test_sim(sim, time_steps, height, width)
    generate_data_same_sims(sim, time_steps, height, width, "val_data/val_dataset", num_sims=1)


if __name__ == "__main__":
    main()