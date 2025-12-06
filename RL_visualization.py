import numpy as np
import torch
import time
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from GridEnvironment import DIR_VECS, sample_free_cell, MovingObstacle, GridEnv
from astar_V1 import AStar, DetOccupancyGrid2D
from RL_train import extract_local_patch_local, PolicyNet, ValueNet, sample_free_cell, build_local_value_field
from astar_V1 import AStar, DetOccupancyGrid2D
import copy
from valueCNN import ValueCNN

action_costs = [
    #(Point([0, 0]), 0),
    ((-1, -1), np.sqrt(2)),
    ((-1, 0), 1),
    ((-1, 1), np.sqrt(2)),
    ((0, 1), 1),
    ((1, 1), np.sqrt(2)),
    ((1, 0), 1),
    ((1, -1), np.sqrt(2)),
    ((0, -1), 1)
]

def get_patch(occ_grid, unet_agent, ph, pw):
    oh = occ_grid.shape[0]
    ow = occ_grid.shape[1]
    patch = np.full((ph, pw), -1, dtype=np.float32)
    offset = ph // 2
    tl = (unet_agent[0]-offset, unet_agent[1]-offset)
    br = (unet_agent[0]+offset, unet_agent[1]+offset)

    x_min = max(tl[0], 0) 
    x_max = min(br[0], ow)
    y_min = max(tl[1], 0)
    y_max = min(br[1], oh)

    x_min_diff = x_min - tl[0]
    x_max_diff = br[0] - x_max
    y_min_diff = y_min - tl[1]
    y_max_diff = br[1] - y_max

    p_x_min = 0 + x_min_diff
    p_x_max = pw - x_max_diff
    p_y_min = 0 + y_min_diff
    p_y_max = ph - y_max_diff 

    patch[p_y_min:p_y_max, p_x_min:p_x_max] = occ_grid[y_min:y_max, x_min:x_max]

    return patch

def patch_goal(goal, local_agent, unet_agent, patch):
    dx = goal[0]-unet_agent[0]
    dy = goal[1]-unet_agent[1]
    mag = np.sqrt(dx**2 + dy**2)
    unit_v = [dx / mag, dy / mag]
    start_x = local_agent[0]
    start_y = local_agent[1]
    best = [start_x, start_y]
    while True:
        start_x += unit_v[0]
        start_y += unit_v[1]
        closest_x = int(round(start_x))
        closest_y = int(round(start_y))
        if closest_x < 0 or closest_x >= patch.shape[1] or closest_y < 0 or closest_y >= patch.shape[0]:
            break
        if abs(patch[closest_y, closest_x]) < 1e-6:
            best[0] = closest_x
            best[1] = closest_y
    return best

def is_obstacle(grid, pose):
    return grid[pose[1], pose[0]] > 1e-6

def heuristic(curr, goal):
    return np.sqrt((curr[0]-goal[0])**2 + (curr[1]-goal[1])**2)

# Initial visualization 
def init_visualization(env, start, goal):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlim(-0.5, env.width-0.5)
    ax.set_ylim(-0.5, env.height-0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.imshow(np.ones((env.height, env.width)), cmap='gray', origin='upper', vmin=0, vmax=1)

    # Static obstacles
    static_y, static_x = np.where(env.static_obstacles == 1)
    ax.scatter(static_x, static_y, color='black', marker='s', s=50)

    # Start and goal
    ax.scatter(start[0], start[1], color='green', marker='o', s=100, label='Start')
    ax.scatter(goal[0], goal[1], color='red', marker='*', s=150, label='Goal')

    # RL and A* agents
    rl_point = ax.scatter([], [], color='blue', marker='o', s=100, label='RL Agent')
    astar_point = ax.scatter([], [], color='magenta', marker='x', s=100, label='A* Agent')
    unet_point = ax.scatter([], [], color='green', marker='x', s=100, label='UNET agent')

    # Trajectories
    rl_traj_line, = ax.plot([], [], color='blue', linewidth=2)
    astar_traj_line, = ax.plot([], [], color='magenta', linewidth=2)
    unet_traj_line, = ax.plot([], [], color='green', linewidth=2)

    # Moving obstacles
    mob_patches = []
    for mob in env.moving_obstacles:
        x, y = mob.pos
        w, h = mob.size
        patch = patches.Rectangle((x-0.5, y-0.5), w, h, color='red', alpha=0.6)
        ax.add_patch(patch)
        mob_patches.append(patch)

    ax.legend(loc='upper right')
    plt.ion()
    plt.show()

    return {
        'fig': fig,
        'ax': ax,
        'rl_point': rl_point,
        'astar_point': astar_point,
        'unet_point': unet_point,
        'rl_traj_line': rl_traj_line,
        'astar_traj_line': astar_traj_line,
        'unet_traj_line': unet_traj_line,
        'mob_patches': mob_patches
    }

# Updated visualization for each time step
def update_visualization(artists, rl_traj, unet_traj, astar_traj, env):
    rl_agent = rl_traj[-1]
    astar_agent = astar_traj[-1]
    unet_agent = unet_traj[-1]

    # Update agent points
    artists['rl_point'].set_offsets([rl_agent[0], rl_agent[1]])
    artists['astar_point'].set_offsets([astar_agent[0], astar_agent[1]])
    artists['unet_point'].set_offsets([unet_agent[0], unet_agent[1]])

    # Update trajectories
    if len(rl_traj) > 1:
        artists['rl_traj_line'].set_data([p[0] for p in rl_traj], [p[1] for p in rl_traj])
    if len(astar_traj) > 1:
        artists['astar_traj_line'].set_data([p[0] for p in astar_traj], [p[1] for p in astar_traj])
    if len(unet_traj) > 1:
        artists['unet_traj_line'].set_data([p[0] for p in unet_traj], [p[1] for p in unet_traj])

    # Update moving obstacles
    for i, mob in enumerate(env.moving_obstacles):
        patch = artists['mob_patches'][i]
        x, y = mob.pos
        w, h = mob.size
        patch.set_xy([x-0.5, y-0.5])
        patch.set_width(w)
        patch.set_height(h)

    plt.pause(0.1)

# Run synchronized RL + A* episode
def run_synchronized_episode(env, policy, value_net, unet, start, goal, patch_radius=3, device='cpu', max_steps=300):
    rl_agent = np.array(start, dtype=int)
    astar_agent = np.array(start, dtype=int)
    unet_agent = np.array(start, dtype=int)
    rl_trajectory = [rl_agent.copy()]
    astar_trajectory = [astar_agent.copy()]
    unet_trajectory = [unet_agent.copy()]
    mob_trajs = [[] for _ in env.moving_obstacles]
    unet_active, rl_active, astar_active = True, True, True

    artists = init_visualization(env, start, goal)

    unet_window_h = 64
    unet_window_w = 64
    prev_1_patch = None
    prev_2_patch = None
    norm_const = 1/(max_steps * np.sqrt(2))
    heuristic_bonus_base = 1.0
    heuristic_bonus = heuristic_bonus_base

    for step in range(max_steps):
        env.update_moving_obstacles(goal)
        occ_grid = env.get_obstacle_grid()

        for i, mob in enumerate(env.moving_obstacles):
            mob_trajs[i].append((mob.pos.copy(), mob.size))

        # RL agent
        if rl_active:
            if occ_grid[rl_agent[1], rl_agent[0]] != 0 or np.array_equal(rl_agent, goal):
                rl_active = False
            else:
                local_field, origin = build_local_value_field(env, rl_agent, goal, radius=10)
                patch = extract_local_patch_local(rl_agent, local_field, origin, patch_radius)
                feat_tensor = torch.from_numpy(patch).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = policy(feat_tensor)
                    probs = torch.softmax(logits, dim=-1)
                    action_idx = torch.argmax(probs, dim=-1).item()

                move = DIR_VECS[action_idx]
                nx, ny = np.clip(rl_agent + move, [0,0], [env.width-1, env.height-1])
                if occ_grid[ny, nx] == 0:
                    rl_agent = np.array([nx, ny], dtype=int)
                    rl_trajectory.append(rl_agent.copy())
                else:
                    rl_active = False

        # unet agent
        if unet_active:
            if occ_grid[unet_agent[1], unet_agent[0]] > 1e-6:
                unet_active = False
            else:
                if step >= 1:
                    prev_2_patch = copy.deepcopy(prev_1_patch)
                    prev_1_patch = copy.deepcopy(model_patch)
                raw_patch = get_patch(occ_grid, unet_agent, unet_window_h, unet_window_w)
                out_of_bounds_mask = raw_patch < 0
                patch = copy.deepcopy(raw_patch)
                patch[out_of_bounds_mask] = 1.0
                model_patch = copy.deepcopy(raw_patch)
                model_patch[out_of_bounds_mask] = 0.0
                local_agent = (unet_window_w//2, unet_window_h//2)
                t_x = unet_agent[0] - local_agent[0]
                t_y = unet_agent[1] - local_agent[1]
                local_goal = [goal[0] - t_x, goal[1] - t_y]
                if local_goal[0] < 0 or local_goal[0] >= unet_window_w or local_goal[1] < 0 or local_goal[1] >= unet_window_h:
                    local_goal = patch_goal(goal, local_agent, unet_agent, patch)
                if step == 0:
                    prev_1_patch = copy.deepcopy(model_patch)
                    prev_2_patch = copy.deepcopy(model_patch)
                ys, xs = np.meshgrid(np.arange(unet_window_h), np.arange(unet_window_w), indexing='ij')
                goal_dx = (local_goal[0] - xs) / unet_window_w
                goal_dy = (local_goal[1] - ys) / unet_window_h

                curr_stack = np.stack([prev_2_patch, prev_1_patch, model_patch, goal_dy, goal_dx], axis=0)
                curr_state = torch.from_numpy(curr_stack).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    J_raw = unet(curr_state)
                J = J_raw.squeeze().cpu().numpy()

                min_cost = np.inf
                best_action = None
                for a, cost in action_costs:
                    pot_next_pose = (local_agent[0] + a[0], local_agent[1] + a[1])
                    if is_obstacle(patch, pot_next_pose):
                        continue
                    next_cost = norm_const * cost + J[pot_next_pose[1], pot_next_pose[0]] + norm_const * heuristic_bonus * heuristic(pot_next_pose, local_goal) # + immediate cost if you want but you MUST normalize by some constant or it will dominate (ex time_steps * sqrt(2) -> which is longest possible path given time horizon)
                    if next_cost < min_cost:
                        min_cost = next_cost
                        best_action = a
                if best_action is None:
                    unet_active = False
                    print("no unet action :(")
                else:
                    unet_agent[0] += best_action[0]
                    unet_agent[1] += best_action[1]
                    unet_trajectory.append(unet_agent.copy())
                    if np.array_equal(unet_agent, goal):
                        unet_active = False

        # A* agent
        if astar_active:
            if occ_grid[astar_agent[1], astar_agent[0]] != 0 or np.array_equal(astar_agent, goal):
                astar_active = False
            else:
                grid = DetOccupancyGrid2D(env.width, env.height, occ_grid)
                astar = AStar((0,0),(env.width-1, env.height-1), tuple(astar_agent), tuple(goal), grid)
                found = astar.solve()
                if not found or len(astar.path) < 2:
                    astar_active = False
                else:
                    next_step = np.array(astar.path[1], dtype=int)
                    if occ_grid[next_step[1], next_step[0]] == 0:
                        astar_agent = next_step
                        astar_trajectory.append(astar_agent.copy())
                    else:
                        astar_active = False

        update_visualization(artists, rl_trajectory, unet_trajectory, astar_trajectory, env)

        if not rl_active and not astar_active and not unet_active:
            break
        time.sleep(0.1)

    return {
        'rl': {'success': np.array_equal(rl_agent, goal), 'trajectory': rl_trajectory},
        'astar': {'success': np.array_equal(astar_agent, goal), 'trajectory': astar_trajectory},
        'unet': {'success': np.array_equal(unet_agent, goal), 'trajectory': unet_trajectory},
        'traj': {'mobs_traj': mob_trajs}
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patch_radius = 3

    # Randomized grid size
    width = 128#np.random.randint(32, 129)
    height = 128#np.random.randint(32, 129)

    in_dim = (2*patch_radius + 1)**2
    policy = PolicyNet(in_dim).to(device)
    value_net = ValueNet(in_dim).to(device)
    policy.load_state_dict(torch.load("policy.pth", map_location=device))
    value_net.load_state_dict(torch.load("value.pth", map_location=device))
    policy.eval()
    value_net.eval()

    IN_CHANNELS = 5
    base_ch = 4
    unet = ValueCNN(IN_CHANNELS, base_ch)
    unet.load_state_dict(torch.load(f"unet_base{base_ch}_cnn_model_epoch_6.pt", map_location=device))
    unet.to(device)
    unet.eval()

    env = GridEnv(width, height)
    start = sample_free_cell(env)
    goal = sample_free_cell(env)
    while np.array_equal(start, goal):
        start = sample_free_cell(env)

    print(f"Grid: {width}x{height}, Start: {start}, Goal: {goal}")
    results = run_synchronized_episode(env, policy, value_net, unet, start, goal, patch_radius, device)
    print("RL Success:", results['rl']['success'])
    print("A* Success:", results['astar']['success'])
    print("UNET Success:", results['unet']['success'])
    print("RL path length:", len(results['rl']['trajectory']))
    print("UNET path length:", len(results['unet']['trajectory']))
    print("A* path length:", len(results['astar']['trajectory']))

if __name__ == "__main__":
    main()
    plt.show(block=True) 
