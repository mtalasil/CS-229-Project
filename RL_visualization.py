import numpy as np
import torch
import time
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from GridEnvironment import DIR_VECS, sample_free_cell, MovingObstacle, GridEnv
from astar_V1 import AStar, DetOccupancyGrid2D
from RL_train import extract_local_patch_local, PolicyNet, ValueNet, sample_free_cell, build_local_value_field
from astar_V1 import AStar, DetOccupancyGrid2D

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

    # Trajectories
    rl_traj_line, = ax.plot([], [], color='blue', linewidth=2)
    astar_traj_line, = ax.plot([], [], color='magenta', linewidth=2)

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
        'rl_traj_line': rl_traj_line,
        'astar_traj_line': astar_traj_line,
        'mob_patches': mob_patches
    }

# Updated visualization for each time step
def update_visualization(artists, rl_traj, astar_traj, env):
    rl_agent = rl_traj[-1]
    astar_agent = astar_traj[-1]

    # Update agent points
    artists['rl_point'].set_offsets([rl_agent[0], rl_agent[1]])
    artists['astar_point'].set_offsets([astar_agent[0], astar_agent[1]])

    # Update trajectories
    if len(rl_traj) > 1:
        artists['rl_traj_line'].set_data([p[0] for p in rl_traj], [p[1] for p in rl_traj])
    if len(astar_traj) > 1:
        artists['astar_traj_line'].set_data([p[0] for p in astar_traj], [p[1] for p in astar_traj])

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
def run_synchronized_episode(env, policy, value_net, start, goal, patch_radius=3, device='cpu', max_steps=300):
    rl_agent = np.array(start, dtype=int)
    astar_agent = np.array(start, dtype=int)
    rl_trajectory = [rl_agent.copy()]
    astar_trajectory = [astar_agent.copy()]
    mob_trajs = [[] for _ in env.moving_obstacles]
    rl_active, astar_active = True, True

    artists = init_visualization(env, start, goal)

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

        update_visualization(artists, rl_trajectory, astar_trajectory, env)

        if not rl_active and not astar_active:
            break
        time.sleep(0.1)

    return {
        'rl': {'success': np.array_equal(rl_agent, goal), 'trajectory': rl_trajectory},
        'astar': {'success': np.array_equal(astar_agent, goal), 'trajectory': astar_trajectory},
        'traj': {'mobs_traj': mob_trajs}
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patch_radius = 3

    # Randomized grid size
    width = np.random.randint(32, 129)
    height = np.random.randint(32, 129)

    in_dim = (2*patch_radius + 1)**2
    policy = PolicyNet(in_dim).to(device)
    value_net = ValueNet(in_dim).to(device)
    policy.load_state_dict(torch.load("policy.pth", map_location=device))
    value_net.load_state_dict(torch.load("value.pth", map_location=device))
    policy.eval()
    value_net.eval()

    env = GridEnv(width, height)
    start = sample_free_cell(env)
    goal = sample_free_cell(env)
    while np.array_equal(start, goal):
        start = sample_free_cell(env)

    print(f"Grid: {width}x{height}, Start: {start}, Goal: {goal}")
    results = run_synchronized_episode(env, policy, value_net, start, goal, patch_radius, device)
    print("RL Success:", results['rl']['success'])
    print("A* Success:", results['astar']['success'])

if __name__ == "__main__":
    main()
    plt.show(block=True) 
