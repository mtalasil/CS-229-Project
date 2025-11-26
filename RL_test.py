import numpy as np
import torch, time
from RL_train import GridEnv, DIR_VECS, extract_local_patch_local, PolicyNet, ValueNet, build_local_value_field, sample_free_cell 
from astar_V1 import AStar, DetOccupancyGrid2D
import pandas as pd

# Run synchronized episode for both RL and A* agents
def run_synchronized_episode(env, policy, value_net, start, goal, patch_radius=3, device='cpu', max_steps=300):  
    rl_agent = np.array(start, dtype=int)
    astar_agent = np.array(start, dtype=int)
    rl_steps, astar_steps = 0, 0
    rl_time, astar_time = 0.0, 0.0
    rl_done, astar_done = False, False
    
    # Initialize position tracking for velocity estimation
    env.mob_prev_positions = {id(mob): mob.pos.copy() for mob in env.moving_obstacles}

    # Update steps and obstacles for each step
    for step in range(max_steps):
        env.update_moving_obstacles(goal)
        occ_grid = env.get_obstacle_grid()

        # RL Agent
        if not rl_done:
            t0 = time.time()
            if occ_grid[rl_agent[1], rl_agent[0]] != 0 or np.array_equal(rl_agent, goal):
                rl_done = True
            else:
                local_field, origin = build_local_value_field(env, rl_agent, goal, radius=8)
                patch = extract_local_patch_local(rl_agent, local_field, origin, patch_radius)
                feat_tensor = torch.from_numpy(patch).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = policy(feat_tensor)
                    probs = torch.softmax(logits, dim=-1)
                    action_idx = torch.argmax(probs, dim=-1).item()

                move = DIR_VECS[action_idx]
                nx, ny = np.clip(rl_agent + move, [0,0], [env.width-1, env.height-1])
                if occ_grid[ny, nx] == 0:
                    rl_agent = np.array([nx, ny])
                    rl_steps += 1
                    if np.array_equal(rl_agent, goal):
                        rl_done = True
            t1 = time.time()
            rl_time += t1 - t0

        # A* agent
        if not astar_done:
            t0 = time.time()
            if occ_grid[astar_agent[1], astar_agent[0]] != 0:
                astar_done = True
            else:
                grid = DetOccupancyGrid2D(env.width, env.height, occ_grid)
                astar = AStar((0,0),(env.width-1,env.height-1),
                            tuple(astar_agent), tuple(goal), grid)
                found = astar.solve()
                if found and len(astar.path) >= 2:
                    next_step = np.array(astar.path[1], dtype=int)
                    if occ_grid[next_step[1], next_step[0]] == 0:
                        astar_agent = next_step
                        astar_steps += 1
                        if np.array_equal(astar_agent, goal):
                            astar_done = True
            t1 = time.time()
            astar_time += t1 - t0

        if rl_done and astar_done:
            break

    return {
        'rl_success': np.array_equal(rl_agent, goal),
        'astar_success': np.array_equal(astar_agent, goal),
        'rl_steps': rl_steps,
        'astar_steps': astar_steps,
        'rl_time': rl_time,
        'astar_time': astar_time
    }

# Running test episodes
def compare_astar_vs_rl_from_csv(policy_path, value_path, input_csv, output_csv,
                                 patch_radius=3, device=None): 
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read randomly generated test episodes created with specific seed
    df = pd.read_csv(input_csv)
    results = []   # store each episode result

    for ep in range(len(df)):
        row = df.iloc[ep]
        width  = int(row["width"])
        height = int(row["height"])
        seed   = int(row["env_seed"])

        # Initialize environment with deterministic seed
        env = GridEnv(width, height, seed=seed)

        # Load policy/value networks
        in_dim = (2 * patch_radius + 1) ** 2
        policy = PolicyNet(in_dim).to(device)
        value_net = ValueNet(in_dim).to(device)
        policy.load_state_dict(torch.load(policy_path, map_location=device))
        value_net.load_state_dict(torch.load(value_path, map_location=device))
        policy.eval()
        value_net.eval()

        # Sample deterministic start/goal
        start = sample_free_cell(env)
        goal = sample_free_cell(env)
        while np.array_equal(start, goal):
            goal = sample_free_cell(env)

        # Run episode
        result = run_synchronized_episode(
            env, policy, value_net, start, goal,
            patch_radius=patch_radius, device=device
        )

        # Store results
        results.append({
            "seed": seed,
            "width": width,
            "height": height,

            "rl_success": int(result["rl_success"]),
            "rl_steps": result["rl_steps"],
            "rl_time": result["rl_time"],

            "astar_success": int(result["astar_success"]),
            "astar_steps": result["astar_steps"],
            "astar_time": result["astar_time"],
        })
 
        print(
            f"Episode {ep+1}/{len(df)} | {width}x{height} seed={seed} | "
            f"RL={'OK' if result['rl_success'] else 'FAIL'} "
            f"(steps={result['rl_steps']} time={result['rl_time']:.4f}) | "
            f"A*={'OK' if result['astar_success'] else 'FAIL'} "
            f"(steps={result['astar_steps']} time={result['astar_time']:.4f})"
        )

    # Convert to dataframe and save
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"\n✔ Saved results to: {output_csv}")


if __name__ == "__main__":
    
    compare_astar_vs_rl_from_csv(
        "policy.pth",
        "value.pth",
        "episodes_test.csv",
        "results_rl_vs_astar.csv",
    )
    