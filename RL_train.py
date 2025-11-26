import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import pandas as pd
from GridEnvironment import MovingObstacle, GridEnv, sample_free_cell, DIR_VECS

# Local Gaussian Value Field reward map
# Each cell is given a score indicating how rewarding it is for the agent
def build_local_value_field(env, agent_pos, goal, radius=8):
    
    ax, ay = agent_pos
    # Clamp window to grid boundaries (size is 2*radius +1)
    w0, w1 = max(0, ax-radius), min(env.width-1, ax+radius)
    h0, h1 = max(0, ay-radius), min(env.height-1, ay+radius)
    W, H = w1-w0+1, h1-h0+1

    # Initialize value field
    xs, ys = np.arange(w0, w1+1), np.arange(h0, h1+1)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing='xy')
    val = np.zeros((H, W), dtype=np.float32)

    # Positive Gaussian reward around goal centered at the goal
    dx, dy = grid_x - goal[0], grid_y - goal[1]
    dist = np.linalg.norm(agent_pos - np.array(goal))
    sigma_goal = max(0.2, 60*(dist/128))
    val += 110 * np.exp(-(dx**2 + dy**2)/(2*sigma_goal**2))
    
    # Negative value for static obstacles
    val[env.static_obstacles[h0:h1+1, w0:w1+1] == 1] -= 50

    # Negative Gaussian around moving obstacles
    for mob in env.moving_obstacles:
        x, y = mob.pos
        w, h = mob.size

        # Skip if obstacle does not overlap with the window
        if x+w < w0 or x > w1 or y+h < h0 or y > h1:
            continue
        cx, cy = x + w/2, y + h/2
        sigma_x, sigma_y = w/2, h/2
        val += -50 * np.exp(-(((grid_x-cx)**2)/(2*sigma_x**2) + ((grid_y-cy)**2)/(2*sigma_y**2)))

    # Initialize previous positions for predictive penalty
    if not hasattr(env, 'mob_prev_positions'):
        env.mob_prev_positions = {id(mob): mob.pos.copy() for mob in env.moving_obstacles}

    # Light predictive penalty for moving obstacles based on obstacle velocity calculation from t and t-1 
    for mob in env.moving_obstacles:
        mob_id = id(mob)
        prev_pos = env.mob_prev_positions.get(mob_id, mob.pos.copy())
        velocity = mob.pos - prev_pos
        vx, vy = int(np.sign(velocity[0])), int(np.sign(velocity[1]))
        pred_x, pred_y = int(round(mob.pos[0] + vx)), int(round(mob.pos[1] + vy))
        w, h = mob.size

        for dy in range(h):
            for dx in range(w):
                cx2, cy2 = pred_x+dx, pred_y+dy
                if w0 <= cx2 <= w1 and h0 <= cy2 <= h1:
                    val[cy2-h0, cx2-w0] -= 20  # light predictive penalty

        env.mob_prev_positions[mob_id] = mob.pos.copy()

    return val, (w0, h0) # val = local value map and (w0,h0) = top left origin of local path in the global grid

# Local feature extraction for Neural Network
# Extracts 7x7 field around the agent
# Copy values from value field in patch
# Snapshot of environment at timestep t
def extract_local_patch_local(agent_pos, local_field, origin, patch_radius=3):
    ax, ay = agent_pos
    ox, oy = origin
    lx, ly = ax - ox, ay - oy
    patch = np.zeros((2*patch_radius+1, 2*patch_radius+1), dtype=np.float32)

    for dy in range(-patch_radius, patch_radius+1):
        for dx in range(-patch_radius, patch_radius+1):
            px, py = lx+dx, ly+dy
            if 0 <= px < local_field.shape[1] and 0 <= py < local_field.shape[0]:
                patch[dy+patch_radius, dx+patch_radius] = local_field[py, px]
    return patch.ravel() # converts patch into 1D array with 49 entries for NN input

# Policy and Value networks
# Given value field, outputs action probabilities for the 8 possible directions 
# ReLU activation function and 2 hidden layers, each with 128 neurons
# Value Field on entire grid -> 7x7 local path -> policy outputs probabilities (eventually) -> agent samples & chooses action -> agent receives reward
# Actor - decides what action to take
# 2 hidden layers with 128 neurons each and ReLU activation
class PolicyNet(nn.Module): 
    def __init__(self, in_dim, hidden=128, out_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): 
        return self.net(x)

class ValueNet(nn.Module): # Estimates expected reward from current state (outputs single scalar value)
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x)

# RL episode (TD(0) actor-critic)
# TD (temporal difference) -> updates estimates of future rewards based on next step instead of waiting for the end of the episode
# One attempt for agent to move from start to goal
def run_episode(env, policy, value_net, max_steps=200, patch_radius=3, gamma=0.95, device='cpu', visualize=False):
    import matplotlib.pyplot as plt

    env.reset() # reset obstacles, start, and goal each episode
    env.mob_prev_positions = {id(mob): mob.pos.copy() for mob in env.moving_obstacles} # initializes new prev positions attribute without changing environment

    start = sample_free_cell(env)
    goal = sample_free_cell(env)
    while start == goal: goal = sample_free_cell(env)
    agent = np.array(start)

    traj, log_probs, values, targets, rewards = [], [], [], [], []
    reward_list = []
    prev_dist = np.linalg.norm(agent - np.array(goal))
    visited = set([tuple(agent)])

    if visualize:
        fig, axes = plt.subplots(1,2, figsize=(12,5))
        plt.ion()

    for step in range(max_steps):
        # Update moving obstacles
        env.update_moving_obstacles(goal)
        occ_grid = env.get_obstacle_grid()

        # Build local value field with shrinking goal Gaussian
        local_field, origin = build_local_value_field(env, agent, goal, radius=8)

        # Feature extraction
        feat = extract_local_patch_local(agent, local_field, origin, patch_radius)
        feat_tensor = torch.from_numpy(feat).float().unsqueeze(0).to(device)

        # Policy & value
        logits = policy(feat_tensor)
        probs = torch.softmax(logits, dim=-1) # probabilities for each direction
        m = torch.distributions.Categorical(probs) # creates probability distribution object
        action_idx = m.sample() # randomly chooses move weighted by assigned probabilities which allows for exploration
        log_prob = m.log_prob(action_idx).squeeze(0)
        value = value_net(feat_tensor).squeeze(0).squeeze(-1)

        # Execute move
        move = DIR_VECS[action_idx.item()]
        nx, ny = np.clip(agent + move, [0,0], [env.width-1, env.height-1])
        agent_next = np.array([nx, ny]) if occ_grid[ny, nx]==0 else agent.copy()
        lx, ly = agent_next[0]-origin[0], agent_next[1]-origin[1]

        # Compute reward
        reward = local_field[ly, lx] if 0<=lx<local_field.shape[1] and 0<=ly<local_field.shape[0] else 0 # reward from agent's move in local field
        if tuple(agent_next) in visited: reward -= 15 # revisit penalty
        if occ_grid[ny, nx] != 0: reward -= 50 # additional obstacle penalty
        if np.linalg.norm(agent_next - np.array(goal)) >= prev_dist: reward -= 15 # moving away from goal penalty
        reward -= 20 # step penalty
        prev_dist = np.linalg.norm(agent_next - np.array(goal))
        done = tuple(agent_next) == goal
        if done: reward += 10000 # goal reward

        # TD target
        next_value = value_net(torch.from_numpy(extract_local_patch_local(agent_next, local_field, origin, patch_radius)).float().unsqueeze(0).to(device)).squeeze(0).squeeze(-1) # prediction of next_value
        target = torch.tensor(float(reward), device=device) + gamma*next_value # immediate reward + discounted estimated value of next state

        # Record
        log_probs.append(log_prob)
        values.append(value)
        targets.append(target)
        rewards.append(reward)
        traj.append(tuple(agent_next.tolist()))
        reward_list.append(reward)
        agent = agent_next
        visited.add(tuple(agent_next))

        # Visualization
        if visualize:
            axes[0].cla()
            axes[0].imshow(local_field, cmap='viridis', origin='lower',
                           extent=[origin[0], origin[0]+local_field.shape[1],
                                   origin[1], origin[1]+local_field.shape[0]])
            obs_y, obs_x = np.where(env.static_obstacles==1)
            axes[0].scatter(obs_x, obs_y, marker='s', color='black', s=30)
            for mob in env.moving_obstacles:
                x, y = mob.pos
                w, h = mob.size
                axes[0].add_patch(plt.Rectangle((x,y), w, h, color='red', alpha=0.5))
            axes[0].scatter(start[0], start[1], color='green', marker='o')
            axes[0].scatter(goal[0], goal[1], color='yellow', marker='*', s=100)
            axes[0].scatter(agent[0], agent[1], color='red')
            axes[1].cla()
            axes[1].plot(reward_list)
            plt.pause(0.05)

        if done: break

    if visualize:
        plt.ioff()
        plt.show()

    advantages = torch.stack(targets) - torch.stack(values) if values else torch.tensor([]) # how action compares to expectation
    return {'log_probs': torch.stack(log_probs) if log_probs else torch.tensor([]),
            'values': torch.stack(values) if values else torch.tensor([]),
            'targets': torch.stack(targets) if targets else torch.tensor([]),
            'advantages': advantages,
            'reward_sum': sum(rewards),
            'traj_len': len(traj),
            'traj': traj,
            'start': start,
            'goal': goal}

# Training (Actor-Critic TD(0))
def train_rl(policy, value_net, episodes=500, max_steps=200, patch_radius=3,
             lr=1e-4, gamma=0.95, file=None, device=None, visualize_interval=10):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Reads environment info for each episode from episodes_train.csv
    df = pd.read_csv(file)
    policy.to(device)
    value_net.to(device)

    # Adaptive moment estimation optimizer that tracks past movement (uses first and second moment of gradient descent)
    opt_pol = optim.Adam(policy.parameters(), lr=lr) # to maximize expected reward
    opt_val = optim.Adam(value_net.parameters(), lr=lr) # to minimize prediction error

    plt.ion() # allow live updates
    avg_rewards = []
    for ep in range(episodes):
        row = df.iloc[ep]
        width, height = int(row['width']), int(row['height'])
        seed = int(row['env_seed'])

        env = GridEnv(width, height, seed=seed)
        ep_data = run_episode(env, policy, value_net, max_steps, patch_radius, gamma, device)

        log_probs = ep_data['log_probs']
        values = ep_data['values']
        targets = ep_data['targets']
        advantages = ep_data['advantages']

        # Losses
        loss_val = nn.functional.mse_loss(values, targets.detach()) # Value loss: uses mean squared error between predicted values and TD(0) targets
        loss_pol = -(log_probs * advantages.detach()).sum() # Policy loss: loss between expected value of actions vs actual actions value (tries to maximize expected reward)

        opt_pol.zero_grad(); loss_pol.backward(); opt_pol.step()
        opt_val.zero_grad(); loss_val.backward(); opt_val.step()

        if (ep+1) % 1 == 0:
            print(f"[Episode {ep+1:4d}] Grid={width}x{height} Reward={ep_data['reward_sum']:.2f} Steps={ep_data['traj_len']}")
        avg_rewards.append(ep_data['reward_sum'])
        if (ep+1) % visualize_interval == 0:
            vis_env = GridEnv(width, height)
            run_episode(vis_env, policy, value_net, max_steps, patch_radius, gamma, device, visualize=False)

    plt.ioff()

    n = 100
    avg_over_100 = [np.mean(avg_rewards[i:i+n]) for i in range(0, len(avg_rewards), n)]
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(avg_over_100)+1), avg_over_100, marker='o')
    plt.xlabel('Blocks of 100 Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward per 100 Episodes')
    plt.grid(True)
    plt.show()
    return policy, value_net

# Main
def main():
    patch_radius = 3
    in_dim = (2*patch_radius + 1)**2 # input features dimensions
    policy = PolicyNet(in_dim)
    value_net = ValueNet(in_dim)
    training_csv = "episodes_train.csv"

    policy, value_net = train_rl(policy, value_net, episodes=4000, max_steps=200,
                                 patch_radius=patch_radius, lr=1e-4, gamma=0.95,
                                 file=training_csv, visualize_interval=50)

    torch.save(policy.state_dict(), "policy.pth")
    torch.save(value_net.state_dict(), "value.pth")

    env = GridEnv(32, 32)
    run_episode(env, policy, value_net, max_steps=200, patch_radius=patch_radius, visualize=False)

if __name__ == "__main__":
    main()
