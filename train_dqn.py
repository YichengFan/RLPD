import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from production_env_dqn import ProductionSchedulingEnv

# 1) Q‐Network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# 2) Agent
class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        buffer_size=100_000,
        batch_size=64,
        eps_start=1.0,
        eps_decay=0.995,
        eps_min=0.05,
        target_update_steps=1000
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = eps_start
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.learn_steps = 0
        self.target_update_steps = target_update_steps

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.policy_net.net[-1].out_features)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            qvals = self.policy_net(state_t)
        return int(qvals.argmax(dim=1).item())

    def store_transition(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert lists of numpy arrays into 2D arrays for fast tensor conversion
        # states_arr: shape [batch_size, state_dim]
        states_arr = np.stack(states).astype(np.float32)

        # next_states may contain None for terminals—replace with zeros
        zero_state = np.zeros_like(states_arr[0])
        next_states_arr = np.stack([
            ns if ns is not None else zero_state
            for ns in next_states
        ]).astype(np.float32)

        # Now move to torch tensors in one go
        states_t = torch.from_numpy(states_arr).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.from_numpy(next_states_arr).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Current Q estimates
        q_pred = self.policy_net(states_t).gather(1, actions_t)

        # Compute targets with target network
        with torch.no_grad():
            q_next = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            q_target = rewards_t + (1 - dones_t) * self.gamma * q_next

        # MSE loss and optimize
        loss = self.criterion(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update ε and target network
        self.learn_steps += 1
        if self.learn_steps % self.target_update_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)


# 3) Expert schedule loader
def load_expert_schedule(path, env):
    """
    Expects a CSV with columns:
      Date, Material, Planned_Production, Machine
    Skips any rows where Material or Machine is NaN.
    """
    df = pd.read_csv(path)

    # Drop rows missing the key columns
    df = df.dropna(subset=['Material', 'Machine'])

    actions = []
    for _, row in df.iterrows():
        material = str(row['Material']).strip()
        machine  = str(row['Machine']).strip()
        # parse machine index (e.g. 'Machine 1' -> 0)
        try:
            m_idx = int(machine.split()[-1]) - 1
        except:
            # skip malformed machine names
            continue

        # ensure this material actually exists in your environment
        if material not in env.product_idx:
            # you could also log a warning here
            continue

        p_idx = env.product_idx[material]
        s_idx = 0  # no shipment in expert data
        action = (
            m_idx * (env.num_products * env.num_shipment_types)
            + p_idx * env.num_shipment_types
            + s_idx
        )
        actions.append(action)

    return actions


# 4) Training loop with imitation
def train_dqn(
    num_episodes=500,
    pretrain_steps=10000,
    save_every=50,
    model_path='dqn_real'
):
    env = ProductionSchedulingEnv()
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-3,
        gamma=0.99,
        buffer_size=100_000,
        batch_size=64,
        eps_start=1.0,
        eps_decay=0.995,
        eps_min=0.05,
        target_update_steps=1000
    )

    # Imitation pretrain
    expert_actions = load_expert_schedule('Production_Schedules.csv', env)
    state = env.reset()
    for a in expert_actions:
        next_state, reward, done, _ = env.step(a)
        agent.store_transition(state, a, reward, next_state, done)
        state = next_state
        if done: break
    for _ in range(pretrain_steps):
        agent.update()

    # RL training
    for ep in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, = [agent.select_action(state)]
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward

        print(f"Episode {ep}/{num_episodes}  Reward: {total_reward:.2f}  Epsilon: {agent.epsilon:.3f}")
        if ep % save_every == 0:
            torch.save(agent.policy_net.state_dict(), f"{model_path}_ep{ep}.pth")

    # final save
    torch.save(agent.policy_net.state_dict(), f"{model_path}_final.pth")
    print("Training complete, models saved.")

if __name__ == '__main__':
    train_dqn()
