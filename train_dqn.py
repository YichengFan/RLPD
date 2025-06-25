import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from production_env_dqn import ProductionSchedulingEnv
import matplotlib.pyplot as plt

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
        lr=5e-5,
        gamma=0.99,
        buffer_size=100_000,
        batch_size=128,
        eps_start=1.0,
        eps_decay=0.9995,
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

        self.memory = PrioritizedReplayBuffer(capacity=100000, alpha=0.6)
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

    def store_transition(self, state, action, reward, next_state, done):
        # PER: priority will default to current max so new samples get drawn
        transition = (state, action, reward, next_state, done)
        self.memory.add(transition)

    def update(self):
        # only start when buffer big enough
        if not self.memory.full and self.memory.pos < self.batch_size:
            return

        # 1) PER sampling
        beta = 0.4  # you can ramp this to 1.0 over time
        st, ac, rw, ns, dn, idxs, wts = \
            self.memory.sample(self.batch_size, beta)

        # 2) convert to tensors (same as before), plus wts
        states_arr = np.stack(st).astype(np.float32)
        next_arr = np.stack([x if x is not None else np.zeros_like(states_arr[0])
                             for x in ns]).astype(np.float32)
        states_t = torch.from_numpy(states_arr).to(self.device)
        next_t = torch.from_numpy(next_arr).to(self.device)
        actions_t = torch.tensor(ac, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rw, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_t = torch.tensor(dn, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights_t = torch.tensor(wts, dtype=torch.float32, device=self.device).unsqueeze(1)

        # 3) Q‐value & target (Double‐DQN)
        q_pred = self.policy_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            best_a = self.policy_net(next_t).argmax(dim=1, keepdim=True)
            q_next = self.target_net(next_t).gather(1, best_a)
            q_target = rewards_t + (1 - dones_t) * self.gamma * q_next

        # 4) weighted loss
        loss = (weights_t * (q_pred - q_target).pow(2)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 5) update priorities
        td_errors = (q_pred - q_target).abs().detach().cpu().squeeze().numpy()
        new_prios = td_errors + 1e-6
        self.memory.update_priorities(idxs, new_prios)

        # 6) target‐net sync & epsilon decay (unchanged)
        self.learn_steps += 1
        if self.learn_steps % self.target_update_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)


#优先经验回放
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha    = alpha
        self.pos      = 0
        self.full     = False
        self.buffer   = [None] * capacity
        self.prios    = np.zeros((capacity,), dtype=np.float32)

    def add(self, transition, priority=None):
        max_prio = self.prios.max() if self.full else (self.prios[:self.pos].max() if self.pos>0 else 1.0)
        self.buffer[self.pos] = transition
        self.prios[self.pos]  = priority if priority is not None else max_prio
        self.pos += 1
        if self.pos >= self.capacity:
            self.pos  = 0
            self.full = True

    def sample(self, batch_size, beta=0.4):
        total = self.capacity if self.full else self.pos
        prios = self.prios[:total]
        probs = prios ** self.alpha
        P     = probs / probs.sum()
        indices= np.random.choice(total, batch_size, p=P)
        samples= [self.buffer[i] for i in indices]
        weights= (total * P[indices]) ** (-beta)
        weights/= weights.max()
        states, actions, rewards, next_states, dones = zip(*samples)
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, pr in zip(indices, priorities):
            self.prios[idx] = pr



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

        triplet = (m_idx, p_idx, 0)
        if triplet in env.valid_actions:
            action = env.valid_actions.index(triplet)
            actions.append(action)
        else:
            continue

    return actions


# 4) Training loop with imitation
def train_dqn(
    num_episodes=5000,
    pretrain_steps=10000,
    save_every=50,
    model_path='dqn_real'
):
    env = ProductionSchedulingEnv()
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n
    reward_history = []

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=5e-5,
        gamma=0.99,
        buffer_size=100_000,
        batch_size=128,
        eps_start=1.0,
        eps_decay=0.9997,
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

        reward_history.append(total_reward)

        print(f"Episode {ep}/{num_episodes}  Reward: {total_reward:.2f}  Epsilon: {agent.epsilon:.3f}")
        if ep % save_every == 0:
            torch.save(agent.policy_net.state_dict(), f"{model_path}_ep{ep}.pth")

    # final save
    plt.figure(figsize=(8, 4))
    plt.plot(reward_history)
    ma = np.convolve(reward_history, np.ones(50) / 50, mode='valid')
    plt.plot(range(50, len(reward_history) + 1), ma, color='red', label='50-ep MA')
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Curve eps5000')
    plt.grid(True)
    plt.show()
    torch.save(agent.policy_net.state_dict(), f"{model_path}_final.pth")



    print("Training complete, models saved.")


if __name__ == '__main__':
    train_dqn()
