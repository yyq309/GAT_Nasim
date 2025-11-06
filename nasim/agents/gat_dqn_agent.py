"""
GAT-DQN Agent for NASim
-----------------------

基于图神经网络 (GAT) 的 DQN 智能体：
在标准 DQN 的基础上引入 GAT 编码器，将 NASim 环境的图结构输入转化为图嵌入。
"""

import random
import numpy as np
import nasim
from gymnasium import error

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data
    from torch.utils.tensorboard import SummaryWriter
except ImportError as e:
    raise error.DependencyNotInstalled(
        f"{e}. 请安装所需依赖: pip install torch torch-geometric torchvision torchaudio"
    )

from nasim.envs.utils import obs_to_graph


# ======================================
# GAT Encoder
# ======================================
class GATEncoder(nn.Module):
    """将图对象编码为固定维度向量"""
    def __init__(self, input_dim, gat_hidden=64, out_dim=64, gat_heads=4, dropout=0.0):
        super().__init__()
        self.gat1 = GATConv(input_dim, gat_hidden, heads=gat_heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(gat_hidden * gat_heads, out_dim, heads=1, concat=False, dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))
        g = x.mean(dim=0, keepdim=True)  # 全图平均池化
        return g.squeeze(0)


# ======================================
# DQN 网络
# ======================================
class DQN(nn.Module):
    def __init__(self, input_dim, hidden_sizes, num_actions):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, num_actions))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def get_action(self, x):
        with torch.no_grad():
            if len(x.shape) == 1:
                x = x.view(1, -1)
            return self.forward(x).max(1)[1]


# ======================================
# Replay Memory
# ======================================
class ReplayMemory:
    def __init__(self, capacity, s_dim, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.s_buf = np.zeros((capacity, s_dim), dtype=np.float32)
        self.a_buf = np.zeros((capacity, 1), dtype=np.int64)
        self.next_s_buf = np.zeros((capacity, s_dim), dtype=np.float32)
        self.r_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)

    def store(self, s, a, next_s, r, done):
        self.s_buf[self.ptr] = s
        self.a_buf[self.ptr] = a
        self.next_s_buf[self.ptr] = next_s
        self.r_buf[self.ptr] = r
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, batch_size):
        idxs = np.random.choice(self.size, batch_size)
        batch = [self.s_buf[idxs], self.a_buf[idxs], self.next_s_buf[idxs], self.r_buf[idxs], self.done_buf[idxs]]
        return [torch.tensor(x, device=self.device, dtype=torch.float32)
                if x.ndim > 1 else torch.tensor(x, device=self.device)
                for x in batch]


# ======================================
# GAT-DQN Agent
# ======================================
class GATDQNAgent:
    def __init__(self, env,
                 lr=0.001269625695139964,
                 gamma=0.9729296464147921,
                 batch_size=64,
                 replay_size=10000,
                 hidden_sizes=[128, 128],
                 gat_hidden=64,
                 gat_out_dim=64,
                 gat_heads=5,
                 dropout=0.27478135282201754,
                 exploration_steps=10000,
                 final_epsilon=0.0477595747434069,
                 target_update_freq=1000,
                 training_steps=20000,
                 seed=0,
                 verbose=True):

        assert env.flat_actions
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        if self.verbose:
            print(f"\n[INFO] GAT-DQN on device={self.device}")

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.num_actions = env.action_space.n
        self.gamma = gamma
        self.batch_size = batch_size
        self.training_steps = training_steps
        self.target_update_freq = target_update_freq
        self.steps_done = 0

        # sample graph to get input_dim
        obs, _ = env.reset()
        obs_array = np.array(obs)
        num_nodes = len(env.network.hosts) + 1
        feature_dim = int(obs_array.size / num_nodes)
        if obs_array.ndim == 1:
            obs_array = obs_array.reshape(num_nodes, feature_dim)

        sample_graph = obs_to_graph(obs_array, env.network)
        input_dim = sample_graph.x.shape[1]

        # GAT encoder
        self.encoder = GATEncoder(input_dim=input_dim,
                                  gat_hidden=gat_hidden,
                                  out_dim=gat_out_dim,
                                  gat_heads=gat_heads,
                                  dropout=dropout).to(self.device)

        # DQN networks
        self.policy_net = DQN(gat_out_dim, hidden_sizes, self.num_actions).to(self.device)
        self.target_net = DQN(gat_out_dim, hidden_sizes, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.policy_net.parameters()), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        # replay buffer
        self.replay = ReplayMemory(replay_size, gat_out_dim, self.device)

        # epsilon schedule
        self.final_epsilon = final_epsilon
        self.epsilon_schedule = np.linspace(1.0, final_epsilon, exploration_steps)

        # tensorboard logger
        self.logger = SummaryWriter()

    def get_epsilon(self):
        idx = min(self.steps_done, len(self.epsilon_schedule) - 1)
        return self.epsilon_schedule[idx]

    def get_egreedy_action(self, obs, epsilon):
        obs_array = np.array(obs)
        num_nodes = len(self.env.network.hosts) + 1
        feature_dim = int(obs_array.size / num_nodes)
        if obs_array.ndim == 1:
            obs_array = obs_array.reshape(num_nodes, feature_dim)
        graph = obs_to_graph(obs_array, self.env.network).to(self.device)
        with torch.no_grad():
            emb = self.encoder(graph).unsqueeze(0)
            q_values = self.policy_net(emb)
        if random.random() > epsilon:
            return q_values.argmax(dim=1).item()
        return random.randint(0, self.num_actions - 1)

    def optimize(self):
        if self.replay.size < self.batch_size:
            return 0, 0
        s, a, next_s, r, d = self.replay.sample_batch(self.batch_size)
        q_vals = self.policy_net(s).gather(1, a.long()).squeeze()
        with torch.no_grad():
            target_q = r + self.gamma * (1 - d) * self.target_net(next_s).max(1)[0]
        loss = self.loss_fn(q_vals, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item(), q_vals.mean().item()

    def run_train_episode(self, step_limit):
        obs, _ = self.env.reset()
        done, total_reward, steps = False, 0, 0
        while not done and steps < step_limit:
            epsilon = self.get_epsilon()
            action = self.get_egreedy_action(obs, epsilon)
            next_obs, reward, done, limit, _ = self.env.step(action)

            # encode states
            obs_array = np.array(obs)
            next_obs_array = np.array(next_obs)
            num_nodes = len(self.env.network.hosts) + 1
            feature_dim = int(obs_array.size / num_nodes)
            if obs_array.ndim == 1:
                obs_array = obs_array.reshape(num_nodes, feature_dim)
            if next_obs_array.ndim == 1:
                next_obs_array = next_obs_array.reshape(num_nodes, -1)

            s_vec = self.encoder(obs_to_graph(obs_array, self.env.network).to(self.device)).detach().cpu().numpy()
            next_vec = self.encoder(obs_to_graph(next_obs_array, self.env.network).to(self.device)).detach().cpu().numpy()

            self.replay.store(s_vec, action, next_vec, reward, float(done))
            self.optimize()

            total_reward += reward
            obs = next_obs
            steps += 1
            self.steps_done += 1
        return total_reward, steps, self.env.goal_reached()

    def train(self):
        num_episodes = 0
        while self.steps_done < self.training_steps:
            ep_return, ep_steps, goal = self.run_train_episode(self.training_steps - self.steps_done)
            num_episodes += 1

            self.logger.add_scalar("episode", num_episodes, self.steps_done)
            self.logger.add_scalar("epsilon", self.get_epsilon(), self.steps_done)
            self.logger.add_scalar("episode_return", ep_return, self.steps_done)
            self.logger.add_scalar("episode_steps", ep_steps, self.steps_done)
            self.logger.add_scalar("episode_goal_reached", int(goal), self.steps_done)

            if num_episodes % 10 == 0 and self.verbose:
                print(f"Episode {num_episodes}: steps={self.steps_done}, return={ep_return}, goal={goal}")

        self.logger.close()

    def run_eval_episode(self, env=None, render=False, eval_epsilon=0.05, render_mode="human"):
        if env is None:
            env = self.env
        obs, _ = env.reset()
        done, steps, ep_ret = False, 0, 0
        while not done:
            a = self.get_egreedy_action(obs, eval_epsilon)
            obs, r, done, limit, _ = env.step(a)
            ep_ret += r
            steps += 1
            if render:
                env.render()
        return ep_ret, steps, env.goal_reached()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="benchmark scenario name")
    parser.add_argument("-t", "--training_steps", type=int, default=20000)
    parser.add_argument("--render_eval", action="store_true")
    args = parser.parse_args()

    env = nasim.make_benchmark(args.env_name, seed=0, fully_obs=True, flat_actions=True, flat_obs=True)
    agent = GATDQNAgent(env, training_steps=args.training_steps)
    agent.train()
    agent.run_eval_episode(render=args.render_eval)
