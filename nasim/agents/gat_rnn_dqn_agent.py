"""
GAT-RNN-DQN Agent for NASim
---------------------------
在 GAT-DQN 基础上加入 GRU 模块，对时间序列的图嵌入进行递归建模。
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
    from torch.utils.tensorboard import SummaryWriter
except ImportError as e:
    raise error.DependencyNotInstalled(
        f"{e}. 请安装依赖: pip install torch torch-geometric torchvision torchaudio"
    )

from nasim.envs.utils import obs_to_graph


# ======================================
# GAT Encoder
# ======================================
class GATEncoder(nn.Module):
    """图结构编码器"""
    def __init__(self, input_dim, gat_hidden=64, out_dim=64, gat_heads=4, dropout=0.1):
        super().__init__()
        self.gat1 = GATConv(input_dim, gat_hidden, heads=gat_heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(gat_hidden * gat_heads, out_dim, heads=1, concat=False, dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))
        g = x.mean(dim=0, keepdim=True)  # 图平均池化
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
        s = torch.tensor(self.s_buf[idxs], device=self.device)
        a = torch.tensor(self.a_buf[idxs], device=self.device, dtype=torch.long)
        next_s = torch.tensor(self.next_s_buf[idxs], device=self.device)
        r = torch.tensor(self.r_buf[idxs], device=self.device)
        d = torch.tensor(self.done_buf[idxs], device=self.device)
        return s, a, next_s, r, d


# ======================================
# GAT-RNN-DQN Agent
# ======================================
class GATRNNDQNAgent:
    def __init__(self, env,
                 lr=1e-3,
                 gamma=0.97,
                 batch_size=54,
                 replay_size=10000,
                 hidden_sizes=[128, 128],
                 gat_hidden=97,
                 gat_out_dim=64,
                 rnn_hidden_dim=112,
                 rnn_num_layers=1,
                 rnn_type="LSTM",
                 gat_heads=4,
                 gat_dropout=0.3130828152682149,
                 rnn_dropout=0.08771617264038689,
                 exploration_steps=10000,
                 final_epsilon=0.06961374261255864,
                 target_update_freq=818,
                 training_steps=20000,
                 seed=0,
                 device=None,
                 verbose=True):

        assert env.flat_actions
        self.env = env
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        if verbose:
            print(f"[INFO] GAT-RNN-DQN running on {self.device}")

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.num_actions = env.action_space.n
        self.gamma = gamma
        self.batch_size = batch_size
        self.training_steps = training_steps
        self.target_update_freq = target_update_freq
        self.steps_done = 0

        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers

        # 获取输入维度
        obs, _ = env.reset()
        obs_array = np.array(obs)
        num_nodes = len(env.network.hosts) + 1
        feature_dim = int(obs_array.size / num_nodes)
        obs_array = obs_array.reshape(num_nodes, feature_dim)
        sample_graph = obs_to_graph(obs_array, env.network)
        input_dim = sample_graph.x.shape[1]

        # GAT 编码器
        self.encoder = GATEncoder(input_dim, gat_hidden, gat_out_dim, gat_heads, gat_dropout).to(self.device)

        # RNN 模块
        if rnn_type.upper() == "LSTM":
            self.rnn = nn.LSTM(gat_out_dim, rnn_hidden_dim, rnn_num_layers, batch_first=True, dropout=rnn_dropout)
        else:
            self.rnn = nn.GRU(gat_out_dim, rnn_hidden_dim, rnn_num_layers, batch_first=True, dropout=rnn_dropout)
        self.rnn.to(self.device)

        # DQN 网络
        self.policy_net = DQN(rnn_hidden_dim, hidden_sizes, self.num_actions).to(self.device)
        self.target_net = DQN(rnn_hidden_dim, hidden_sizes, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # 优化器与损失函数
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.rnn.parameters()) + list(self.policy_net.parameters()), lr=lr
        )
        self.loss_fn = nn.SmoothL1Loss()

        self.replay = ReplayMemory(replay_size, rnn_hidden_dim, self.device)
        self.final_epsilon = final_epsilon
        self.epsilon_schedule = np.linspace(1.0, final_epsilon, exploration_steps)
        self.logger = SummaryWriter()

    def get_epsilon(self):
        idx = min(self.steps_done, len(self.epsilon_schedule) - 1)
        return self.epsilon_schedule[idx]

    def encode_obs(self, obs, hidden):
        """GAT + RNN编码，确保 hidden 为 3D"""
        obs_array = np.array(obs)
        num_nodes = len(self.env.network.hosts) + 1
        feature_dim = int(obs_array.size / num_nodes)
        obs_array = obs_array.reshape(num_nodes, feature_dim)

        graph = obs_to_graph(obs_array, self.env.network).to(self.device)
        gat_emb = self.encoder(graph).unsqueeze(0).unsqueeze(0)  # [batch=1, seq=1, feat]
        
        # 确保 hidden 是 3D
        if isinstance(self.rnn, nn.LSTM):
            h, c = hidden
            if h.dim() == 2:
                h = h.unsqueeze(0)
            if c.dim() == 2:
                c = c.unsqueeze(0)
            rnn_out, hidden = self.rnn(gat_emb, (h, c))
        else:
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)
            rnn_out, hidden = self.rnn(gat_emb, hidden)

        return rnn_out.squeeze(0).squeeze(0), hidden

    def get_egreedy_action(self, obs, epsilon, hidden):
        state_vec, hidden = self.encode_obs(obs, hidden)
        with torch.no_grad():
            q_values = self.policy_net(state_vec.unsqueeze(0))
        if random.random() > epsilon:
            return q_values.argmax(dim=1).item(), hidden
        return random.randint(0, self.num_actions - 1), hidden

    def optimize(self):
        if self.replay.size < self.batch_size:
            return 0, 0
        s, a, next_s, r, d = self.replay.sample_batch(self.batch_size)
        q_vals = self.policy_net(s).gather(1, a).squeeze()
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
        # 初始化 RNN 隐状态
        if isinstance(self.rnn, nn.LSTM):
            hidden = (
                torch.zeros(self.rnn_num_layers, 1, self.rnn_hidden_dim, device=self.device),
                torch.zeros(self.rnn_num_layers, 1, self.rnn_hidden_dim, device=self.device),
            )
        else:
            hidden = torch.zeros(self.rnn_num_layers, 1, self.rnn_hidden_dim, device=self.device)

        done, total_reward, steps = False, 0, 0
        while not done and steps < step_limit:
            epsilon = self.get_epsilon()
            action, hidden = self.get_egreedy_action(obs, epsilon, hidden)
            next_obs, reward, done, _, _ = self.env.step(action)
            s_vec, _ = self.encode_obs(obs, hidden)
            next_vec, _ = self.encode_obs(next_obs, hidden)

            # detach 避免梯度追踪
            self.replay.store(
                s_vec.detach().cpu().numpy(),
                [action],
                next_vec.detach().cpu().numpy(),
                float(reward),
                float(done)
            )

            self.optimize()
            total_reward += reward
            obs = next_obs
            steps += 1
            self.steps_done += 1
        return total_reward, steps, self.env.goal_reached()

    def train(self):
        num_episodes = 0
        while self.steps_done < self.training_steps:
            ep_ret, ep_steps, goal = self.run_train_episode(self.training_steps - self.steps_done)
            num_episodes += 1
            
            self.logger.add_scalar("episode", num_episodes, self.steps_done)
            self.logger.add_scalar("epsilon", self.get_epsilon(), self.steps_done)
            self.logger.add_scalar("episode_return", ep_ret, self.steps_done)
            self.logger.add_scalar("episode_steps", ep_steps, self.steps_done)
            self.logger.add_scalar("episode_goal_reached", int(goal), self.steps_done)
            
            if num_episodes % 10 == 0:
                print(f"Episode {num_episodes}: steps={self.steps_done}, return={ep_ret}, goal={goal}")
        self.logger.close()

    def run_eval_episode(self, env=None, render=False, eval_epsilon=0.05):
        if env is None:
            env = self.env
        obs, _ = env.reset()
        done, steps, ep_ret = False, 0, 0

        # 初始化 RNN 隐状态
        if isinstance(self.rnn, nn.LSTM):
            h_rnn = (
                torch.zeros(self.rnn_num_layers, 1, self.rnn_hidden_dim, device=self.device),
                torch.zeros(self.rnn_num_layers, 1, self.rnn_hidden_dim, device=self.device),
            )
        else:
            h_rnn = torch.zeros(self.rnn_num_layers, 1, self.rnn_hidden_dim, device=self.device)

        while not done:
            obs_array = np.array(obs)
            num_nodes = len(env.network.hosts) + 1
            feature_dim = int(obs_array.size / num_nodes)
            obs_array = obs_array.reshape(num_nodes, feature_dim)
            graph = obs_to_graph(obs_array, env.network).to(self.device)

            with torch.no_grad():
                g_emb = self.encoder(graph).unsqueeze(0).unsqueeze(0)
                if isinstance(self.rnn, nn.LSTM):
                    rnn_out, h_rnn = self.rnn(g_emb, h_rnn)
                else:
                    rnn_out, h_rnn = self.rnn(g_emb, h_rnn)

                q_values = self.policy_net(rnn_out.squeeze(0))
                if random.random() > eval_epsilon:
                    a = q_values.argmax(dim=1).item()
                else:
                    a = random.randint(0, self.num_actions - 1)

            obs, r, done, _, _ = env.step(a)
            ep_ret += r
            steps += 1
            if render:
                env.render()
        return ep_ret, steps, env.goal_reached()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="NASim benchmark scenario name")
    parser.add_argument("--training_steps", type=int, default=20000)
    parser.add_argument("--render_eval", action="store_true")
    args = parser.parse_args()

    env = nasim.make_benchmark(args.env_name, seed=0, fully_obs=True, flat_actions=True, flat_obs=True)
    agent = GATRNNDQNAgent(env, training_steps=args.training_steps)
    agent.train()
    agent.run_eval_episode(render=args.render_eval)
