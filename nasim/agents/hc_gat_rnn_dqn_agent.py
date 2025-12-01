"""
Hippocampus-Inspired GAT-RNN-DQN Agent for NASim
----------------------------------------
DG(GAT+TopK) → CA3(RNN attractor) → CA1(DQN fusion)
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


# ============================================================
# Dentate Gyrus (DG) — 图模式分离：GAT + Top-K 节点稀疏编码
# ============================================================
class GATEncoder(nn.Module):
    def __init__(self, input_dim, gat_hidden=64, out_dim=64,
                 gat_heads=4, dropout=0.1, topk=5):
        super().__init__()
        self.gat1 = GATConv(input_dim, gat_hidden, heads=gat_heads,
                            concat=True, dropout=dropout)
        self.gat2 = GATConv(gat_hidden * gat_heads, out_dim,
                            heads=1, concat=False, dropout=dropout)

        # small MLP for importance score
        self.att_mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Linear(out_dim // 2, 1)
        )

        self.topk = topk

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))   # [N, out_dim]

        # global embedding (图整体布局)
        global_emb = x.mean(dim=0)

        # DG: 稀疏 Top-K 节点
        att_logits = self.att_mlp(x).squeeze(-1)       # [N]
        att_scores = torch.softmax(att_logits, dim=0)

        N = x.size(0)
        k = min(self.topk, N)
        _, idx = torch.topk(att_scores, k, dim=0)

        topk_nodes = x[idx]                            # [k, out_dim]

        # ---- 关键：如果图节点少于 topk，补零 ----
        if k < self.topk:
            pad = torch.zeros(self.topk - k, topk_nodes.size(1),
                            device=topk_nodes.device)
            topk_nodes = torch.cat([topk_nodes, pad], dim=0)   # [topk, out_dim]

        # 最终 flat 输出永远是 topk*out_dim
        topk_flat = topk_nodes.reshape(-1)

        return topk_flat, global_emb



# ============================================================
# CA1 (DQN)：动作价值网络
# 输入将是 concat(CA3_out, global_emb)
# ============================================================
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


# ============================================================
# Replay Memory — 保存 DG seq + global_emb
# ============================================================
class ReplayMemory:
    def __init__(self, capacity, seq_len, feat_dim, global_dim, device="cpu"):
        self.capacity = capacity
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.global_dim = global_dim
        self.device = device
        self.ptr = 0
        self.size = 0

        # DG 序列
        self.s_buf = np.zeros((capacity, seq_len, feat_dim), dtype=np.float32)
        self.next_s_buf = np.zeros((capacity, seq_len, feat_dim), dtype=np.float32)

        # global embedding
        self.g_buf = np.zeros((capacity, global_dim), dtype=np.float32)
        self.next_g_buf = np.zeros((capacity, global_dim), dtype=np.float32)

        # 通用字段
        self.a_buf = np.zeros((capacity, 1), dtype=np.int64)
        self.r_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)

    def store(self, seq_s, g, a, seq_next_s, g2, r, done):
        self.s_buf[self.ptr] = seq_s
        self.g_buf[self.ptr] = g
        self.a_buf[self.ptr] = a
        self.next_s_buf[self.ptr] = seq_next_s
        self.next_g_buf[self.ptr] = g2
        self.r_buf[self.ptr] = r
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, batch_size)

        s = torch.tensor(self.s_buf[idxs], device=self.device)
        next_s = torch.tensor(self.next_s_buf[idxs], device=self.device)
        g = torch.tensor(self.g_buf[idxs], device=self.device)
        next_g = torch.tensor(self.next_g_buf[idxs], device=self.device)
        a = torch.tensor(self.a_buf[idxs], device=self.device)
        r = torch.tensor(self.r_buf[idxs], device=self.device)
        d = torch.tensor(self.done_buf[idxs], device=self.device)

        return s, g, a, next_s, next_g, r, d


# ============================================================
# GAT-DG + CA3(RNN) + CA1(DQN) Agent
# ============================================================
class HippocampusAgent:
    def __init__(self, env,
                 lr=1e-3,
                 gamma=0.97,
                 batch_size=32,
                 replay_size=10000,
                 hidden_sizes=[256, 256],
                 gat_hidden=128,
                 gat_out_dim=64,
                 topk=5,
                 rnn_hidden_dim=32,
                 rnn_num_layers=1,
                 rnn_type="GRU",
                 gat_heads=4,
                 gat_dropout=0.3,
                 rnn_dropout=0.1,
                 seq_len=4,
                 final_epsilon=0.05,
                 target_update_freq=500,
                 training_steps=20000,
                 seed=0,
                 device=None,
                 verbose=True):

        assert env.flat_actions
        self.env = env
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        if verbose:
            print(f"[INFO] Hippocampus Agent running on {self.device}")

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.num_actions = env.action_space.n
        self.gamma = gamma
        self.batch_size = batch_size
        self.training_steps = training_steps
        self.steps_done = 0
        self.seq_len = seq_len
        self.rnn_hidden_dim = rnn_hidden_dim

        # ============ 获取特征维度 ============
        obs, _ = env.reset()
        obs_array = np.array(obs)
        num_nodes = len(env.network.hosts) + 1
        feature_dim = int(obs_array.size / num_nodes)
        obs_array = obs_array.reshape(num_nodes, feature_dim)
        sample_graph = obs_to_graph(obs_array, env.network)
        input_dim = sample_graph.x.shape[1]

        # ============ DG (GAT + Top-K) ============
        self.encoder = GATEncoder(input_dim,
                                  gat_hidden,
                                  gat_out_dim,
                                  gat_heads,
                                  gat_dropout,
                                  topk).to(self.device)

        self.dg_feat_dim = topk * gat_out_dim
        self.global_dim = gat_out_dim

        # ============ CA3 (RNN with attractor) ============
        if rnn_type.upper() == "LSTM":
            self.rnn = nn.LSTM(self.dg_feat_dim, rnn_hidden_dim,
                               rnn_num_layers, batch_first=True,
                               dropout=rnn_dropout)
        else:
            self.rnn = nn.GRU(self.dg_feat_dim, rnn_hidden_dim,
                              rnn_num_layers, batch_first=True,
                              dropout=rnn_dropout)
        self.rnn.to(self.device)

        # ============ CA1 (DQN Fusion) ============
        dqn_input_dim = rnn_hidden_dim + self.global_dim
        self.policy_net = DQN(dqn_input_dim, hidden_sizes, self.num_actions).to(self.device)
        self.target_net = DQN(dqn_input_dim, hidden_sizes, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # ============ Optimizer ============
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.rnn.parameters()) +
            list(self.policy_net.parameters()), lr=lr
        )

        self.loss_fn = nn.SmoothL1Loss()

        # ============ Replay Memory ============
        self.replay = ReplayMemory(replay_size,
                                   seq_len,
                                   self.dg_feat_dim,
                                   self.global_dim,
                                   self.device)

        # ============ Epsilon Schedule ============
        self.final_epsilon = final_epsilon
        self.epsilon_schedule = np.linspace(1.0, final_epsilon, training_steps)

        self.logger = SummaryWriter()
        self.seq_buffer = []

    # ----------------------------------------------------
    # Utilities
    # ----------------------------------------------------
    def get_epsilon(self):
        return self.epsilon_schedule[min(self.steps_done, len(self.epsilon_schedule)-1)]

    def encode_obs(self, obs):
        obs_array = np.array(obs)
        num_nodes = len(self.env.network.hosts) + 1
        feat = int(obs_array.size / num_nodes)
        obs_array = obs_array.reshape(num_nodes, feat)

        graph = obs_to_graph(obs_array, self.env.network).to(self.device)
        topk_flat, global_emb = self.encoder(graph)
        return topk_flat, global_emb

    def get_action(self, fusion, eps):
        with torch.no_grad():
            q_vals = self.policy_net(fusion)

        if random.random() > eps:
            return q_vals.argmax(dim=1).item()
        else:
            return random.randint(0, self.num_actions - 1)

    # ----------------------------------------------------
    # Optimize
    # ----------------------------------------------------
    def optimize(self):
        if self.replay.size < self.batch_size:
            return 0, 0

        s, g, a, next_s, next_g, r, d = self.replay.sample_batch(self.batch_size)

        # CA3 前向
        rnn_out, _ = self.rnn(s)
        rnn_last = rnn_out[:, -1, :]

        # CA1 融合
        fusion = torch.cat([rnn_last, g], dim=1)
        q_vals = self.policy_net(fusion).gather(1, a).squeeze()

        # target CA3
        with torch.no_grad():
            rnn_out2, _ = self.rnn(next_s)
            rnn_last2 = rnn_out2[:, -1, :]
            fusion2 = torch.cat([rnn_last2, next_g], dim=1)
            target_q = r + self.gamma * (1 - d) * self.target_net(fusion2).max(dim=1)[0]

        loss = self.loss_fn(q_vals, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        self.optimizer.step()

        # soft update
        for t, s in zip(self.target_net.parameters(), self.policy_net.parameters()):
            t.data.copy_(0.995 * t.data + 0.005 * s.data)

        return loss.item(), q_vals.mean().item()

    # ----------------------------------------------------
    # Train 1 episode
    # ----------------------------------------------------
    def run_train_episode(self, step_limit):
        obs, _ = self.env.reset()
        self.seq_buffer = []
        hidden = None

        done = False
        total_reward = 0
        steps = 0

        while not done and steps < step_limit:
            eps = self.get_epsilon()
            topk, g = self.encode_obs(obs)
            topk = topk.detach().cpu().numpy()
            g_np = g.detach().cpu().numpy()

            # build seq
            self.seq_buffer.append(topk)
            if len(self.seq_buffer) > self.seq_len:
                self.seq_buffer.pop(0)

            seq_input = np.array(self.seq_buffer)
            if len(seq_input) < self.seq_len:
                pad = np.zeros((self.seq_len - len(seq_input), len(topk)))
                seq_input = np.vstack([pad, seq_input])

            seq_tensor = torch.tensor(seq_input, dtype=torch.float32,
                                      device=self.device).unsqueeze(0)

            # CA3 前向（包含吸引子残差）
            rnn_out, hidden = self.rnn(seq_tensor, hidden)
            rnn_last = rnn_out[:, -1, :]

            # CA3 吸引子：加上 previous hidden
            if hidden is not None:
                if isinstance(hidden, tuple):
                    h_prev = hidden[0][-1]
                else:
                    h_prev = hidden[-1]
                rnn_last = rnn_last + h_prev
            rnn_last = F.layer_norm(rnn_last, rnn_last.shape)

            # CA1 融合
            fusion = torch.cat([rnn_last, g.unsqueeze(0)], dim=1)

            action = self.get_action(fusion, eps)

            next_obs, reward, done, _, _ = self.env.step(action)

            # build next seq & next g
            next_topk, next_g = self.encode_obs(next_obs)
            next_topk = next_topk.detach().cpu().numpy()
            next_g_np = next_g.detach().cpu().numpy()

            next_seq_buf = self.seq_buffer.copy()
            next_seq_buf.append(next_topk)
            if len(next_seq_buf) > self.seq_len:
                next_seq_buf.pop(0)

            next_seq_input = np.array(next_seq_buf)
            if len(next_seq_input) < self.seq_len:
                pad = np.zeros((self.seq_len - len(next_seq_input), len(topk)))
                next_seq_input = np.vstack([pad, next_seq_input])

            # store in replay
            self.replay.store(seq_input,
                              g_np,
                              [action],
                              next_seq_input,
                              next_g_np,
                              float(reward),
                              float(done))

            self.optimize()

            total_reward += reward
            obs = next_obs
            steps += 1
            self.steps_done += 1

        return total_reward, steps, self.env.goal_reached()

    # ----------------------------------------------------
    # Main training
    # ----------------------------------------------------
    def train(self):
        num_episodes = 0
        while self.steps_done < self.training_steps:
            ep_ret, ep_steps, goal = self.run_train_episode(self.training_steps - self.steps_done)
            loss, q_mean = self.optimize()
            num_episodes += 1

            self.logger.add_scalar("episode", num_episodes, self.steps_done)
            self.logger.add_scalar("epsilon", self.get_epsilon(), self.steps_done)
            self.logger.add_scalar("episode_return", ep_ret, self.steps_done)
            self.logger.add_scalar("episode_steps", ep_steps, self.steps_done)
            self.logger.add_scalar("episode_goal_reached", int(goal), self.steps_done)
            
            self.logger.add_scalar("loss", loss, self.steps_done)
            self.logger.add_scalar("q_value_mean", q_mean, self.steps_done)

            if num_episodes % 10 == 0:
                print(f"Episode {num_episodes}: steps={self.steps_done}, return={ep_ret}, goal={goal}")
        self.logger.close()

    # ----------------------------------------------------
    # Eval
    # ----------------------------------------------------
    def run_eval_episode(self, env=None, render=False):
        if env is None:
            env = self.env
        obs, _ = env.reset()

        self.seq_buffer = []
        hidden = None

        done = False
        total_reward = 0
        steps = 0

        while not done:
            topk, g = self.encode_obs(obs)
            topk = topk.detach().cpu().numpy()

            self.seq_buffer.append(topk)
            if len(self.seq_buffer) > self.seq_len:
                self.seq_buffer.pop(0)

            seq_input = np.array(self.seq_buffer)
            if len(seq_input) < self.seq_len:
                pad = np.zeros((self.seq_len - len(seq_input), len(topk)))
                seq_input = np.vstack([pad, seq_input])

            seq_tensor = torch.tensor(seq_input, dtype=torch.float32,
                                      device=self.device).unsqueeze(0)
            rnn_out, hidden = self.rnn(seq_tensor, hidden)
            rnn_last = rnn_out[:, -1, :]

            if hidden is not None:
                if isinstance(hidden, tuple):
                    h_prev = hidden[0][-1]
                else:
                    h_prev = hidden[-1]
                rnn_last = rnn_last + h_prev

            rnn_last = F.layer_norm(rnn_last, rnn_last.shape)
            fusion = torch.cat([rnn_last, g.unsqueeze(0)], dim=1)

            with torch.no_grad():
                q = self.policy_net(fusion)
                action = q.argmax(dim=1).item()

            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1

            if render:
                env.render()

        return total_reward, steps, env.goal_reached()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="NASim benchmark")
    parser.add_argument("--training_steps", type=int, default=20000)
    parser.add_argument("--render_eval", action="store_true")
    args = parser.parse_args()

    env = nasim.make_benchmark(args.env_name, seed=0,
                               fully_obs=False, flat_actions=True, flat_obs=False)
    agent = HippocampusAgent(env, training_steps=args.training_steps)
    agent.train()
    
