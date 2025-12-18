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


# ===========================
# GAT Encoder
# ===========================
class GATEncoder(nn.Module):
    def __init__(self, input_dim, gat_hidden=64, out_dim=64, gat_heads=4, dropout=0.1):
        super().__init__()
        self.gat1 = GATConv(input_dim, gat_hidden, heads=gat_heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(gat_hidden * gat_heads, out_dim, heads=1, concat=False, dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))
        g = x.mean(dim=0, keepdim=True)
        return g.squeeze(0)


# ===========================
# DQN 网络
# ===========================
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


# ===========================
# Replay Memory (序列化)
# ===========================
class ReplayMemory:
    def __init__(self, capacity, seq_len, feat_dim, device="cpu"):
        self.capacity = capacity
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.device = device
        self.ptr = 0
        self.size = 0

        self.s_buf = np.zeros((capacity, seq_len, feat_dim), dtype=np.float32)
        self.a_buf = np.zeros((capacity, 1), dtype=np.int64)
        self.next_s_buf = np.zeros((capacity, seq_len, feat_dim), dtype=np.float32)
        self.r_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)

    def store(self, seq_s, a, seq_next_s, r, done):
        self.s_buf[self.ptr] = seq_s
        self.a_buf[self.ptr] = a
        self.next_s_buf[self.ptr] = seq_next_s
        self.r_buf[self.ptr] = r
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, batch_size):
        # 修复采样索引越界问题：确保起始索引 >=0 且结束索引 < size
        idxs = np.random.randint(0, max(1, self.size - self.seq_len + 1), batch_size)
        s = torch.tensor(self.s_buf[idxs], device=self.device, dtype=torch.float32)
        a = torch.tensor(self.a_buf[idxs], device=self.device, dtype=torch.long)
        next_s = torch.tensor(self.next_s_buf[idxs], device=self.device, dtype=torch.float32)
        r = torch.tensor(self.r_buf[idxs], device=self.device, dtype=torch.float32)
        d = torch.tensor(self.done_buf[idxs], device=self.device, dtype=torch.float32)
        return s, a, next_s, r, d


# ===========================
# GAT-RNN-DQN 智能体
# ===========================
class GATRNNDQNAgent:
    def __init__(self, env,
                 lr=1e-3,
                 gamma=0.93,
                 batch_size=32,
                 replay_size=50000,
                 hidden_sizes=[256, 256],
                 gat_hidden=128,
                 gat_out_dim=64,
                 rnn_hidden_dim=32,
                 rnn_num_layers=2,
                 rnn_type="LSTM",
                 gat_heads=4,
                 gat_dropout=0.3,
                 rnn_dropout=0.05522311878617756, 
                 seq_len=6,
                 final_epsilon=0.05,
                 target_update_freq=500,
                 training_steps=20000,
                 seed=None,
                 device=None,
                 success_bonus = 80,        
                 step_penalty = -0.1,
                 info_gain_bonus = 0.5,
                 verbose=True):

        assert env.flat_actions
        self.env = env
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        if verbose:
            print(f"[INFO] GAT-RNN-DQN running on {self.device}")

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

        self.num_actions = env.action_space.n
        self.gamma = gamma
        self.batch_size = batch_size
        self.training_steps = training_steps
        self.target_update_freq = target_update_freq
        self.steps_done = 0
        self.seq_len = seq_len
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        
        self.success_bonus = success_bonus
        self.step_penalty = step_penalty
        self.info_gain_bonus = info_gain_bonus

        # ========== 关键修改1：定义 obs_dim 和 gat_out_dim 实例属性 ==========
        self.obs_dim = env.observation_space.shape
        self.gat_out_dim = gat_out_dim

        # 获取GAT输入维度
        obs, _ = env.reset()
        obs_array = np.array(obs)
        num_nodes = len(env.network.hosts) + 1
        feature_dim = int(obs_array.size / num_nodes)
        obs_array = obs_array.reshape(num_nodes, feature_dim)
        sample_graph = obs_to_graph(obs_array, env.network)
        self.gat_input_dim = sample_graph.x.shape[1]

        # GAT 编码器
        self.encoder = GATEncoder(self.gat_input_dim, gat_hidden, gat_out_dim, gat_heads, gat_dropout).to(self.device)

        # RNN 模块（修复dropout警告）
        if rnn_type.upper() == "LSTM":
            self.rnn = nn.LSTM(gat_out_dim, rnn_hidden_dim, rnn_num_layers, 
                               batch_first=True, dropout=rnn_dropout if rnn_num_layers>1 else 0.0)
        else:
            self.rnn = nn.GRU(gat_out_dim, rnn_hidden_dim, rnn_num_layers, 
                              batch_first=True, dropout=rnn_dropout if rnn_num_layers>1 else 0.0)
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

        # Replay Memory
        self.replay = ReplayMemory(replay_size, seq_len, gat_out_dim, self.device)

        self.final_epsilon = final_epsilon
        self.epsilon_schedule = np.linspace(1.0, final_epsilon, training_steps)
        
        import os
        import datetime
        logdir = os.environ.get("TORCH_TB_DIR", "runs")
        run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logdir = os.path.join(logdir, run_name)
        self.logger = SummaryWriter(logdir)

        # 当前序列缓存（存储图嵌入序列）
        self.seq_buffer = []

    # ------------------------------
    # 工具函数
    # ------------------------------
    def get_epsilon(self):
        idx = min(self.steps_done, len(self.epsilon_schedule) - 1)
        return self.epsilon_schedule[idx]

    def encode_obs(self, obs):
        """将原始观测编码为GAT图嵌入"""
        obs_array = np.array(obs)
        num_nodes = len(self.env.network.hosts) + 1
        feature_dim = int(obs_array.size / num_nodes)
        obs_array = obs_array.reshape(num_nodes, feature_dim)

        graph = obs_to_graph(obs_array, self.env.network).to(self.device)
        with torch.no_grad():
            gat_emb = self.encoder(graph)  # [gat_out_dim]
        return gat_emb.detach().cpu().numpy()

    def get_egreedy_action(self, obs, epsilon):
        """修复：适配输入为obs，内部完成编码和RNN前向"""
        # 编码当前观测为图嵌入
        gat_emb = self.encode_obs(obs)
        # 更新序列缓存
        self.seq_buffer.append(gat_emb)
        if len(self.seq_buffer) > self.seq_len:
            self.seq_buffer.pop(0)
        # 序列补零
        seq_input = np.array(self.seq_buffer)
        if len(seq_input) < self.seq_len:
            pad = np.zeros((self.seq_len - len(seq_input), self.gat_out_dim), dtype=np.float32)
            seq_input = np.vstack([pad, seq_input])
        # RNN前向
        seq_tensor = torch.tensor(seq_input, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            rnn_out, _ = self.rnn(seq_tensor)
            rnn_last = rnn_out[:, -1, :]
            q_values = self.policy_net(rnn_last)
        # 贪心选择动作
        if random.random() > epsilon:
            return q_values.argmax(dim=1).item()
        return random.randint(0, self.num_actions - 1)

    # ------------------------------
    # 训练优化
    # ------------------------------
    def optimize(self):
        if self.replay.size < self.batch_size:
            return 0, 0

        s, a, next_s, r, d = self.replay.sample_batch(self.batch_size)

        # RNN 前向传播
        rnn_out, _ = self.rnn(s)  # [B, seq_len, rnn_hidden]
        rnn_last = rnn_out[:, -1, :]
        q_vals = self.policy_net(rnn_last).gather(1, a).squeeze()

        with torch.no_grad():
            rnn_next, _ = self.rnn(next_s)
            rnn_next_last = rnn_next[:, -1, :]
            target_q = r + self.gamma * (1 - d) * self.target_net(rnn_next_last).max(1)[0]

        loss = self.loss_fn(q_vals, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪（所有参数）
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.rnn.parameters()) + list(self.policy_net.parameters()), 
            max_norm=1.0
        )
        self.optimizer.step()

        # 软更新目标网络
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item(), q_vals.mean().item()

    # ------------------------------
    # 训练一个 episode
    # ------------------------------
    def run_train_episode(self, step_limit):
        obs, _ = self.env.reset()
        done = False
        total_reward = 0
        steps = 0
        self.last_obs = None   # 用于信息增益奖励
        # 重置序列缓存
        self.seq_buffer = []

        while not done and steps < step_limit:
            # epsilon-greedy 选择动作
            epsilon = self.get_epsilon()
            action = self.get_egreedy_action(obs, epsilon)

            # 环境步进
            next_obs, raw_reward, done, limit, _ = self.env.step(action)

            # ---------------------------------------------------------
            # Reward Shaping
            # ---------------------------------------------------------
            reward = self.step_penalty
            # 成功奖励
            if self.env.goal_reached():
                reward += self.success_bonus
            # 信息增益奖励
            obs_array = np.array(obs)
            next_obs_array = np.array(next_obs)
            if self.last_obs is not None and not np.array_equal(next_obs_array, self.last_obs):
                reward += self.info_gain_bonus
            self.last_obs = next_obs_array.copy()
            # 保留原始正向奖励
            if raw_reward > 0:
                reward += raw_reward

            # ---------------------------------------------------------
            # 编码观测并构建序列
            # ---------------------------------------------------------
            # 编码当前和下一观测为图嵌入
            s_emb = self.encode_obs(obs)
            next_s_emb = self.encode_obs(next_obs)
            # 构建当前序列（补零）
            self.seq_buffer.append(s_emb)
            if len(self.seq_buffer) > self.seq_len:
                self.seq_buffer.pop(0)
            seq_s = np.array(self.seq_buffer)
            if len(seq_s) < self.seq_len:
                pad = np.zeros((self.seq_len - len(seq_s), self.gat_out_dim), dtype=np.float32)
                seq_s = np.vstack([pad, seq_s])
            # 构建下一状态序列
            next_seq_buffer = self.seq_buffer[1:] + [next_s_emb]
            seq_next_s = np.array(next_seq_buffer)
            if len(seq_next_s) < self.seq_len:
                pad = np.zeros((self.seq_len - len(seq_next_s), self.gat_out_dim), dtype=np.float32)
                seq_next_s = np.vstack([pad, seq_next_s])

            # 存储经验
            self.replay.store(seq_s, action, seq_next_s, reward, float(done))

            # 优化网络
            loss, q_mean = self.optimize()

            # 更新状态
            total_reward += reward
            obs = next_obs
            steps += 1
            self.steps_done += 1

            # 终止条件（步数超限）
            if self.steps_done >= self.training_steps:
                break

        return total_reward, steps, self.env.goal_reached()

    # ------------------------------
    # 训练主循环
    # ------------------------------
    def train(self):
        num_episodes = 0
        while self.steps_done < self.training_steps:
            ep_ret, ep_steps, goal = self.run_train_episode(self.training_steps - self.steps_done)
            num_episodes += 1

            # 记录训练指标
            self.logger.add_scalar("episode", num_episodes, self.steps_done)
            self.logger.add_scalar("epsilon", self.get_epsilon(), self.steps_done)
            self.logger.add_scalar("episode_return", ep_ret, self.steps_done)
            self.logger.add_scalar("episode_steps", ep_steps, self.steps_done)
            self.logger.add_scalar("episode_goal_reached", int(goal), self.steps_done)
            
            # 优化时已返回loss和q_mean，此处可记录最后一次的结果
            loss, q_mean = self.optimize()
            self.logger.add_scalar("loss", loss, self.steps_done)
            self.logger.add_scalar("q_value_mean", q_mean, self.steps_done)

            if num_episodes % 10 == 0 and self.verbose:
                print(f"Episode {num_episodes}: steps={self.steps_done}, return={ep_ret}, goal={goal}, loss={loss}, q={q_mean}")
        
        self.logger.close()

    # ------------------------------
    # 评估
    # ------------------------------
    def run_eval_episode(self, env=None, render=False, eval_epsilon=0.05):
        if env is None:
            env = self.env
        obs, _ = env.reset()
        self.seq_buffer = []
        
        done, steps, ep_ret = False, 0, 0
        while not done and steps < 2000:  # 限制最大评估步数
            # 选择动作（评估时用低探索率）
            action = self.get_egreedy_action(obs, eval_epsilon)
            # 环境步进
            obs, r, done, _, _ = env.step(action)
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
    parser.add_argument("--seed", type=int, default=0,help="(default=0)")
    parser.add_argument("--render_eval", action="store_true")
    args = parser.parse_args()

    env = nasim.make_benchmark(args.env_name, seed=args.seed, fully_obs=True, flat_actions=True, flat_obs=True)
    agent = GATRNNDQNAgent(env, training_steps=args.training_steps)
    agent.train()
    agent.run_eval_episode(render=args.render_eval)