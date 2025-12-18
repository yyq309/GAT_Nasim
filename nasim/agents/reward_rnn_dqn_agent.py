"""
Sequence RNN-DQN Agent for NASim
--------------------------------
在原始 DQN 基础上引入 RNN（LSTM/GRU），
并改为 **序列化输入**：
h_t = f(h_{t-1}, [obs_{t-k+1}, ..., obs_t])

能够建模长期攻击轨迹的时序依赖。
"""

import random
from pprint import pprint
import numpy as np
import nasim
import gc

from gymnasium import error

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.tensorboard import SummaryWriter
except ImportError as e:
    raise error.DependencyNotInstalled(
        f"{e}. (HINT: run 'pip install nasim[dqn]')"
    )


# ==================================================
# Replay Memory (支持序列存储)
# ==================================================
class SequenceReplayMemory:
    def __init__(self, capacity, s_dims, seq_len, device=None):
        self.capacity = capacity
        self.seq_len = seq_len
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.s_buf = np.zeros((capacity, *s_dims), dtype=np.float32)
        self.a_buf = np.zeros((capacity, 1), dtype=np.int64)
        self.next_s_buf = np.zeros((capacity, *s_dims), dtype=np.float32)
        self.r_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size = 0, 0

    def store(self, s, a, next_s, r, done):
        self.s_buf[self.ptr] = s
        self.a_buf[self.ptr] = a
        self.next_s_buf[self.ptr] = next_s
        self.r_buf[self.ptr] = r
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_sequence_batch(self, batch_size):
        """
        采样一批长度为 seq_len 的时间序列
        """
        idxs = np.random.randint(self.seq_len, self.size, size=batch_size)
        s_seqs, next_s_seqs = [], []
        a, r, d = [], [], []

        for idx in idxs:
            start = idx - self.seq_len
            end = idx
            s_seqs.append(self.s_buf[start:end])
            next_s_seqs.append(self.next_s_buf[start:end])
            a.append(self.a_buf[end - 1])
            r.append(self.r_buf[end - 1])
            d.append(self.done_buf[end - 1])

        s = torch.tensor(np.array(s_seqs), dtype=torch.float32, device=self.device)
        next_s = torch.tensor(np.array(next_s_seqs), dtype=torch.float32, device=self.device)
        a = torch.tensor(np.array(a), dtype=torch.long, device=self.device)
        r = torch.tensor(np.array(r), dtype=torch.float32, device=self.device)
        d = torch.tensor(np.array(d), dtype=torch.float32, device=self.device)
        return s, a, next_s, r, d


# ==================================================
# RNN + DQN 网络结构（序列化输入）
# ==================================================
class SequenceRNN_DQN(nn.Module):
    def __init__(self, input_dim, hidden_layers, num_actions,
                 rnn_type="LSTM", rnn_hidden_dim=128, rnn_layers=1, rnn_dropout=0.1):
        super().__init__()
        self.use_rnn = rnn_type.lower() != "none"
        self.rnn_type = rnn_type.upper()
        self.input_dim = input_dim[0]

        self.feature_layer = nn.Linear(self.input_dim, hidden_layers[0])

        if self.use_rnn:
            rnn_cls = nn.LSTM if self.rnn_type == "LSTM" else nn.GRU
            self.rnn = rnn_cls(
                input_size=hidden_layers[0],
                hidden_size=rnn_hidden_dim,
                num_layers=rnn_layers,
                batch_first=True,
                dropout=rnn_dropout if rnn_layers > 1 else 0.0
            )
            last_dim = rnn_hidden_dim
        else:
            last_dim = hidden_layers[0]

        self.fc_layers = nn.ModuleList()
        for i in range(1, len(hidden_layers)):
            self.fc_layers.append(nn.Linear(last_dim, hidden_layers[i]))
            last_dim = hidden_layers[i]

        self.out = nn.Linear(last_dim, num_actions)

    def forward(self, x):
        """
        输入: x.shape = [B, T, obs_dim]
        输出: Q(s_t, a)
        """
        B, T, _ = x.size()
        x = F.relu(self.feature_layer(x))

        if self.use_rnn:
            self.rnn.flatten_parameters()
            x, _ = self.rnn(x)
        x = x[:, -1, :]  # 取最后时间步隐藏状态

        for layer in self.fc_layers:
            x = F.relu(layer(x))
        q = self.out(x)
        return q


# ==================================================
# RNN-DQN 智能体
# ==================================================
class SequenceRNNDQNAgent:
    def __init__(self, 
                 env, 
                 seed=None, 
                 lr=1e-3, 
                 seq_len=6, 
                 batch_size=32,
                 replay_size=50000, 
                 training_steps=20000, 
                 hidden_sizes=[256, 256],
                 rnn_type="LSTM", 
                 rnn_hidden_dim=32, 
                 rnn_layers=2, 
                 rnn_dropout=0.05,
                 final_epsilon=0.05, 
                 exploration_steps=10000,
                 target_update_freq=500, 
                 gamma=0.93, 
                 success_bonus = 80,        
                 step_penalty = -0.1,
                 info_gain_bonus = 0.5,
                 verbose=True,
                 device=None):

        assert env.flat_actions
        self.verbose = verbose
        self.env = env
        self.seq_len = seq_len
        self.num_actions = env.action_space.n
        self.obs_dim = env.observation_space.shape
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

        self.dqn = SequenceRNN_DQN(
            self.obs_dim, hidden_sizes, self.num_actions,
            rnn_type=rnn_type,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_layers=rnn_layers,
            rnn_dropout=rnn_dropout
        ).to(self.device)

        self.target_dqn = SequenceRNN_DQN(
            self.obs_dim, hidden_sizes, self.num_actions,
            rnn_type=rnn_type,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_layers=rnn_layers,
            rnn_dropout=rnn_dropout
        ).to(self.device)

        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = SequenceReplayMemory(replay_size, self.obs_dim, seq_len, self.device)
        
        self.success_bonus = success_bonus
        self.step_penalty = step_penalty
        self.info_gain_bonus = info_gain_bonus

        self.gamma = gamma
        self.batch_size = batch_size
        self.training_steps = training_steps
        self.target_update_freq = target_update_freq
        self.steps_done = 0

        self.final_epsilon = final_epsilon
        self.epsilon_schedule = np.linspace(1.0, final_epsilon, exploration_steps)
        
        import os
        import datetime
        logdir = os.environ.get("TORCH_TB_DIR", "runs")
        run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logdir = os.path.join(logdir, run_name)
        self.logger = SummaryWriter(logdir)

        # 初始化观测序列缓存
        self.obs_seq = [np.zeros(self.obs_dim) for _ in range(self.seq_len)]

    def get_epsilon(self):
        idx = min(self.steps_done, len(self.epsilon_schedule) - 1)
        return self.epsilon_schedule[idx]

    def select_action(self, seq_obs, epsilon):
        if random.random() > epsilon:
            seq_tensor = torch.tensor(seq_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.dqn(seq_tensor)
            return q_vals.argmax(dim=1).item()
        else:
            return random.randrange(self.num_actions)

    # ========== 关键修改1：新增 get_egreedy_action 函数（兼容原调用） ==========
    def get_egreedy_action(self, obs, epsilon):
        # 更新观测序列
        self.obs_seq.pop(0)
        self.obs_seq.append(np.array(obs))
        # 调用已定义的 select_action
        return self.select_action(np.stack(self.obs_seq), epsilon)

    def optimize(self):
        if self.replay.size < self.seq_len or self.replay.size < self.batch_size:
            return 0, 0
        s, a, next_s, r, d = self.replay.sample_sequence_batch(self.batch_size)
        q_vals = self.dqn(s).gather(1, a)
        with torch.no_grad():
            next_q = self.target_dqn(next_s).max(1, keepdim=True)[0]
            target = r.unsqueeze(1) + self.gamma * (1 - d.unsqueeze(1)) * next_q
        loss = self.loss_fn(q_vals, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), 1.0)
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

        return loss.item(), q_vals.mean().item()

    def run_train_episode(self, step_limit):
        obs, _ = self.env.reset()
        done = False
        total_reward = 0
        steps = 0
        self.last_obs = None   # 用于信息增益奖励
        # 重置观测序列
        self.obs_seq = [np.zeros(self.obs_dim) for _ in range(self.seq_len - 1)]
        self.obs_seq.append(np.array(obs))

        while not done and steps < step_limit:
            # epsilon-greedy
            epsilon = self.get_epsilon()
            # 调用修正后的动作选择函数
            action = self.get_egreedy_action(obs, epsilon)

            # env step
            next_obs, raw_reward, done, limit, _ = self.env.step(action)

            # ---------------------------------------------------------
            #  Reward Shaping（统一写在这里）
            # ---------------------------------------------------------
            reward = self.step_penalty

            # 成功奖励
            if self.env.goal_reached():
                reward += self.success_bonus

            # 信息增益奖励（状态观察发生变化）
            obs_array = np.array(obs)
            next_obs_array = np.array(next_obs)

            if self.last_obs is not None:
                if not np.array_equal(next_obs_array, self.last_obs):
                    reward += self.info_gain_bonus
            self.last_obs = next_obs_array.copy()

            # 保留 NASim 自带的正向奖励
            if raw_reward > 0:
                reward += raw_reward
            # ---------------------------------------------------------

            # ========== 关键修改2：删除GAT相关冗余代码（RNN-DQN无需图编码） ==========
            # 直接使用原始观测，无需转化为图向量
            s_vec = np.array(obs)
            next_vec = np.array(next_obs)

            # store transition
            self.replay.store(s_vec, action, next_vec, reward, float(done))

            # optimize
            loss, q_mean = self.optimize()

            total_reward += reward
            obs = next_obs
            steps += 1
            self.steps_done += 1

        return total_reward, steps, self.env.goal_reached()

    
    def train(self):
        num_episodes = 0
        while self.steps_done < self.training_steps:
            ep_return, ep_steps, goal = self.run_train_episode(self.training_steps - self.steps_done)
            loss, q_mean = self.optimize()
            num_episodes += 1
            
            self.logger.add_scalar("episode", num_episodes, self.steps_done)
            self.logger.add_scalar("epsilon", self.get_epsilon(), self.steps_done)
            self.logger.add_scalar("episode_return", ep_return, self.steps_done)
            self.logger.add_scalar("episode_steps", ep_steps, self.steps_done)
            self.logger.add_scalar("episode_goal_reached", int(goal), self.steps_done)
            
            self.logger.add_scalar("loss", loss, self.steps_done)
            self.logger.add_scalar("q_value_mean", q_mean, self.steps_done)

            if num_episodes % 10 == 0 and self.verbose:
                print(f"Episode {num_episodes}: steps={self.steps_done}, return={ep_return}, goal={goal}, loss={loss}, q={q_mean}")

        self.logger.close()


    def run_eval_episode(self, env=None, render=False):
        if env is None:
            env = self.env
        obs, _ = env.reset()
        done, steps, ep_ret = False, 0, 0
        obs_seq = [np.zeros_like(obs) for _ in range(self.seq_len - 1)]
        obs_seq.append(obs)
        max_eval_steps = 2000
        while not done and steps < max_eval_steps:
            seq_tensor = torch.tensor(np.stack(obs_seq), dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.dqn(seq_tensor)
            a = q_vals.argmax(dim=1).item()
            obs, r, done, _, _ = env.step(a)
            obs_seq.pop(0)
            obs_seq.append(obs)
            ep_ret += r
            steps += 1
            if render:
                env.render()
        env.close()
        return ep_ret, steps, env.goal_reached()


# ==================================================
# 运行
# ==================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str)
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--rnn_type", type=str, default="LSTM", choices=["LSTM", "GRU", "None"])
    parser.add_argument("--rnn_hidden_dim", type=int, default=128)
    parser.add_argument("--rnn_layers", type=int, default=1)
    parser.add_argument("--rnn_dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0,help="(default=0)")
    # ========== 关键修改3：修复参数名拼写（training_step → training_steps） ==========
    parser.add_argument("--training_steps", type=int, default=20000)
    parser.add_argument("--render_eval", action="store_true", default=False, help="Whether to render evaluation episode")
    args = parser.parse_args()

    env = nasim.make_benchmark(args.env_name, args.seed, fully_obs=False, flat_actions=True, flat_obs=True)

    # 修复：过滤掉 Agent __init__ 不接收的参数（如 env_name、render_eval）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent_kwargs = {
        "seq_len": args.seq_len,
        "rnn_type": args.rnn_type,
        "rnn_hidden_dim": args.rnn_hidden_dim,
        "rnn_layers": args.rnn_layers,
        "rnn_dropout": args.rnn_dropout,
        "training_steps": args.training_steps,
        "device": device
    }
    # 实例化 Agent 时只传入有效参数
    agent = SequenceRNNDQNAgent(env, **agent_kwargs)
    agent.train()
    agent.run_eval_episode(render=args.render_eval)