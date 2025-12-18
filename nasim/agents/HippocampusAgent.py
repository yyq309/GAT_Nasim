"""
Hippocampus-Inspired Modular Agent for NASim
---------------------------------------------------------------------
DG  = Attention Encoder (GATv2 / VectorAttention)
CA3 = Temporal Memory   (GRU / LSTM)
CA1 = Value Head        (DQN)
"""

import os
import random
import argparse
from typing import Optional

import numpy as np
import nasim

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch_geometric.nn import GATv2Conv
    from torch_geometric.data import Data
except ImportError as e:
    raise ImportError(
        f"{e}. Please install dependencies: pip install torch torch-geometric nasim"
    )


# -------------------------
# Utility: obs -> PyG Data (GPU-friendly)
# -------------------------
def obs_to_data_gpu(obs, network, device: torch.device):
    """Convert NASim observation to torch_geometric.data.Data on given device."""
    # accept Observation objects or numpy arrays
    if hasattr(obs, "numpy"):
        obs_arr = obs.numpy()
    else:
        obs_arr = np.asarray(obs)

    # If flat (1D), try safe reshape using network.hosts count
    if obs_arr.ndim == 1:
        num_nodes = len(network.hosts) + 1
        if obs_arr.size % num_nodes != 0:
            raise ValueError("[obs_to_data_gpu] Received 1D flat obs with incompatible size; "
                             "create env with flat_obs=False.")
        feat_dim = obs_arr.size // num_nodes
        obs_arr = obs_arr.reshape(num_nodes, feat_dim)

    if obs_arr.ndim != 2:
        raise ValueError("obs must be 2D array-like of shape (num_nodes+1, feat_dim)")

    if obs_arr.shape[0] != len(network.hosts) + 1:
        raise ValueError(f"[obs_to_data_gpu] obs rows ({obs_arr.shape[0]}) != network.hosts+1 ({len(network.hosts)+1}). "
                         "Ensure env created with flat_obs=False and fully_obs=False.")

    # node features: all but last row; last row is global auxiliary info
    node_feats = obs_arr[:-1]
    global_feat = obs_arr[-1]

    # tile global feature to nodes
    global_tiled = np.tile(global_feat.reshape(1, -1), (node_feats.shape[0], 1))
    x_np = np.concatenate([node_feats, global_tiled], axis=1).astype(np.float32)
    x = torch.from_numpy(x_np).to(device)

    # build edges using network topology
    edges = []
    hosts_list = list(network.hosts.keys())
    addr_to_idx = {addr: i for i, addr in enumerate(hosts_list)}

    for i, src in enumerate(hosts_list):
        subnet_src = src[0]
        for j, dst in enumerate(hosts_list):
            if i == j:
                continue
            subnet_dst = dst[0]
            try:
                if network.subnets_connected(subnet_src, subnet_dst):
                    edges.append([i, j])
            except Exception:
                edges.append([i, j])

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()

    data = Data(x=x, edge_index=edge_index, num_nodes=x.shape[0])
    return data


# -------------------------
# DG Modules: GATEncoder and VectorAttentionEncoder
# -------------------------
class VectorAttention(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        logits = self.att(x).squeeze(-1)
        attn = torch.softmax(logits, dim=0)
        return attn


class GATEncoder(nn.Module):
    def __init__(self, in_dim, gat_hidden=64, out_dim=64, heads=4, dropout=0.1, topk=5):
        super().__init__()
        self.topk = topk
        self.gat1 = GATv2Conv(in_dim, gat_hidden, heads=heads, dropout=dropout)
        self.gat2 = GATv2Conv(gat_hidden * heads, out_dim, heads=1, dropout=dropout)
        self.att_mlp = nn.Linear(out_dim, 1)
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        h = F.relu(self.gat1(x, edge_index))
        h = F.relu(self.gat2(h, edge_index))
        h = self.out_proj(h)
        att_logits = self.att_mlp(h).squeeze(-1)
        scores = torch.softmax(att_logits, dim=0)
        N = h.size(0)
        k = min(self.topk, N)
        _, idx = torch.topk(scores, k, sorted=False)
        topk_nodes = h[idx]
        if k < self.topk:
            pad = torch.zeros(self.topk - k, topk_nodes.size(1), device=h.device, dtype=h.dtype)
            topk_nodes = torch.cat([topk_nodes, pad], dim=0)
        flat = topk_nodes.reshape(-1)
        global_emb = h.mean(dim=0)
        return flat, global_emb


class VectorAttentionEncoder(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=64, topk=5):
        super().__init__()
        self.topk = topk
        self.lin = nn.Linear(in_dim, out_dim)
        self.att = VectorAttention(out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, data: Data):
        x = F.relu(self.lin(data.x))
        x = self.out_proj(x)
        scores = self.att(x)
        N = x.size(0)
        k = min(self.topk, N)
        _, idx = torch.topk(scores, k, sorted=False)
        topk = x[idx]
        if k < self.topk:
            pad = torch.zeros(self.topk - k, topk.size(1), device=x.device, dtype=x.dtype)
            topk = torch.cat([topk, pad], dim=0)
        flat = topk.reshape(-1)
        global_emb = x.mean(dim=0)
        return flat, global_emb


# -------------------------
# CA3 (RNN) and CA1 (DQN head)
# -------------------------
class CA3Memory(nn.Module):
    def __init__(self, input_dim, hidden_dim, rnn_type="gru", num_layers=1, dropout=0.0):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)

    def forward(self, seq, hx=None):
        return self.rnn(seq, hx)


class CA1ValueHead(nn.Module):
    def __init__(self, input_dim, hidden_sizes, num_actions):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -------------------------
# Replay Buffer 
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, seq_len: int, feat_dim: int, device: torch.device):
        self.capacity = int(capacity)
        self.seq_len = int(seq_len)
        self.feat_dim = int(feat_dim)
        self.device = device

        self.ptr = 0
        self.size = 0

        # 核心优化：直接在GPU分配内存
        self.s = torch.zeros((self.capacity, self.seq_len, self.feat_dim), dtype=torch.float32, device=self.device)
        self.a = torch.zeros((self.capacity, 1), dtype=torch.int64, device=self.device)
        self.ns = torch.zeros((self.capacity, self.seq_len, self.feat_dim), dtype=torch.float32, device=self.device)
        self.r = torch.zeros(self.capacity, dtype=torch.float32, device=self.device)
        self.d = torch.zeros(self.capacity, dtype=torch.float32, device=self.device)

    def store(self, s_np, a_int, ns_np, r_float, done_float):
        self.s[self.ptr] = torch.from_numpy(s_np).to(self.device)
        self.a[self.ptr] = torch.tensor([a_int], dtype=torch.int64, device=self.device)
        self.ns[self.ptr] = torch.from_numpy(ns_np).to(self.device)
        self.r[self.ptr] = torch.tensor(r_float, dtype=torch.float32, device=self.device)
        self.d[self.ptr] = torch.tensor(done_float, dtype=torch.float32, device=self.device)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int):
        if self.size < batch:
            return None
        idxs = torch.randint(0, self.size, (batch,), device=self.device)
        return (
            self.s[idxs], self.a[idxs], self.ns[idxs],
            self.r[idxs], self.d[idxs]
        )


# -------------------------
# 核心改动1：新增设备选择函数（优先cuda:1）
# -------------------------
def get_target_device(manual_device: Optional[str] = None) -> torch.device:
    """
    优先选择第二块GPU（cuda:1），兼容手动指定/单GPU/无GPU场景
    :param manual_device: 手动指定的设备（如 "cuda:0", "cpu"）
    :return: 最终使用的设备
    """
    # 手动指定设备时优先使用
    if manual_device is not None:
        return torch.device(manual_device)
    
    # 自动选择：优先cuda:1，其次cuda:0，最后cpu
    if torch.cuda.is_available():
        if torch.cuda.device_count() >= 2:
            device = torch.device("cuda:1")
            print(f"自动选择第二块显卡: {torch.cuda.get_device_name(1)} (cuda:1)")
        else:
            device = torch.device("cuda:0")
            print(f"仅检测到1块显卡，使用: {torch.cuda.get_device_name(0)} (cuda:0)")
    else:
        device = torch.device("cpu")
        print("无可用GPU，使用CPU")
    return device


# -------------------------
# Hippocampus Agent (精简版，适配批量脚本+独立运行)
# -------------------------
class HippocampusAgent:
    def __init__(self,
                 env,
                 att_type: str = "gat",
                 rnn_type: str = "gru",
                 topk: int = 5,
                 gat_hidden: int = 64,
                 gat_out: int = 64,
                 heads: int = 4,
                 dropout: float = 0.1,
                 rnn_hidden: int = 128,
                 rnn_layers: int = 1,
                 hidden_sizes=(256, 256),
                 seq_len: int = 6,
                 lr: float = 1e-3,
                 gamma: float = 0.99,
                 batch_size: int = 128,
                 training_steps: int = 50000,
                 replay_size: int = 20000,
                 final_epsilon: float = 0.05,
                 eps_decay_steps: int = 20000,
                 warmup_steps: int = 2000,
                 target_update_interval: int = 2000,
                 success_bonus: float = 60.0,
                 step_penalty: float = -0.1,
                 info_gain_bonus: float = 0.5,
                 device: Optional[torch.device] = None,
                 seed: int = 0,
                 verbose: bool = False):
        assert env.flat_actions, "agent supports flat actions only"
        self.env = env
        # 核心改动2：使用新的设备选择逻辑，优先cuda:1
        self.device = get_target_device(device) if device is None else device
        self.verbose = verbose

        # GPU优化配置（绑定到目标设备）
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)  # 设为当前线程默认GPU
            torch.backends.cudnn.benchmark = True
            self.stream = torch.cuda.Stream(device=self.device)
        else:
            self.stream = None

        # 固定随机种子（按设备适配）
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        # 获取观测维度
        obs, _ = env.reset()
        obs_arr = np.asarray(obs)
        if obs_arr.ndim == 1:
            num_nodes = len(env.network.hosts) + 1
            feat_dim = obs_arr.size // num_nodes
            obs_arr = obs_arr.reshape(num_nodes, feat_dim)
        data0 = obs_to_data_gpu(obs_arr, env.network, device=self.device)
        in_dim = data0.x.shape[1]

        # 初始化模型组件
        self.att_type = att_type.lower()
        if self.att_type == "gat":
            self.dg = GATEncoder(in_dim, gat_hidden, gat_out, heads, dropout, topk)
            feat_dim = topk * gat_out
        else:
            self.dg = VectorAttentionEncoder(in_dim, gat_hidden, gat_out, topk)
            feat_dim = topk * gat_out

        self.ca3 = CA3Memory(feat_dim, rnn_hidden, rnn_type, num_layers=rnn_layers, dropout=0.0)
        self.ca1 = CA1ValueHead(rnn_hidden, list(hidden_sizes), env.action_space.n)

        # 目标网络（强制移到目标设备）
        self.target_ca1 = CA1ValueHead(rnn_hidden, list(hidden_sizes), env.action_space.n)
        self.target_ca1.load_state_dict(self.ca1.state_dict())
        self.target_ca1.to(self.device)
        self.target_ca1.eval()

        # 移到目标GPU（显式指定设备）
        self.dg.to(self.device)
        self.ca3.to(self.device)
        self.ca1.to(self.device)

        # 经验回放池（绑定目标设备）
        self.replay = ReplayBuffer(replay_size, seq_len, feat_dim, self.device)

        # 超参数
        self.gamma = gamma
        self.lr = lr
        self.batch = batch_size
        self.training_steps = int(training_steps)
        self.seq_len = int(seq_len)
        self.final_epsilon = final_epsilon
        self.eps_decay = float(eps_decay_steps)
        self.warmup_steps = int(warmup_steps)
        self.target_update_interval = int(target_update_interval)
        
        # 环境专属奖励参数
        self.success_bonus = success_bonus
        self.step_penalty = step_penalty
        self.info_gain_bonus = info_gain_bonus

        # 优化器
        self.optimizer = optim.Adam(
            list(self.dg.parameters()) + list(self.ca3.parameters()) + list(self.ca1.parameters()),
            lr=lr, weight_decay=1e-5, betas=(0.9, 0.999), eps=1e-8
        )
        # 权重初始化
        for m in list(self.dg.modules()) + list(self.ca3.modules()) + list(self.ca1.modules()):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.loss_fn = nn.SmoothL1Loss()

        # 训练状态记录
        self.steps_done = 0
        self.episodes = 0
        self.last_obs = None

    # epsilon衰减策略
    def epsilon(self):
        # 更慢的线性衰减，防止过早收敛
        decay = min(1.0, self.steps_done / max(1, self.eps_decay))
        return (1.0 - decay) * 1.0 + decay * self.final_epsilon

    # 编码观测（强制使用目标设备）
    def encode_obs(self, obs):
        data = obs_to_data_gpu(obs, self.env.network, self.device)
        flat, glob = self.dg(data)
        return flat

    # 单步优化（适配目标设备的CUDA流）
    def optimize_step(self):
        batch = self.replay.sample(self.batch)
        if batch is None:
            return 0.0, 0.0
        s, a, ns, r, d = batch

        # 奖励归一化/裁剪
        r = torch.clamp(r, -10.0, 10.0)

        # 适配目标设备的CUDA流
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                return self._compute_optimization(s, a, ns, r, d)
        else:
            # CPU场景直接计算
            return self._compute_optimization(s, a, ns, r, d)

    # 拆分优化计算逻辑（便于设备适配）
    def _compute_optimization(self, s, a, ns, r, d):
        # RNN前向传播（已在目标设备）
        rnn_out, _ = self.ca3(s)
        q_vals = self.ca1(rnn_out[:, -1, :])
        q = q_vals.gather(1, a).squeeze(1)

        with torch.no_grad():
            rnn_ns, _ = self.ca3(ns)
            next_q_vals = self.target_ca1(rnn_ns[:, -1, :])
            next_q = next_q_vals.max(1)[0]
            target = r + self.gamma * (1.0 - d) * next_q

        loss = self.loss_fn(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        # 更严格的梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            list(self.dg.parameters()) + list(self.ca3.parameters()) + list(self.ca1.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()

        # 定期更新目标网络
        if self.steps_done % self.target_update_interval == 0:
            self.target_ca1.load_state_dict(self.ca1.state_dict())

        q_mean = q_vals.max(1)[0].mean().item()
        return loss.item(), q_mean

    # 单轮训练episode（批量脚本核心调用）
    def run_train_episode(self, max_steps_per_episode=500):
        obs, _ = self.env.reset()
        obs_arr = np.asarray(obs)

        # 确保观测维度正确
        if obs_arr.ndim == 1:
            num_nodes = len(self.env.network.hosts) + 1
            feat_dim = obs_arr.size // num_nodes
            obs_arr = obs_arr.reshape(num_nodes, feat_dim)

        self.last_obs = obs_arr.copy()
        done = False
        ep_ret = 0.0
        steps = 0
        seq = []

        # 单episode训练循环
        while not done and steps < max_steps_per_episode and self.steps_done < self.training_steps:
            # 1. 编码当前观测（目标设备）
            emb = self.encode_obs(obs).detach().cpu().numpy()
            seq.append(emb)
            if len(seq) > self.seq_len:
                seq.pop(0)

            # 构造序列输入
            s_np = np.zeros((self.seq_len, emb.shape[0]), dtype=np.float32)
            s_np[-len(seq):] = np.stack(seq, axis=0)

            # 2. 选择动作（ε-贪心）
            eps = self.epsilon()
            if random.random() > eps and self.steps_done > self.warmup_steps:
                with torch.no_grad():
                    s_tensor = torch.tensor(s_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                    rnn_out, _ = self.ca3(s_tensor, None)
                    a = int(self.ca1(rnn_out[:, -1, :]).argmax(dim=1).item())
            else:
                a = random.randint(0, self.env.action_space.n - 1)

            # 3. 环境步进（兼容4/5值返回）
            step_result = self.env.step(a)
            if len(step_result) == 4:
                next_obs, raw_r, done, info = step_result
            else:
                next_obs, raw_r, done, trunc, info = step_result

            next_obs_arr = np.asarray(next_obs)
            if next_obs_arr.ndim == 1:
                num_nodes = len(self.env.network.hosts) + 1
                feat_dim = next_obs_arr.size // num_nodes
                next_obs_arr = next_obs_arr.reshape(num_nodes, feat_dim)

            # 4. 奖励塑形（环境专属参数）
            reward = float(raw_r) + self.step_penalty
            # 信息增益奖励
            info_gain = (next_obs_arr[:-1] > self.last_obs[:-1]).sum()
            reward += self.info_gain_bonus * info_gain
            # 成功奖励
            if self.env.goal_reached():
                reward += self.success_bonus
            # 奖励裁剪，防止极端值影响训练
            reward = max(min(reward, 10.0), -10.0)

            # 5. 编码下一个观测（目标设备）
            emb2 = self.encode_obs(next_obs).detach().cpu().numpy()
            seq2 = seq.copy()
            seq2.append(emb2)
            if len(seq2) > self.seq_len:
                seq2.pop(0)

            ns_np = np.zeros((self.seq_len, emb2.shape[0]), dtype=np.float32)
            ns_np[-len(seq2):] = np.stack(seq2, axis=0)

            # 6. 存储经验（目标设备）
            self.replay.store(s_np, a, ns_np, reward, float(done))

            # 7. 优化（预热后）
            if self.steps_done > self.warmup_steps:
                self.optimize_step()

            # 8. 更新状态
            ep_ret += reward
            obs = next_obs
            self.last_obs = next_obs_arr.copy()
            seq = seq2
            steps += 1
            self.steps_done += 1

        self.episodes += 1
        self.last_obs = None

        return ep_ret, steps, self.env.goal_reached()

    # 评估函数（批量脚本核心调用）
    def evaluate(self, n_episodes: int = 30, max_steps_per_episode: int = 1500):
        stats = {"returns": [], "steps": [], "goals": []}
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            ret = 0.0
            steps = 0
            seq = []

            while not done and steps < max_steps_per_episode:
                # 编码观测（目标设备）
                emb = self.encode_obs(obs).detach()
                seq.append(emb.cpu().numpy())
                if len(seq) > self.seq_len:
                    seq.pop(0)

                # 构造序列输入（目标设备）
                padded = np.zeros((self.seq_len, emb.numel()), dtype=np.float32)
                padded[-len(seq):] = np.stack(seq[-self.seq_len:], axis=0)
                s_tensor = torch.tensor(padded, dtype=torch.float32, device=self.device).unsqueeze(0)

                # 贪心选动作
                with torch.no_grad():
                    rnn_out, _ = self.ca3(s_tensor, None)
                    a = int(self.ca1(rnn_out[:, -1, :]).argmax(dim=1).item())

                # 环境步进
                step_result = self.env.step(a)
                if len(step_result) == 4:
                    obs, raw_r, done, info = step_result
                else:
                    obs, raw_r, done, trunc, info = step_result

                ret += raw_r
                steps += 1

            stats["returns"].append(ret)
            stats["steps"].append(steps)
            stats["goals"].append(float(self.env.goal_reached()))

        return stats


# -------------------------
# 独立运行入口（恢复命令行执行功能）
# -------------------------
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Hippocampus Agent for NASim")
    parser.add_argument("env_name", type=str, help="NASim environment name (e.g., tiny/small/medium)")
    parser.add_argument("--att-type", choices=["gat", "vector"], default="gat", help="Attention encoder type")
    parser.add_argument("--rnn-type", choices=["gru", "lstm"], default="gru", help="RNN type for CA3")
    parser.add_argument("--training-steps", type=int, default=50000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--seq-len", type=int, default=6, help="Sequence length for RNN")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--device", type=str, default=None, help="Device (e.g., cuda:1/cpu)")  # 支持手动指定cuda:1
    args = parser.parse_args()

    # 创建NASim环境
    env = nasim.make_benchmark(
        args.env_name,
        seed=args.seed,
        fully_obs=False,
        flat_actions=True,
        flat_obs=False
    )

    # 确定运行设备（优先手动指定，否则自动选cuda:1）
    device = get_target_device(args.device)

    # 根据环境自动适配奖励参数
    env_reward_config = {
        "tiny": {"success_bonus": 40.0, "step_penalty": -0.2, "info_gain_bonus": 0.3},
        "small": {"success_bonus": 80.0, "step_penalty": -0.12, "info_gain_bonus": 0.6},
        "medium": {"success_bonus": 150.0, "step_penalty": -0.07, "info_gain_bonus": 1.0}
    }
    reward_config = env_reward_config.get(args.env_name, env_reward_config["small"])

    # 初始化Agent
    agent = HippocampusAgent(
        env=env,
        att_type=args.att_type,
        rnn_type=args.rnn_type,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=args.seed,
        device=device,  # 传入目标设备（cuda:1）
        verbose=args.verbose,
        # 环境专属奖励参数
        success_bonus=reward_config["success_bonus"],
        step_penalty=reward_config["step_penalty"],
        info_gain_bonus=reward_config["info_gain_bonus"]
    )

    # 打印初始化信息
    if args.verbose:
        print(f"[INFO] Starting training on {args.env_name} environment")
        print(f"[INFO] Device: {device} | Attention: {args.att_type} | RNN: {args.rnn_type}")
        print(f"[INFO] Reward config: {reward_config}")

    # 训练循环（带进度条）
    try:
        from tqdm import tqdm
        pbar = tqdm(total=args.training_steps, desc="Training", unit="step")
        last_steps = 0
        
        while agent.steps_done < args.training_steps:
            ep_ret, ep_steps, goal = agent.run_train_episode()
            pbar.update(agent.steps_done - last_steps)
            last_steps = agent.steps_done
            
            # 打印每轮训练信息
            if args.verbose and agent.episodes % 10 == 0:
                pbar.set_postfix({
                    "ep": agent.episodes,
                    "ret": f"{ep_ret:.2f}",
                    "goal": int(goal),
                    "eps": f"{agent.epsilon():.3f}",
                    "device": str(device)  # 新增：显示当前设备
                })
        
        pbar.close()
        
        # 训练完成后评估
        if args.verbose:
            print("\n[INFO] Training completed, starting evaluation...")
        eval_stats = agent.evaluate(n_episodes=30)
        
        # 打印评估结果
        print("\n[Final Evaluation Results]")
        print(f"Mean Return: {np.mean(eval_stats['returns']):.2f}")
        print(f"Success Rate: {np.mean(eval_stats['goals']):.2f}")
        print(f"Mean Steps per Episode: {np.mean(eval_stats['steps']):.2f}")
        print(f"[INFO] Training completed on device: {device}")  # 确认设备
        
    except ImportError:
        # 无tqdm时的简易训练
        print("[WARN] tqdm not found, running without progress bar")
        while agent.steps_done < args.training_steps:
            ep_ret, ep_steps, goal = agent.run_train_episode()
            if args.verbose and agent.episodes % 10 == 0:
                print(f"Ep {agent.episodes}: Ret={ep_ret:.2f}, Steps={ep_steps}, Goal={int(goal)}, Device={device}")
        
        # 评估
        eval_stats = agent.evaluate(n_episodes=30)
        print("\n[Final Evaluation Results]")
        print(f"Mean Return: {np.mean(eval_stats['returns']):.2f}")
        print(f"Success Rate: {np.mean(eval_stats['goals']):.2f}")
        print(f"[INFO] Training completed on device: {device}")

if __name__ == "__main__":
    main()