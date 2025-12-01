"""
Hippocampus-Inspired Modular Agent for NASim
--------------------------------------------

This agent implements the biological DG–CA3–CA1 architecture:

DG  = Attention Encoder (GATv2 / VectorAttention)
CA3 = Temporal Memory   (GRU / LSTM)
CA1 = Value Head        (DQN)

All modules are REPLACEABLE via command-line flags:
    --att-type gat|vector
    --rnn-type gru|lstm

Usage:
    python -m nasim.agents.HippocampusAgent tiny --att-type gat --rnn-type gru
"""

import random
import numpy as np
import nasim

from gymnasium import error

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch_geometric.nn import GATv2Conv
    from torch.utils.tensorboard import SummaryWriter
    from torch_geometric.data import Data
except ImportError as e:
    raise error.DependencyNotInstalled(
        f"{e}. Please install dependencies: pip install torch torch-geometric nasim[dqn]"
    )

from nasim.envs.utils import obs_to_graph


# ============================================================
# DG MODULES (ATTENTION ENCODERS)
# ============================================================

class VectorAttention(nn.Module):
    """Simple learnable vector attention: att = softmax(Wx)."""

    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x, edge_index=None):
        scores = self.att(x).squeeze(-1)   # [N]
        scores = torch.softmax(scores, dim=0)
        return scores


class GATEncoder(nn.Module):
    """DG module: GATv2 attention + TopK + padding to fixed size."""

    def __init__(self, in_dim, gat_hidden, out_dim, heads, dropout, topk=5):
        super().__init__()
        self.topk = topk
        self.gat1 = GATv2Conv(in_dim, gat_hidden, heads=heads, dropout=dropout)
        self.gat2 = GATv2Conv(gat_hidden * heads, out_dim, heads=1, dropout=dropout)
        self.att_mlp = nn.Linear(out_dim, 1)

    def forward(self, graph: Data):
        x, edge_index = graph.x, graph.edge_index

        h = F.relu(self.gat1(x, edge_index))
        h = F.relu(self.gat2(h, edge_index))     # [N, out_dim]

        # global embedding (mean pooling)
        g = h.mean(dim=0)

        # attention scores
        att_logits = self.att_mlp(h).squeeze(-1)
        scores = torch.softmax(att_logits, dim=0)

        N = h.size(0)
        k = min(self.topk, N)
        _, idx = torch.topk(scores, k, dim=0)
        topk_nodes = h[idx]                     # [k, out_dim]

        # ---- PADDING TO FIXED SIZE (topk*out_dim) ----
        if k < self.topk:
            pad = torch.zeros(self.topk - k, topk_nodes.size(1), device=h.device)
            topk_nodes = torch.cat([topk_nodes, pad], dim=0)  # [topk, out_dim]

        flat = topk_nodes.reshape(-1)            # [topk*out_dim]
        return flat, g


class VectorAttentionEncoder(nn.Module):
    """DG module using vector attention instead of GAT."""

    def __init__(self, in_dim, hidden, out_dim, topk=5):
        super().__init__()
        self.topk = topk
        self.lin = nn.Linear(in_dim, out_dim)
        self.att = VectorAttention(out_dim)

    def forward(self, graph: Data):
        x = F.relu(self.lin(graph.x))         
        g = x.mean(dim=0)

        scores = self.att(x)                 
        N = x.size(0)
        k = min(self.topk, N)
        _, idx = torch.topk(scores, k, dim=0)
        topk_nodes = x[idx]

        if k < self.topk:
            pad = torch.zeros(self.topk - k, topk_nodes.size(1), device=x.device)
            topk_nodes = torch.cat([topk_nodes, pad], dim=0)

        flat = topk_nodes.reshape(-1)
        return flat, g


# ============================================================
# CA3 MODULE (TEMPORAL MEMORY: GRU/LSTM)
# ============================================================

class CA3Memory(nn.Module):
    def __init__(self, input_dim, hidden_dim, rnn_type="gru"):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        if self.rnn_type == "gru":
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, seq, hidden=None):
        return self.rnn(seq, hidden)


# ============================================================
# CA1 MODULE (DQN Head)
# ============================================================

class CA1ValueHead(nn.Module):
    def __init__(self, input_dim, hidden_sizes, num_actions):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# REPLAY BUFFER
# ============================================================

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

    def store(self, s, a, next_s, r, done):
        self.s_buf[self.ptr] = s
        self.a_buf[self.ptr] = a
        self.next_s_buf[self.ptr] = next_s
        self.r_buf[self.ptr] = r
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch):
        idxs = np.random.randint(0, self.size, batch)
        s = torch.tensor(self.s_buf[idxs], device=self.device)
        a = torch.tensor(self.a_buf[idxs], device=self.device)
        ns = torch.tensor(self.next_s_buf[idxs], device=self.device)
        r = torch.tensor(self.r_buf[idxs], device=self.device)
        d = torch.tensor(self.done_buf[idxs], device=self.device)
        return s, a, ns, r, d


# ============================================================
# MAIN HIPPOCAMPUS AGENT
# ============================================================

class HippocampusAgent:

    def __init__(self,
                 env,
                 att_type="gat",
                 rnn_type="gru",
                 topk=5,
                 gat_hidden=64,
                 gat_out=64,
                 heads=4,
                 dropout=0.1,
                 rnn_hidden=128,
                 hidden_sizes=[256, 256],
                 seq_len=4,
                 lr=1e-3,
                 gamma=0.99,
                 batch_size=32,
                 training_steps=20000,
                 replay_size=20000,
                 final_epsilon=0.05,
                 seed=0,
                 verbose=True):

        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.seq_len = seq_len

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Get obs shape → graph feature dim
        obs, _ = env.reset()
        obs_arr = np.array(obs)
        num_nodes = len(env.network.hosts) + 1
        fdim = obs_arr.size // num_nodes
        g = obs_to_graph(obs_arr.reshape(num_nodes, fdim), env.network)
        in_dim = g.x.size(1)

        # --------- DG (ENCODER) ---------
        if att_type == "gat":
            self.dg = GATEncoder(in_dim, gat_hidden, gat_out, heads, dropout, topk)
            feat_dim = topk * gat_out
        else:
            self.dg = VectorAttentionEncoder(in_dim, gat_hidden, gat_out, topk)
            feat_dim = topk * gat_out

        # --------- CA3 (RNN) ------------
        self.ca3 = CA3Memory(feat_dim, rnn_hidden, rnn_type)

        # --------- CA1 (DQN Head) -------
        self.ca1 = CA1ValueHead(rnn_hidden, hidden_sizes, env.action_space.n)

        self.dg.to(self.device)
        self.ca3.to(self.device)
        self.ca1.to(self.device)

        # --------- Replay Memory --------
        self.replay = ReplayMemory(replay_size, seq_len, feat_dim, self.device)

        self.gamma = gamma
        self.batch = batch_size
        self.lr = lr

        self.optimizer = optim.Adam(
            list(self.dg.parameters()) +
            list(self.ca3.parameters()) +
            list(self.ca1.parameters()),
            lr=lr
        )

        self.loss_fn = nn.SmoothL1Loss()
        self.steps = 0
        self.final_eps = final_epsilon
        self.training_steps = training_steps

        self.logger = SummaryWriter()

        if verbose:
            print(f"[INFO] Hippocampus Agent running on {self.device}")
            print(f"DG type={att_type}, CA3={rnn_type}, feat_dim={feat_dim}")

    # ---------------------------------------------------------------
    # PLAY ONE STEP
    # ---------------------------------------------------------------
    def encode(self, obs):
        obs = np.array(obs)
        num_nodes = len(self.env.network.hosts) + 1
        fdim = obs.size // num_nodes
        graph = obs_to_graph(obs.reshape(num_nodes, fdim), self.env.network).to(self.device)
        flat, global_emb = self.dg(graph)
        return flat

    def epsilon(self):
        return max(self.final_eps, 1 - self.steps / self.training_steps)

    # ---------------------------------------------------------------
    # TRAINING
    # ---------------------------------------------------------------

    def optimize(self):
        if self.replay.size < self.batch:
            return 0, 0

        s, a, ns, r, d = self.replay.sample(self.batch)

        # RNN forward
        rnn_out, _ = self.ca3(s)
        q = self.ca1(rnn_out[:, -1, :]).gather(1, a).squeeze()

        with torch.no_grad():
            rnn_ns, _ = self.ca3(ns)
            q_next = self.ca1(rnn_ns[:, -1, :]).max(1)[0]
            target = r + self.gamma * (1 - d) * q_next

        loss = self.loss_fn(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), q.mean().item()

    # ---------------------------------------------------------------
    def run_train_episode(self, limit):

        obs, _ = self.env.reset()
        seq = []
        hidden = None
        done = False
        ret = 0
        steps = 0

        while not done and steps < limit:
            emb = self.encode(obs).detach().cpu().numpy()
            seq.append(emb)
            if len(seq) > self.seq_len:
                seq.pop(0)

            # pad seq
            padded = np.zeros((self.seq_len, emb.size), np.float32)
            padded[-len(seq):] = seq
            seq_tensor = torch.tensor(padded, device=self.device).unsqueeze(0)

            rnn_out, hidden = self.ca3(seq_tensor, hidden)
            eps = self.epsilon()

            if random.random() > eps:
                with torch.no_grad():
                    a = self.ca1(rnn_out[:, -1, :]).argmax().item()
            else:
                a = random.randint(0, self.env.action_space.n - 1)

            next_obs, reward, done, _, _ = self.env.step(a)

            # next seq
            emb2 = self.encode(next_obs).detach().cpu().numpy()
            seq2 = seq + [emb2]
            if len(seq2) > self.seq_len:
                seq2.pop(0)
            padded2 = np.zeros((self.seq_len, emb2.size), np.float32)
            padded2[-len(seq2):] = seq2

            self.replay.store(padded, [a], padded2, reward, float(done))

            l, q = self.optimize()
            self.logger.add_scalar("loss", l, self.steps)

            ret += reward
            obs = next_obs
            steps += 1
            self.steps += 1

        return ret, steps, self.env.goal_reached()

    # ---------------------------------------------------------------
    def train(self):
        ep = 0
        while self.steps < self.training_steps:
            ret, s, goal = self.run_train_episode(self.training_steps - self.steps)
            l, q = self.optimize()

            ep += 1
            self.logger.add_scalar("episode",ep, self.steps)
            self.logger.add_scalar("epsilon", self.epsilon(), self.steps)
            self.logger.add_scalar("episode_return", ret, self.steps)
            self.logger.add_scalar("episode_steps", s, self.steps)
            self.logger.add_scalar("episode_goal_reached", int(goal), self.steps)


            self.logger.add_scalar("loss", l, self.steps)
            self.logger.add_scalar("q_value", q, self.steps)
            
            if ep % 10 == 0:
                print(f"[EP{ep}] steps={self.steps} ret={ret:.2f} goal={goal}")

        print("[TRAINING COMPLETED]")

    # ---------------------------------------------------------------
    def run_eval_episode(self, eval_eps=0.05):
        obs, _ = self.env.reset()
        seq = []
        hidden = None

        done = False
        ret = 0
        steps = 0

        while not done:
            emb = self.encode(obs).detach().cpu().numpy()
            seq.append(emb)
            if len(seq) > self.seq_len:
                seq.pop(0)

            padded = np.zeros((self.seq_len, emb.size), np.float32)
            padded[-len(seq):] = seq
            seq_tensor = torch.tensor(padded, device=self.device).unsqueeze(0)

            rnn_out, hidden = self.ca3(seq_tensor, hidden)

            if random.random() > eval_eps:
                with torch.no_grad():
                    a = self.ca1(rnn_out[:, -1, :]).argmax().item()
            else:
                a = random.randint(0, self.env.action_space.n - 1)

            obs, r, done, _, _ = self.env.step(a)
            ret += r
            steps += 1

        return ret, steps, self.env.goal_reached()


# ============================================================
# CMD ENTRY
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name")
    parser.add_argument("--att-type", type=str, default="gat", choices=["gat", "vector"])
    parser.add_argument("--rnn-type", type=str, default="gru", choices=["gru", "lstm"])
    parser.add_argument("--training_steps", type=int, default=10000)
    parser.add_argument("--render_eval", action="store_true")
    args = parser.parse_args()

    env = nasim.make_benchmark(args.env_name, seed=0, fully_obs=True, flat_actions=True, flat_obs=True)

    agent = HippocampusAgent(env, att_type=args.att_type, rnn_type=args.rnn_type,
                             training_steps=args.training_steps)
    agent.train()
    agent.run_eval_episode(render=args.render_eval)
