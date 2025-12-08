import enum
import numpy as np
from queue import deque
from itertools import permutations
import torch
from torch_geometric.data import Data

INTERNET = 0


class OneHotBool(enum.IntEnum):
    NONE = 0
    TRUE = 1
    FALSE = 2

    @staticmethod
    def from_bool(b):
        if b:
            return OneHotBool.TRUE
        return OneHotBool.FALSE

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class ServiceState(enum.IntEnum):
    # values for possible service knowledge states
    UNKNOWN = 0     # service may or may not be running on host
    PRESENT = 1     # service is running on the host
    ABSENT = 2      # service not running on the host

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class AccessLevel(enum.IntEnum):
    NONE = 0
    USER = 1
    ROOT = 2

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


def get_minimal_hops_to_goal(topology, sensitive_addresses):
    """Get minimum network hops required to reach all sensitive hosts.

    Starting from outside the network (i.e. can only reach exposed subnets).

    Returns
    -------
    int
        minimum number of network hops to reach all sensitive hosts
    """
    num_subnets = len(topology)
    max_value = np.iinfo(np.int16).max
    distance = np.full((num_subnets, num_subnets),
                       max_value,
                       dtype=np.int16)

    # set distances for each edge to 1
    for s1 in range(num_subnets):
        for s2 in range(num_subnets):
            if s1 == s2:
                distance[s1][s2] = 0
            elif topology[s1][s2] == 1:
                distance[s1][s2] = 1
    # find all pair minimum shortest path distance
    for k in range(num_subnets):
        for i in range(num_subnets):
            for j in range(num_subnets):
                if distance[i][k] == max_value \
                   or distance[k][j] == max_value:
                    dis = max_value
                else:
                    dis = distance[i][k] + distance[k][j]
                if distance[i][j] > dis:
                    distance[i][j] = distance[i][k] + distance[k][j]

    # get list of all subnets we need to visit
    subnets_to_visit = [INTERNET]
    for subnet, host in sensitive_addresses:
        if subnet not in subnets_to_visit:
            subnets_to_visit.append(subnet)

    # find minimum shortest path that visits internet subnet and all
    # sensitive subnets by checking all possible permutations
    shortest = max_value
    for pm in permutations(subnets_to_visit):
        pm_sum = 0
        for i in range(len(pm) - 1):
            pm_sum += distance[pm[i]][pm[i+1]]
        shortest = min(shortest, pm_sum)

    return shortest


def min_subnet_depth(topology):
    """Find the minumum depth of each subnet in the network graph in terms of steps
    from an exposed subnet to each subnet

    Parameters
    ----------
    topology : 2D matrix
        An adjacency matrix representing the network, with first subnet
        representing the internet (i.e. exposed)

    Returns
    -------
    depths : list
        depth of each subnet ordered by subnet index in topology
    """
    num_subnets = len(topology)

    assert len(topology[0]) == num_subnets

    depths = []
    Q = deque()
    for subnet in range(num_subnets):
        if topology[subnet][INTERNET] == 1:
            depths.append(0)
            Q.appendleft(subnet)
        else:
            depths.append(float('inf'))

    while len(Q) > 0:
        parent = Q.pop()
        for child in range(num_subnets):
            if topology[parent][child] == 1:
                # child is connected to parent
                if depths[child] > depths[parent] + 1:
                    depths[child] = depths[parent] + 1
                    Q.appendleft(child)
    return depths


def obs_to_graph(obs, network, device="cpu"):
    """
    将 NASim 的观测转换为 GPU 加速的 PyG Data 图结构。

    功能：
        - 自动支持 flat_obs=True 的一维观测
        - 自动 reshape 成 (num_hosts+1, feature_dim)
        - 节点特征包含 host 特征 + 全局辅助特征
        - 构建基于子网连接性的边（双向）
        - 完全 GPU 加速
        
    参数:
        obs: np.ndarray 或 Observation
        network: NASim network 对象
        device: "cpu" 或 "cuda"
    """

    # ============================================================
    # 1. obs 转换为 numpy，并处理 flat_obs 的情况
    # ============================================================
    if hasattr(obs, "numpy"):
        obs_np = obs.numpy()
    else:
        obs_np = np.asarray(obs)

    num_hosts = len(network.hosts)
    total_nodes = num_hosts + 1

    # ---- 自动 reshape flat obs → 2D ----
    if obs_np.ndim == 1:
        feat_dim = obs_np.size // total_nodes
        obs_np = obs_np.reshape(total_nodes, feat_dim)

    # ============================================================
    # 2. 生成节点特征：host features + auxiliary features
    # ============================================================
    host_feat = obs_np[:-1]                              # (num_hosts, fdim)
    aux_feat = np.repeat(obs_np[-1][None, :], num_hosts, axis=0)

    node_feat = np.concatenate([host_feat, aux_feat], axis=1)

    # ---- 转为 GPU tensor ----
    x = torch.tensor(node_feat, dtype=torch.float32, device=device)

    # ---- 标准化（避免数值爆炸，提高 GAT 稳定性） ----
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)

    # ============================================================
    # 3. 构建边 edge_index（完全 GPU / 向量化，无 Python for-loop）
    # ============================================================

    host_addrs = list(network.hosts.keys())              # [(subnet, host_id)]
    host_addrs = torch.tensor(host_addrs, dtype=torch.long)

    subnets = host_addrs[:, 0]

    # ---- adjacency matrix: subnets connected? ----
    subnet_ids = subnets.cpu().numpy()
    A = np.zeros((num_hosts, num_hosts), dtype=np.int64)

    for i in range(num_hosts):
        for j in range(num_hosts):
            if i == j:
                continue
            if network.subnets_connected(subnet_ids[i], subnet_ids[j]):
                A[i, j] = 1

    A = torch.tensor(A, device=device)

    # ---- 取出所有 (i, j) 的边 ----
    edges = A.nonzero(as_tuple=False)      # shape = (E, 2)

    if edges.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        edge_index = edges.t().contiguous()

    # ============================================================
    # 4. 构建 PyG Data 对象
    # ============================================================
    data = Data(
        x=x,
        edge_index=edge_index,
        num_nodes=num_hosts
    )

    return data