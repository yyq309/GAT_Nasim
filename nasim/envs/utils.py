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


def obs_to_graph(obs, network):
    """
    将 NASim 的观测 obs 转换为图结构 Data 对象
    ---------------------------------------------
    参数:
        obs: numpy.ndarray 或 nasim.envs.observation.Observation
            NASim 的观测输出，形状通常为 (num_hosts+1, feature_dim)
        network: nasim.envs.network.Network
            NASim 网络结构对象，用于构造边信息
    返回:
        torch_geometric.data.Data
            PyTorch Geometric 的图对象，包含节点特征、边索引及节点地址映射
    """
    # ---------- Step 1. 处理节点特征 ----------
    if hasattr(obs, "numpy"):
        # 兼容 Observation 对象
        obs_tensor = obs.numpy()
    else:
        # 兼容 numpy.ndarray
        obs_tensor = np.array(obs)

    # 验证输入维度
    if obs_tensor.ndim != 2:
        raise ValueError("obs 必须是 2D 数组或 Observation 对象")
    if obs_tensor.shape[0] != len(network.hosts) + 1:
        raise ValueError(f"观测的主机数量与网络不匹配: 观测{obs_tensor.shape[0]-1}台，网络{len(network.hosts)}台")

    # 保留辅助信息或加入到节点特征
    node_features = obs_tensor[:-1]
    aux_features = obs_tensor[-1].reshape(1, -1).repeat(len(node_features), axis=0)
    x = np.concatenate([node_features, aux_features], axis=1)
    x = torch.tensor(x, dtype=torch.float)


    # ---------- Step 2. 构建边索引（基于子网连接性） ----------
    edges = []
    host_list = list(network.hosts.keys())  # 主机地址列表 [(subnet, id), ...]
    host_index_map = {addr: i for i, addr in enumerate(host_list)}  # 地址到索引的映射

    # 遍历所有主机对，根据子网连接性构建边
    for i, src_addr in enumerate(host_list):
        src_subnet = src_addr[0]
        for j, dst_addr in enumerate(host_list):
            if i == j:
                continue  # 排除自环边
            dst_subnet = dst_addr[0]
            # 检查两个子网是否连接（基于 network.topology）
            if network.subnets_connected(src_subnet, dst_subnet):
                edges.append([i, j])
                edges.append([j, i])

    # 转换为 PyTorch Geometric 要求的边索引格式 (2, E)
    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # ---------- Step 3. 补充节点元信息（可选，便于调试） ----------
    # 存储主机地址与索引的映射关系（如 (subnet, id) -> index）
    node_addresses = torch.tensor(
        [[addr[0], addr[1]] for addr in host_list],
        dtype=torch.int
    )
    
    # 归一化
    x = (x - x.mean(dim=0)) / (x.std(dim=0)+1e-6)

    # ---------- Step 4. 生成图对象 ----------
    data = Data(
        x=x,
        edge_index=edge_index,
        node_addresses=node_addresses,  # 节点地址元信息
        num_nodes=len(host_list)
    )

    return data