import os, sys
# === 强制使用本地 nasim 路径 ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
print("当前 sys.path[0]:", sys.path[0])


import nasim
from nasim import make_benchmark
from nasim.envs.utils import obs_to_graph


# 加载 NASim 环境
env = nasim.make_benchmark("tiny")
obs, _ = env.reset()

# 生成图结构
obs_graph = obs_to_graph(env.last_obs, env.network)
print(obs_graph)