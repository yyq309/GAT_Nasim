import os
import nasim
import numpy as np
import importlib
from datetime import datetime

from logging_utils import CSVLogger, CheckpointManager
from eval_utils import evaluate_agent


# ----------------------------------------------------
# 注册你的 agent
# ----------------------------------------------------
AGENT_MAP = {
    "dqn":          "nasim.agents.dqn_agent:DQNAgent",
    "rnn_dqn":      "nasim.agents.rnn_dqn_agent:RNNDQNAgent",
    "gat_dqn":      "nasim.agents.gat_dqn_agent:GATDQNAgent",
    "gat_rnn_dqn":  "nasim.agents.gat_rnn_dqn_agent:GATRNNDQNAgent",
}

# ----------------------------------------------------
# 动态加载 agent 类
# ----------------------------------------------------
def load_agent(agent_name):
    module_path, cls_name = AGENT_MAP[agent_name].split(":")
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


# ----------------------------------------------------
# 主实验入口
# ----------------------------------------------------
def run_experiment(agent_name, env_name, seed, training_steps,
                   eval_interval=5000, eval_episodes=50, final_eval_episodes=200):

    # 创建目录
    exp_name = f"{agent_name}_{env_name}_seed{seed}"
    base_dir = f"results/{exp_name}"
    os.makedirs(base_dir, exist_ok=True)

    # CSV + checkpoints
    csv_logger = CSVLogger(os.path.join(base_dir, "train_log.csv"))
    eval_logger = CSVLogger(os.path.join(base_dir, "eval_log.csv"))
    ckpt = CheckpointManager(os.path.join(base_dir, "checkpoints"))

    # 环境
    env = nasim.make_benchmark(env_name, seed=seed,
        fully_obs=True, flat_actions=True, flat_obs=True)

    # 加载 agent
    AgentClass = load_agent(agent_name)
    agent = AgentClass(env, seed=seed, training_steps=training_steps)

    steps_done = 0
    episode = 0
    best_return = -1e9

    # ----------------------------
    # 主训练循环
    # ----------------------------
    while steps_done < training_steps:
        
        ep_ret, ep_steps, goal = agent.run_train_episode(training_steps - steps_done)
        episode += 1
        steps_done += ep_steps

        # 训练记录
        csv_logger.write({
            "episode": episode,
            "steps_done": steps_done,
            "return": ep_ret,
            "steps": ep_steps,
            "goal": int(goal),
        })

        # 保存最好模型
        if ep_ret > best_return:
            best_return = ep_ret
            ckpt.save(agent, "best")

        # 周期评估
        if steps_done % eval_interval < ep_steps:
            avg_r, avg_s, sr = evaluate_agent(agent, env, eval_episodes)
            eval_logger.write({
                "steps_done": steps_done,
                "avg_return": avg_r,
                "avg_steps": avg_s,
                "success_rate": sr
            })
            print(f"[EVAL] {exp_name}  steps={steps_done}  SR={sr:.2f}")

    # ----------------------------
    # Final Evaluation (200 episodes)
    # ----------------------------
    final_r, final_s, final_sr = evaluate_agent(agent, env, final_eval_episodes)
    eval_logger.write({
        "steps_done": steps_done,
        "avg_return": final_r,
        "avg_steps": final_s,
        "success_rate": final_sr,
        "final": 1
    })
    print(f"[FINAL] {exp_name}: SR={final_sr:.2f}, Return={final_r:.2f}")



# ----------------------------------------------------
# 批量运行（消融）
# ----------------------------------------------------
if __name__ == "__main__":

    agents = ["dqn", "rnn_dqn", "gat_dqn", "gat_rnn_dqn"]
    envs = ["small", "medium"]
    seeds = [0, 1, 2, 3, 4]

    training_steps_cfg = {
        "small": 50000,
        "medium": 100000
    }

    for agent_name in agents:
        for env_name in envs:
            for seed in seeds:
                run_experiment(agent_name, env_name, seed,
                               training_steps=training_steps_cfg[env_name])
