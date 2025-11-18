"""
Optuna 调参脚本 for GAT-RNN-DQN Agent
------------------------------------
搜索 GAT + GRU/LSTM DQN 智能体在 NASim 环境中的最优超参数。
"""

import optuna
import numpy as np
import torch
import nasim
from nasim.agents.gat_rnn_dqn_agent import GATRNNDQNAgent


def objective(trial):
    """Optuna 优化目标函数"""

    # 超参数搜索空间
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.99)
    gat_hidden = trial.suggest_int("gat_hidden", 64, 256, step=32)
    gat_out_dim = trial.suggest_int("gat_out_dim", 32, 128, step=16)
    rnn_hidden_dim = trial.suggest_int("rnn_hidden_dim", 32, 128, step=16)
    rnn_num_layers = trial.suggest_int("rnn_num_layers", 1, 2)
    rnn_dropout = trial.suggest_float("rnn_dropout", 0.0, 0.3)
    seq_len = trial.suggest_int("seq_len", 3, 6)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    final_epsilon = trial.suggest_float("final_epsilon", 0.01, 0.1)
    rnn_type = trial.suggest_categorical("rnn_type", ["GRU", "LSTM"])

    # 构建 tiny 环境
    env = nasim.make_benchmark("tiny", seed=0, fully_obs=False, flat_actions=True, flat_obs=True)

    # 初始化智能体
    agent = GATRNNDQNAgent(
        env=env,
        lr=lr,
        gamma=gamma,
        batch_size=batch_size,
        gat_hidden=gat_hidden,
        gat_out_dim=gat_out_dim,
        rnn_hidden_dim=rnn_hidden_dim,
        rnn_num_layers=rnn_num_layers,
        rnn_dropout=rnn_dropout,
        rnn_type=rnn_type,
        seq_len=seq_len,
        final_epsilon=final_epsilon,
        training_steps=10000,  # 调参阶段快速训练
        verbose=False
    )

    # 简短训练
    try:
        agent.train()
        ep_ret, steps, goal = agent.run_eval_episode()
        score = ep_ret + (50 if goal else 0)
    except Exception as e:
        print(f"[ERROR] Trial failed: {e}")
        return -9999.0

    return score


if __name__ == "__main__":
    import os
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    study_name = "gat_rnn_dqn_optuna"
    storage_name = f"sqlite:///{study_name}.db"  # 保存结果到 SQLite
    study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage_name, load_if_exists=True)

    print("启动调参...")
    study.optimize(objective, n_trials=20, timeout=None)

    print("\n调参完成！最优参数如下：")
    best_params = study.best_params
    for k, v in best_params.items():
        print(f"{k}: {v}")

    print(f"\nBest reward: {study.best_value:.2f}")
