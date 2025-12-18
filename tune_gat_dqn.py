import optuna
import nasim
import torch
from nasim.agents.reward_gat_dqn_agent import GATDQNAgent
from statistics import mean

# ==========================
# 💡 调参目标函数
# ==========================
def objective(trial):
    # 创建 NASim 环境
    env = nasim.make_benchmark("medium", fully_obs=True, flat_actions=True, flat_obs=True)
    
    # ==========================
    # 📦 定义待搜索参数空间
    # ==========================
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.99)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    hidden_dim = trial.suggest_int("hidden_dim", 64, 256)
    num_heads = trial.suggest_int("num_heads", 2, 8)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    target_update_freq = trial.suggest_int("target_update_freq", 500, 5000)
    exploration_steps = trial.suggest_int("exploration_steps", 5000, 20000)
    final_epsilon = trial.suggest_float("final_epsilon", 0.01, 0.1)

    # ==========================
    # 🚀 初始化代理
    # ==========================
    agent = GATDQNAgent(
        env=env,
        lr=lr,
        gamma=gamma,
        batch_size=batch_size,
        hidden_sizes=[hidden_dim, hidden_dim],
        num_heads=num_heads,
        dropout=dropout,
        target_update_freq=target_update_freq,
        exploration_steps=exploration_steps,
        final_epsilon=final_epsilon,
        training_steps=100000,
        verbose=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # ==========================
    # 🧠 运行训练并返回评估指标
    # ==========================
    try:
        rewards = []
        for _ in range(3):  # 每组参数跑3次取平均，提高稳健性
            agent.train()
            ep_return, _, _ = agent.run_eval_episode(render=False)
            rewards.append(ep_return)
        avg_reward = mean(rewards)
    except Exception as e:
        print(f"[Trial failed] {e}")
        avg_reward = -9999  # 出错则给个很低的分

    return avg_reward


# ==========================
# 🏁 启动Optuna搜索
# ==========================
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name="GATDQN_Tuning")
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    print("\n✅ 最优超参数组合:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print(f"\n🏆 对应的平均reward: {study.best_value:.2f}")

    # 保存结果
    study.trials_dataframe().to_csv("gat_dqn_tuning_results.csv", index=False)
