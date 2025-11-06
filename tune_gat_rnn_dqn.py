import optuna
import torch
import numpy as np
import nasim
from nasim.agents.gat_rnn_dqn_agent import GATRNNDQNAgent
import warnings
warnings.filterwarnings("ignore")


def objective(trial):
    """Optuna调参目标函数"""
    # -------- 1. 采样超参数 --------
    gat_hidden = trial.suggest_int("gat_hidden", 32, 128)
    gat_heads = trial.suggest_int("gat_heads", 1, 8)
    gat_dropout = trial.suggest_float("gat_dropout", 0.0, 0.5)

    rnn_hidden_dim = trial.suggest_int("rnn_hidden_dim", 32, 128)
    rnn_num_layers = trial.suggest_int("rnn_num_layers", 1, 3)
    rnn_type = trial.suggest_categorical("rnn_type", ["LSTM", "GRU"])
    rnn_dropout = trial.suggest_float("rnn_dropout", 0.0, 0.5)

    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.99)
    batch_size = trial.suggest_int("batch_size", 32, 128)
    final_epsilon = trial.suggest_float("final_epsilon", 0.01, 0.1)
    target_update_freq = trial.suggest_int("target_update_freq", 500, 5000)

    # -------- 2. 构建环境 --------
    env = nasim.make_benchmark("tiny", fully_obs=True, flat_actions=True, flat_obs=True)

    # -------- 3. 初始化 Agent --------
    agent = GATRNNDQNAgent(
        env=env,
        gat_hidden=gat_hidden,
        gat_heads=gat_heads,
        gat_dropout=gat_dropout,
        rnn_hidden_dim=rnn_hidden_dim,
        rnn_num_layers=rnn_num_layers,
        rnn_type=rnn_type,
        rnn_dropout=rnn_dropout,
        lr=lr,
        gamma=gamma,
        batch_size=batch_size,
        final_epsilon=final_epsilon,
        target_update_freq=target_update_freq,
        training_steps=5000,  # 短训练以便快速调参
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=False
    )

    # -------- 4. 训练若干 episode --------
    avg_returns = []
    for _ in range(5):  # 训练 5 个 episode 测性能
        ep_return, _, _ = agent.run_train_episode(step_limit=500)
        avg_returns.append(ep_return)

    mean_return = np.mean(avg_returns)
    return mean_return


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        study_name="GAT_RNN_DQN_Tuning",
        storage="sqlite:///gat_rnn_dqn_tuning.db",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=20, n_jobs=1)

    print("\n=== Best trial ===")
    trial = study.best_trial
    print(f"Reward = {trial.value:.2f}")
    print("Best Params:")
    for k, v in trial.params.items():
        print(f"  {k}: {v}")
