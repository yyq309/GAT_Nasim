import optuna
import nasim
import torch
import gc
import multiprocessing
from nasim.agents.rnn_dqn_agent import SequenceRNNDQNAgent


def run_trial(trial_params):
    """在独立进程中运行单个 trial，确保资源完全释放"""
    trial_number, params = trial_params

    env = nasim.make_benchmark("tiny", 0, fully_obs=False, flat_actions=True, flat_obs=True)
    try:
        agent = SequenceRNNDQNAgent(
            env=env,
            lr=params["lr"],
            seq_len=params["seq_len"],
            hidden_sizes=[params["fc1"], params["fc2"]],
            rnn_type=params["rnn_type"],
            rnn_hidden_dim=params["rnn_hidden_dim"],
            rnn_layers=params["rnn_layers"],
            rnn_dropout=params["rnn_dropout"],
            gamma=params["gamma"],
            training_steps=10000,
            verbose=False,
        )

        agent.train()
        ret, steps, goal = agent.run_eval_episode(env)
        score = ret + (100 if goal else 0)
        print(f"[Trial {trial_number}] 完成 | Ret={ret:.1f} | Goal={goal}")
        env.close()
        del agent
        torch.cuda.empty_cache()
        gc.collect()
        return score

    except Exception as e:
        print(f"[Trial {trial_number}] 出错: {e}")
        env.close()
        torch.cuda.empty_cache()
        gc.collect()
        return -9999.0


def objective(trial):
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
        "seq_len": trial.suggest_int("seq_len", 3, 10),
        "rnn_hidden_dim": trial.suggest_int("rnn_hidden_dim", 64, 256),
        "fc1": trial.suggest_int("fc1", 64, 256),
        "fc2": trial.suggest_int("fc2", 64, 256),
        "rnn_type": trial.suggest_categorical("rnn_type", ["LSTM", "GRU"]),
        "rnn_layers": trial.suggest_int("rnn_layers", 1, 2),
        "rnn_dropout": trial.suggest_float("rnn_dropout", 0.0, 0.3),
        "gamma": trial.suggest_float("gamma", 0.90, 0.99),
    }

    # ⚠️ 使用子进程运行 trial，防止主进程阻塞
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(1) as pool:
        result = pool.map(run_trial, [(trial.number, params)])
        score = result[0]
    return score


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10, gc_after_trial=True)
    print("\n✅ 最佳超参数：")
    print(study.best_params)
    print(f"最佳得分: {study.best_value:.2f}")
