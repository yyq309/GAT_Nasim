import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# ===============================================
# 你实验中的 4 个模型组合
# ===============================================
MODEL_CONFIGS = [
    ("gat", "gru"),
    ("gat", "lstm"),
    ("vector", "gru"),
    ("vector", "lstm"),
]

# 显示名称
MODEL_NAMES = {
    ("gat", "gru"): "GAT+GRU",
    ("gat", "lstm"): "GAT+LSTM",
    ("vector", "gru"): "Vector+GRU",
    ("vector", "lstm"): "Vector+LSTM",
}

# 环境列表
ENV_LIST = ["small", "medium"]

# ===============================================
# 帮助函数：读取 train_log.csv
# ===============================================
def load_train_log(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df


# ===============================================
# 聚合 seed 的数据
# ===============================================
def aggregate_train_metrics(env, model, base_dir):
    att, rnn = model
    folder = os.path.join(base_dir, env, f"{att}_{rnn}")

    all_success_rates = []
    all_steps_to_goal = []
    all_returns_curve = []

    seed_dirs = [d for d in os.listdir(folder) if d.startswith("seed")]

    for sd in seed_dirs:
        log_file = os.path.join(folder, sd, "train_log.csv")
        df = load_train_log(log_file)
        if df is None:
            continue

        # 成功率（训练中 goal==1 的比例）
        success_rate = df["goal"].mean()
        all_success_rates.append(success_rate)

        # steps-to-goal（仅统计成功 episode）
        success_steps = df[df["goal"] == 1]["steps"]
        if len(success_steps) > 0:
            all_steps_to_goal.append(success_steps.mean())

        # 训练曲线（return）
        all_returns_curve.append(df["return"].values)

    return {
        "success_rates": np.array(all_success_rates),
        "steps_to_goal": np.array(all_steps_to_goal),
        "return_curves": all_returns_curve,
    }


# ===============================================
# 图 1: Success Rate（柱状图 + 标准差误差棒）
# ===============================================
def plot_success_rate(results, env, save_dir):

    fig, ax = plt.subplots(figsize=(8, 6))

    means = []
    stds = []
    labels = []

    for model in MODEL_CONFIGS:
        data = results[model]["success_rates"]
        means.append(data.mean())
        stds.append(data.std())
        labels.append(MODEL_NAMES[model])

    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Training Success Rate")
    ax.set_title(f"{env.upper()} - Training Success Rate (Replaceability)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{env}_success_rate.png"))
    plt.close()


# ===============================================
# 图 2: Steps-to-Goal（成功 episode）
# ===============================================
def plot_steps_to_goal(results, env, save_dir):

    fig, ax = plt.subplots(figsize=(8, 6))

    means = []
    stds = []
    labels = []

    for model in MODEL_CONFIGS:
        data = results[model]["steps_to_goal"]
        means.append(data.mean())
        stds.append(data.std())
        labels.append(MODEL_NAMES[model])

    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Avg Steps (successful episodes)")
    ax.set_title(f"{env.upper()} - Steps to Goal")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{env}_steps_to_goal.png"))
    plt.close()


# ===============================================
# 图 3: Learning Curves（多 seed 平均 + std）
# ===============================================
def plot_learning_curves(results, env, save_dir):

    fig, ax = plt.subplots(figsize=(8, 6))

    for model in MODEL_CONFIGS:
        curves = results[model]["return_curves"]
        min_len = min([len(c) for c in curves])
        curves = [c[:min_len] for c in curves]

        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)
        episodes = np.arange(min_len)

        ax.plot(episodes, mean_curve, label=MODEL_NAMES[model])
        ax.fill_between(episodes,
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        alpha=0.2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(f"{env.upper()} - Training Learning Curves")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{env}_learning_curve.png"))
    plt.close()


# ===============================================
# 主函数：读取结果并生成图表
# ===============================================
def main():
    base_dir = "replacement_results"
    save_dir = "replacement_plots"
    os.makedirs(save_dir, exist_ok=True)

    print("\n===== Generating Replaceability Experiment Plots =====")

    for env in ENV_LIST:
        print(f"\nProcessing environment: {env}")

        results = {}

        # 读取所有模型组合的结果
        for model in MODEL_CONFIGS:
            results[model] = aggregate_train_metrics(env, model, base_dir)

        # 绘图
        plot_success_rate(results, env, save_dir)
        plot_steps_to_goal(results, env, save_dir)
        plot_learning_curves(results, env, save_dir)

        print(f"Saved plots for {env} → {save_dir}")

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
