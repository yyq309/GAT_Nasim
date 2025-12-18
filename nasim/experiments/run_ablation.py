import os
import csv
import subprocess
import argparse
from tqdm import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 四个 agent module
AGENTS = {
    "gat_rnn_dqn":"nasim.agents.reward_gat_rnn_dqn_agent"
}


SEEDS = [0,1,2,3,4]

TRAIN_STEPS = {
    "small": 50000,
    "medium": 100000
}

# TensorBoard scalar 名称（和你原代码完全一致）
SCALARS = [
    "episode_return",
    "episode_steps",
    "episode_goal_reached",
    "q_value_mean",
    "loss",
]

def read_tensorboard_scalars(log_dir):
    """
    解析 TensorBoard event 文件，将需要的 scalar 全部读成字典
    """

    # 自动查找 log_dir 下最新的子文件夹（即实际 event 文件所在目录）
    if os.path.isdir(log_dir):
        subdirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
        if subdirs:
            # 取最新的子文件夹
            subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            log_dir = subdirs[0]

    ea = EventAccumulator(log_dir)
    ea.Reload()

    data = {}

    for tag in SCALARS:
        if tag not in ea.Tags().get("scalars", []):
            continue

        events = ea.Scalars(tag)
        data[tag] = [(e.step, e.value) for e in events]

    return data

def write_csv(path, rows, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

def run_one_experiment(env, agent_key, seed, save_root):

    # 保存路径
    save_dir = os.path.join(save_root, env, agent_key, f"seed{seed}")
    os.makedirs(save_dir, exist_ok=True)

    tb_log_dir = os.path.join(save_dir, "tb_log")
    os.makedirs(tb_log_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, "train_log.csv")
    if os.path.exists(csv_path):
        print(f"[Skip] exists: {csv_path}")
        return

    # 运行 agent
    cmd = [
        "python", "-m", AGENTS[agent_key],
        env,
        "--training_steps", str(TRAIN_STEPS[env]),
        "--seed", str(seed),
    ]

    # 覆盖默认 TensorBoard logdir
    env_vars = os.environ.copy()
    env_vars["TORCH_TB_DIR"] = tb_log_dir   # 你需要在 agent 中改两行（已告诉你怎么做）

    subprocess.run(cmd, env=env_vars)

    # 读取 TensorBoard Scalars
    scalar_data = read_tensorboard_scalars(tb_log_dir)

    # 整理为 CSV 行
    rows = []
    # 使用 episode_return 作为主时间线
    steps = [s for s, _ in scalar_data["episode_return"]]

    for i, step in enumerate(steps):
        row = [i+1]  # episode index
        for tag in SCALARS:
            if tag in scalar_data and i < len(scalar_data[tag]):
                row.append(scalar_data[tag][i][1])
            else:
                row.append("")
        rows.append(row)

    write_csv(
        csv_path,
        rows,
        header=["episode"] + SCALARS
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="small",
                        choices=["small", "medium"])
    parser.add_argument("--save-dir", type=str, default="ablation_results")
    args = parser.parse_args()

    total = len(AGENTS) * len(SEEDS)
    pbar = tqdm(total=total, ncols=120)

    for agent in AGENTS:
        for seed in SEEDS:
            run_one_experiment(args.env, agent, seed, args.save_dir)
            pbar.update(1)

    pbar.close()
    print("\nAblation study finished!")

if __name__ == "__main__":
    main()
