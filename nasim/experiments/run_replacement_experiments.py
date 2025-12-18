import os
import csv
from tqdm import tqdm
import argparse
import numpy as np
import nasim
import torch

from nasim.agents.HippocampusAgent import HippocampusAgent


def setup_gpu_device():
    """配置 GPU 设备，优先使用第二块显卡（GPU 1），兼容单 GPU/无 GPU 场景"""
    # 检查可用 GPU 数量
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count >= 2:
            # 优先使用第二块显卡（索引为 1，第一块是 0）
            device = torch.device("cuda:1")
            torch.cuda.set_device(device)  # 设置全局默认 GPU
            print(f"使用第二块显卡 (cuda:1)，GPU 名称: {torch.cuda.get_device_name(1)}")
        elif gpu_count == 1:
            # 只有一块显卡时使用 GPU 0
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
            print(f"仅检测到一块显卡，使用 cuda:0，GPU 名称: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("未检测到 GPU，使用 CPU 运行")
    else:
        device = torch.device("cpu")
        print("CUDA 不可用，使用 CPU 运行")
    return device

# 初始化 GPU 设备（程序启动时立即执行）
DEVICE = setup_gpu_device()


# ===============================================
# 自动创建目录
# ===============================================
def write_csv(path, rows, header=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(rows)


# ===============================================
# 四组可替代模块
# ===============================================
MODEL_CONFIGS = [
    ("gat", "gru"),
    ("gat", "lstm"),
    ("vector", "gru"),
    ("vector", "lstm"),
]

# 固定的 seed 列表
SEED_LIST = [0, 1, 2, 3, 4]

# ===============================================
# 环境专属配置（训练步数 + 奖励函数参数）
# ===============================================
ENV_CONFIGS = {
    "small": {
        "training_steps": 50000,
        # small环境奖励配置：适度探索 + 中等成功奖励
        "reward_config": {
            "success_bonus": 80.0,
            "step_penalty": -0.12,
            "info_gain_bonus": 0.6
        }
    },
    "medium": {
        "training_steps": 100000,
        # medium环境奖励配置：高成功奖励 + 低步惩罚 + 强探索激励
        "reward_config": {
            "success_bonus": 150.0,
            "step_penalty": -0.07,
            "info_gain_bonus": 1.0
        }
    }
}


# ===============================================
# 单次实验（适配环境专属奖励函数 + 精细化进度条 + GPU 适配）
# ===============================================
def run_one_experiment(env_name, att_type, rnn_type, seed, base_dir):
    # 保存路径： replacement_experiment_results/env/att_rnn/seedX
    save_dir = os.path.join(
        base_dir, env_name, f"{att_type}_{rnn_type}", f"seed{seed}"
    )
    os.makedirs(save_dir, exist_ok=True)

    eval_path = os.path.join(save_dir, "eval_log.csv")
    if os.path.exists(eval_path):
        print(f"\n[Skip] Already completed: {save_dir}")
        return

    # 获取环境专属配置（训练步数 + 奖励参数）
    env_config = ENV_CONFIGS.get(env_name, ENV_CONFIGS["small"])
    train_steps = env_config["training_steps"]
    reward_config = env_config["reward_config"]

    # 创建环境
    env = nasim.make_benchmark(
        env_name,
        seed=seed,
        fully_obs=False,
        flat_actions=True,
        flat_obs=False,
    )

    # 创建 agent（传入环境专属奖励参数）
    agent = HippocampusAgent(
        env,
        att_type=att_type,
        rnn_type=rnn_type,
        training_steps=train_steps,
        batch_size=128,
        verbose=False,
        seed=seed,
        # 环境专属奖励函数参数
        success_bonus=reward_config["success_bonus"],
        step_penalty=reward_config["step_penalty"],
        info_gain_bonus=reward_config["info_gain_bonus"]
    )

    # 核心改动2：将 agent 的模型强制移到指定 GPU（兜底保障）
    if hasattr(agent, 'model') and torch.cuda.is_available():
        agent.model = agent.model.to(DEVICE)
        print(f"将 {att_type}+{rnn_type} 模型移到 {DEVICE}")

    # 训练进度条（增加描述信息，展示奖励配置）
    pbar_desc = (
        f"{env_name.upper()}-{att_type.upper()}-{rnn_type.upper()}-seed{seed} | "
        f"bonus={reward_config['success_bonus']} | penalty={reward_config['step_penalty']}"
    )
    pbar = tqdm(
        total=train_steps,
        desc=pbar_desc,
        leave=True,  # 保留进度条，便于查看历史任务
        ncols=120,   # 加宽进度条，完整展示描述
        unit="step"  # 单位标注为step
    )

    last_steps = 0
    train_rows = []

    # Training Loop
    while agent.steps_done < agent.training_steps:
        ep_ret, ep_steps, goal = agent.run_train_episode()

        train_rows.append([agent.episodes, ep_ret, ep_steps, int(goal)])

        # 更新进度条（按实际完成步数更新）
        current_steps = agent.steps_done
        pbar.update(current_steps - last_steps)
        last_steps = current_steps

        # 进度条动态展示关键指标
        pbar.set_postfix({
            "ep": agent.episodes,
            "last_ret": f"{ep_ret:.1f}",
            "success": f"{int(goal)}",
            "eps": f"{agent.epsilon():.3f}",
            "gpu": DEVICE  # 新增：进度条展示当前使用的 GPU/CPU
        })

    pbar.close()

    # 保存训练日志
    write_csv(
        os.path.join(save_dir, "train_log.csv"),
        train_rows,
        header=["episode", "return", "steps", "goal"],
    )

    # =============================
    #   Evaluation（带进度条）
    # =============================
    eval_pbar = tqdm(
        total=30,
        desc=f"Evaluating {env_name}-{att_type}-{rnn_type}-seed{seed}",
        leave=False,
        ncols=120,
        unit="ep"
    )

    eval_stats = agent.evaluate(n_episodes=30)
    eval_pbar.update(30)
    eval_pbar.close()

    eval_rows = []
    for i in range(30):
        eval_rows.append([
            i,
            eval_stats["returns"][i],
            eval_stats["steps"][i],
            eval_stats["goals"][i],
        ])

    write_csv(
        eval_path,
        eval_rows,
        header=["episode", "return", "steps", "goal"],
    )

    # 计算评估指标并打印
    avg_return = np.mean(eval_stats["returns"])
    success_rate = np.mean(eval_stats["goals"])
    print(f"\n[Done] {save_dir} | AvgReturn={avg_return:.2f} | SuccessRate={success_rate:.2f} | GPU={DEVICE}")


# ===============================================
# 主函数——依次运行所有可替换性实验（优化全局进度条）
# ===============================================
def main():
    # 核心改动3：启动时打印 GPU 信息（确认设备）
    print(f"\n 实验启动 - 使用设备: {DEVICE}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", default=["small", "medium"])
    parser.add_argument("--save-dir", type=str, default="replacement_results")
    args = parser.parse_args()

    base_dir = args.save_dir
    os.makedirs(base_dir, exist_ok=True)

    env_list = args.envs
    # 过滤有效环境（仅保留ENV_CONFIGS中定义的）
    env_list = [env for env in env_list if env in ENV_CONFIGS]
    if not env_list:
        print(f"Error: No valid envs! Valid envs are {list(ENV_CONFIGS.keys())}")
        return

    # 计算总任务数
    total_tasks = len(env_list) * len(MODEL_CONFIGS) * len(SEED_LIST)
    print(f"\n========= Replacement Experiment =========")
    print(f"Valid envs: {env_list}")
    print(f"Model configs: {MODEL_CONFIGS}")
    print(f"Seeds: {SEED_LIST}")
    print(f"Total tasks: {total_tasks}")
    print(f"Save dir: {base_dir}\n")

    # 全局进度条（展示整体进度）
    overall_pbar = tqdm(
        total=total_tasks,
        desc="Overall Progress",
        leave=True,
        ncols=120,
        unit="task",
        colour="green"  # 全局进度条用绿色区分
    )

    for env_name in env_list:
        print(f"\n=== Starting experiments for {env_name.upper()} environment ===")
        for (att_type, rnn_type) in MODEL_CONFIGS:
            print(f"\n--- Model: {att_type.upper()}+{rnn_type.upper()} ---")
            for seed in SEED_LIST:
                try:
                    run_one_experiment(
                        env_name=env_name,
                        att_type=att_type,
                        rnn_type=rnn_type,
                        seed=seed,
                        base_dir=base_dir
                    )
                except Exception as e:
                    print(f"\n[Error] Failed to run {env_name}-{att_type}-{rnn_type}-seed{seed}: {str(e)}")
                finally:
                    overall_pbar.update(1)
                    # 全局进度条展示当前完成的任务
                    overall_pbar.set_postfix({
                        "current_env": env_name,
                        "current_model": f"{att_type}+{rnn_type}",
                        "current_seed": seed,
                        "gpu": DEVICE  # 新增：展示当前使用的 GPU
                    })

    overall_pbar.close()
    print(f"\nAll experiments completed! Results saved to {base_dir}")
    print(f"本次实验使用设备: {DEVICE}")


if __name__ == "__main__":
    main()