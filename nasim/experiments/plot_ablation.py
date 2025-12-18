import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict
import argparse

# 修复seaborn样式兼容问题
try:
    plt.style.use("seaborn-v0_8-darkgrid")
except:
    plt.style.use("seaborn-darkgrid")

# ------------------------------
# 1. 读取 train_log.csv（修复浮点转int + 容错）
# ------------------------------
def load_train_log(csv_path):
    """读取CSV，兼容浮点型的1.0/0.0，转换为int"""
    episodes, returns, steps, goals = [], [], [], []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required_cols = ["episode", "episode_return", "episode_steps", "episode_goal_reached"]
            if not all(col in reader.fieldnames for col in required_cols):
                print(f"[Warning] CSV列名不匹配，期望: {required_cols}，实际: {reader.fieldnames}，跳过: {csv_path}")
                return None
            
            for row in reader:
                try:
                    # 核心修复：episode_goal_reached先转float再转int（兼容1.0/0.0）
                    episodes.append(int(row["episode"]))
                    returns.append(float(row["episode_return"]))
                    steps.append(float(row["episode_steps"]))
                    goals.append(int(float(row["episode_goal_reached"])))  # 关键修复
                except (ValueError, KeyError) as e:
                    # 仅打印错误行，不刷屏
                    if "1.0" not in str(e):  # 过滤已知的1.0转int错误提示
                        print(f"[Warning] 数据解析错误 {csv_path}: {e}，跳过该行")
                    continue
        
        if not episodes:
            print(f"[Warning] CSV无有效数据，跳过: {csv_path}")
            return None
            
        return {
            "episode": np.array(episodes),
            "return": np.array(returns),
            "steps": np.array(steps),
            "goal": np.array(goals)
        }
    except FileNotFoundError:
        print(f"[Warning] 文件不存在，跳过: {csv_path}")
        return None
    except Exception as e:
        print(f"[Error] 读取文件失败 {csv_path}: {e}，跳过")
        return None


# ------------------------------
# 2. 平滑曲线 + 统一数组长度（核心修复维度不一致）
# ------------------------------
def smooth(y, weight=0.9):
    if len(y) == 0:
        return np.array([])
    smoothed = []
    last = y[0]
    for val in y:
        last = last * weight + (1 - weight) * val
        smoothed.append(last)
    return np.array(smoothed)

def align_curve_length(curves, max_length=None):
    """
    统一所有曲线的长度：短曲线补最后一个值，长曲线截断
    :param curves: 多个seed的曲线列表（不同长度）
    :param max_length: 目标长度（默认取所有曲线的最小长度）
    :return: 长度统一的曲线列表
    """
    if not curves:
        return []
    
    # 取所有曲线的最小长度作为统一长度（避免截断过多有效数据）
    if max_length is None:
        max_length = min(len(c) for c in curves)
    
    aligned_curves = []
    for curve in curves:
        if len(curve) >= max_length:
            # 长曲线截断到max_length
            aligned = curve[:max_length]
        else:
            # 短曲线补最后一个值到max_length
            pad_value = curve[-1] if len(curve) > 0 else 0.0
            pad_length = max_length - len(curve)
            aligned = np.pad(curve, (0, pad_length), mode='constant', constant_values=pad_value)
        aligned_curves.append(aligned)
    
    return np.array(aligned_curves)


# ------------------------------
# 3. 汇总多 seed 数据
# ------------------------------
def gather_results(root, env="small"):
    models = ["dqn", "rnn_dqn", "gat_dqn", "gat_rnn_dqn"]
    results = defaultdict(lambda: defaultdict(list))

    for model in models:
        model_path = os.path.join(root, env, model)
        if not os.path.exists(model_path):
            print(f"[Warning] 模型目录不存在，跳过: {model_path}")
            continue
        
        seed_dirs = sorted(glob(os.path.join(model_path, "seed*")))
        if not seed_dirs:
            print(f"[Warning] 未找到seed目录，跳过模型: {model_path}")
            continue
        
        for seed_dir in seed_dirs:
            if not os.path.isdir(seed_dir):
                print(f"[Warning] 非目录，跳过: {seed_dir}")
                continue
            
            csv_path = os.path.join(seed_dir, "train_log.csv")
            log_data = load_train_log(csv_path)
            
            if log_data is not None:
                results[model]["episodes"].append(log_data["episode"])
                results[model]["returns"].append(log_data["return"])
                results[model]["steps"].append(log_data["steps"])
                results[model]["goals"].append(log_data["goal"])
            else:
                print(f"[Warning] 无效数据，跳过seed: {seed_dir}")
        
        if not results[model]["episodes"]:
            print(f"[Warning] 模型 {model} 无有效seed数据，从结果中移除")
            del results[model]

    return results


# ------------------------------
# 4. 绘制消融实验图（修复维度不一致 + 统一长度）
# ------------------------------
def plot_ablation(all_results, save_dir):
    if not all_results:
        print("[Error] 无有效实验数据，无法绘图")
        return
    
    os.makedirs(save_dir, exist_ok=True)

    model_names = {
        "dqn": "DQN",
        "rnn_dqn": "RNN + DQN",
        "gat_dqn": "GAT + DQN",
        "gat_rnn_dqn": "GAT + RNN + DQN"
    }

    colors = {
        "dqn": "#1f77b4",
        "rnn_dqn": "#9467bd",
        "gat_dqn": "#ff7f0e",
        "gat_rnn_dqn": "#2ca02c"
    }

    # ---------- 图 1：Episode Return ----------
    plt.figure(figsize=(8,5))
    has_data = False
    for model, data in all_results.items():
        if data["returns"]:
            # 步骤1：平滑每个seed的曲线
            returns_smoothed = [smooth(r) for r in data["returns"] if len(r) > 0]
            if not returns_smoothed:
                continue
            # 步骤2：统一所有seed曲线的长度
            returns_aligned = align_curve_length(returns_smoothed)
            if len(returns_aligned) == 0:
                continue
            # 步骤3：计算均值（此时维度一致）
            mean_curve = np.mean(returns_aligned, axis=0)
            plt.plot(mean_curve, label=model_names.get(model, model), 
                     color=colors.get(model, "#888888"))
            has_data = True
    if has_data:
        plt.xlabel("Episodes")
        plt.ylabel("Episode Return")
        plt.title("Ablation Study - Episode Return")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "ablation_return.png"), dpi=300)
    else:
        print("[Warning] 无Return数据，跳过绘制return图")
    plt.close()

    # ---------- 图 2：Success Rate ----------
    plt.figure(figsize=(8,5))
    has_data = False
    for model, data in all_results.items():
        if data["goals"]:
            goals_smoothed = [smooth(g, weight=0.95) for g in data["goals"] if len(g) > 0]
            if not goals_smoothed:
                continue
            goals_aligned = align_curve_length(goals_smoothed)
            if len(goals_aligned) == 0:
                continue
            mean_curve = np.mean(goals_aligned, axis=0)
            plt.plot(mean_curve, label=model_names.get(model, model), 
                     color=colors.get(model, "#888888"))
            has_data = True
    if has_data:
        plt.xlabel("Episodes")
        plt.ylabel("Success Rate")
        plt.title("Ablation Study - Success Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "ablation_success_rate.png"), dpi=300)
    else:
        print("[Warning] 无Success Rate数据，跳过绘制成功率图")
    plt.close()

    # ---------- 图 3：Episode Steps ----------
    plt.figure(figsize=(8,5))
    has_data = False
    for model, data in all_results.items():
        if data["steps"]:
            steps_smoothed = [smooth(s) for s in data["steps"] if len(s) > 0]
            if not steps_smoothed:
                continue
            steps_aligned = align_curve_length(steps_smoothed)
            if len(steps_aligned) == 0:
                continue
            mean_curve = np.mean(steps_aligned, axis=0)
            plt.plot(mean_curve, label=model_names.get(model, model), 
                     color=colors.get(model, "#888888"))
            has_data = True
    if has_data:
        plt.xlabel("Episodes")
        plt.ylabel("Episode Steps")
        plt.title("Ablation Study - Episode Length / Steps")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "ablation_steps.png"), dpi=300)
    else:
        print("[Warning] 无Steps数据，跳过绘制步数图")
    plt.close()

    # ---------- 图 4：最终性能柱状图 ----------
    plt.figure(figsize=(8,5))
    final_returns = []
    final_success = []
    labels = []

    for model, data in all_results.items():
        returns_last = []
        for r in data["returns"]:
            if len(r) > 0:
                returns_last.append(r[-1])
        goals_last = []
        for g in data["goals"]:
            if len(g) > 0:
                goals_last.append(g[-1])
        
        if returns_last and goals_last:
            final_returns.append(np.mean(returns_last))
            final_success.append(np.mean(goals_last))
            labels.append(model_names.get(model, model))

    if labels:
        x = np.arange(len(labels))
        width = 0.35

        plt.bar(x - width/2, final_returns, width, label="Return")
        plt.bar(x + width/2, final_success, width, label="Success Rate")
        plt.xticks(x, labels, rotation=20, ha="right")
        plt.title("Final Performance Comparison (Ablation Study)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "ablation_final_bar.png"), dpi=300)
    else:
        print("[Warning] 无最终性能数据，跳过绘制柱状图")
    plt.close()

    print(f"\n所有可用图表已保存到: {save_dir}")


# ------------------------------
# 主函数
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="ablation_results")
    parser.add_argument("--env", type=str, default="small")
    parser.add_argument("--save-dir", type=str, default="ablation_plots")
    args = parser.parse_args()

    print(f"开始读取数据，根目录: {args.root}, 环境: {args.env}")
    results = gather_results(args.root, args.env)
    
    print(f"\n有效模型数量: {len(results)}")
    for model in results:
        seed_count = len(results[model]["episodes"])
        print(f"  - {model}: {seed_count} 个有效seed")
    
    plot_ablation(results, args.save_dir)