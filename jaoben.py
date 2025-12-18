import os
import shutil
import argparse
from pathlib import Path

# 智能体名称映射（简化用于文件夹命名）
AGENT_MAPPING = {
    "rnn_dqn": "rnndqn",
    "gat_dqn": "gatdqn",
    "gat_rnn_dqn": "gatrnndqn"
}

# 支持的环境类型
SUPPORTED_ENVS = ["medium"]
# 原始tb_log文件夹名
ORIG_TB_DIR_NAME = "tb_log"

def copy_and_rename_tb_dirs(root_dir, target_root):
    """
    遍历目录结构，复制tb_log文件夹到目标目录并重命名文件夹
    文件夹命名规则：{env}_{agent_short}_{seed}
    内部的events.out.tfevents.xxx文件名保持不变
    """
    # 遍历环境（small/medium）
    for env in SUPPORTED_ENVS:
        env_dir = Path(root_dir) / env
        if not env_dir.exists():
            print(f"[Warning] 环境目录不存在，跳过: {env_dir}")
            continue
        
        # 目标根目录：run_small/run_medium（按环境名创建）
        target_env_root = Path(target_root) / f"run_{env}"
        target_env_root.mkdir(parents=True, exist_ok=True)
        
        # 遍历智能体
        for agent_key, agent_short in AGENT_MAPPING.items():
            agent_dir = env_dir / agent_key
            if not agent_dir.exists():
                print(f"[Warning] 智能体目录不存在，跳过: {agent_dir}")
                continue
            
            # 遍历种子（0-4）
            for seed in range(5):
                seed_dir = agent_dir / f"seed{seed}"
                orig_tb_dir = seed_dir / ORIG_TB_DIR_NAME
                
                # 检查原始tb_log目录是否存在且非空
                if not orig_tb_dir.exists():
                    print(f"[Warning] 原始tb_log目录不存在，跳过: {orig_tb_dir}")
                    continue
                if not list(orig_tb_dir.glob("*")):
                    print(f"[Warning] tb_log目录为空，跳过: {orig_tb_dir}")
                    continue
                
                # 生成新文件夹名称
                new_tb_dir_name = f"{env}_{agent_short}_{seed}"
                # 目标文件夹路径（run_small/xxx 下的新命名文件夹）
                target_tb_dir = target_env_root / new_tb_dir_name
                
                # 如果目标文件夹已存在，先删除（避免覆盖冲突）
                if target_tb_dir.exists():
                    shutil.rmtree(target_tb_dir)
                    print(f"[Info] 已删除已存在的目标文件夹: {target_tb_dir}")
                
                # 复制整个tb_log文件夹（重命名），保留内部文件不变
                try:
                    shutil.copytree(orig_tb_dir, target_tb_dir)
                    print(f"[Success] 复制完成: {orig_tb_dir} -> {target_tb_dir}")
                    print(f"         内部文件：{list(target_tb_dir.glob('*'))[0].name}（文件名未修改）")
                except Exception as e:
                    print(f"[Error] 复制失败 {orig_tb_dir} -> {target_tb_dir}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="复制并重命名嵌套TensorBoard文件的文件夹（保留内部文件名）")
    parser.add_argument("--root-dir", type=str, default="ablation_results",
                        help="实验结果根目录（默认：ablation_results）")
    parser.add_argument("--target-dir", type=str, default=".",
                        help="目标根目录（默认：当前目录）")
    args = parser.parse_args()
    
    # 执行复制并重命名文件夹
    copy_and_rename_tb_dirs(args.root_dir, args.target_dir)
    print("\n所有操作执行完毕！")

if __name__ == "__main__":
    main()