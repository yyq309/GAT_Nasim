import csv
import torch
import os


class CSVLogger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, "w", newline="")
        self.writer = None

    def write(self, dict_data):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=dict_data.keys())
            self.writer.writeheader()
        self.writer.writerow(dict_data)
        self.file.flush()


class CheckpointManager:
    def __init__(self, dirpath):
        self.dir = dirpath
        os.makedirs(self.dir, exist_ok=True)

    def save(self, agent, name):
        path = os.path.join(self.dir, f"{name}.pt")
        torch.save({
            "encoder": getattr(agent, "encoder", None).state_dict() if hasattr(agent, "encoder") else None,
            "rnn": getattr(agent, "rnn", None).state_dict() if hasattr(agent, "rnn") else None,
            "policy_net": agent.policy_net.state_dict(),
            "target_net": agent.target_net.state_dict(),
            "steps_done": agent.steps_done,
        }, path)
