import os
import torch
from collections import defaultdict


def save_agents(agents, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)

    counters = defaultdict(int)

    for agent in agents:
        name = agent.get_name()
        idx = counters[name]

        path = os.path.join(ckpt_dir, f"{name}_{idx}.pt")

        agent.save(path)
        print(f"[SAVE] {path}")

        counters[name] += 1


def load_agents(agents, ckpt_dir, device="cpu"):
    counters = defaultdict(int)

    for agent in agents:
        name = agent.get_name()
        idx = counters[name]

        path = os.path.join(ckpt_dir, f"{name}_{idx}.pt")

        if os.path.exists(path):
            agent.load(path, device)
            print(f"[LOAD] {path}")
        else:
            print(f"[MISS] {path}")

        counters[name] += 1