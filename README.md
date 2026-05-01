# MASS
Multi Agent Spectrum Sharing for Cognitive Radar

Cognitive Spectrum Access Simulation (Multi-Agent RL)

This project simulates a dynamic radio spectrum environment where multiple cognitive agents compete for bandwidth using reinforcement learning and heuristic strategies. The system models spectrum occupancy, interference, and adaptive transmission decisions over time.

🧠 Overview

The simulation compares multiple agent types operating in a shared frequency spectrum:

Static Agents – Fixed spectrum users with predefined behavior
Random Start Agents – Baseline random transmission strategy
SAA Agents – Sense-and-Avoid heuristic agents
PPO Agents – Proximal Policy Optimization agents
DQN Agents – Deep Q-Network agents
DPG Agents – Deterministic Policy Gradient agents
MFOS Agents – Evolutionary/meta-learning agents
Ablated MFOS Agents – Simplified MFOS variants (ablation study)

Agents learn or act to maximize spectrum efficiency while minimizing collisions and interference.

📡 Environment Model

The spectrum is represented as:

FFT-based frequency bins (fftSize = 1024)
Time-stepped simulation using CPI/PRI structure
Static + dynamic occupancy model
Collision detection via bin ownership tracking

Key concepts:

Occupancy state: boolean spectrum usage map
Bin ownership: which agent controls each frequency bin
Collisions: overlapping or invalid spectrum use
Dead space: unused spectrum availability
🧮 Reward Function

Agents are rewarded based on:

Successful transmission bandwidth
Collision penalties
Spectrum efficiency (clean vs interfered transmission)
Adaptation penalties (bandwidth and center frequency deviation)
🧠 Learning Algorithms

Implemented agent frameworks include:

PPO (on-policy actor-critic)
DQN (value-based RL)
DPG (deterministic policy gradients)
MFOS (evolutionary meta-learning)
Rule-based baselines (SAA, Random)
📊 Outputs

The simulation produces:

1. Spectrum Visualization
Time vs frequency occupancy heatmap
Color-coded agent ownership
Collision highlighting
2. Performance Metrics
Reward over time
Bandwidth usage trends
Collision rates
Frequency drift (Δ center frequency)
Bandwidth adaptation (Δ bandwidth)
3. Evaluation Summary

Exported to:

agent_eval_summary.xlsx

Includes per-agent:

Average reward
Collision rate
Bandwidth usage
Frequency stability metrics
⚙️ Configuration

Key parameters:

fftSize: spectrum resolution
cpiLen, pri: radar timing structure
numAgents: per-agent-type counts
collisionWeight: interference penalty scaling
sim: toggle synthetic vs real spectrum data

Live spectrum files:

spectrum_245ghz.dat
spectrum_264ghz.dat
▶️ Running the Simulation
python main.py

(Optional dependencies)

pip install numpy torch pandas matplotlib openpyxl
📁 Project Structure (current logical view)
Agents/
    PPO/
    DQN/
    DPG/
    MFOS/
    Control/

main.py   # full simulation + evaluation pipeline
⚠️ Notes
This script is research-oriented and not optimized for production use.
All agents share a centralized environment loop.
Execution time scales heavily with number of agents and iterations.
GPU acceleration is partially used via PyTorch agents