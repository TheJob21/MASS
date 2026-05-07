# MASS

Multi-Agent Spectrum Sharing for Cognitive Radar

MASS is a research-oriented simulation framework for studying autonomous spectrum sharing in congested radio frequency (RF) environments using reinforcement learning, heuristic control methods, and evolutionary optimization techniques.

The framework models dynamic spectrum occupancy, interference, adaptive transmission behavior, and multi-agent competition within radar-relevant RF environments. Agents learn or act to maximize spectrum efficiency while minimizing collisions and interference.

---

# 🧠 Overview

The simulation compares multiple cognitive agent architectures operating in a shared RF spectrum.

Implemented agent types include:

* Static Agents — Fixed spectrum users with predefined behavior
* Random Start Agents — Baseline random transmission strategy
* SAA Agents — Sense-and-Avoid heuristic agents
* PPO Agents — Proximal Policy Optimization agents
* DQN Agents — Deep Q-Network agents
* DPG Agents — Deterministic Policy Gradient agents
* MFOS Agents — Evolutionary/meta-learning agents
* Ablated MFOS Agents — Simplified MFOS variants used for ablation studies

The environment supports both synthetic and recorded spectrum data for evaluating adaptive RF behavior under realistic interference conditions.

---

# 📡 Environment Model

The RF spectrum is represented using FFT-based frequency bins and simulated over discrete time intervals.

Core environment characteristics:

* FFT-based spectrum representation (`FFT_SIZE = 1024`)
* CPI/PRI radar timing structure
* Dynamic and static occupancy generation
* Collision tracking via spectrum ownership
* Multi-agent simultaneous transmission support
* Real or synthetic spectrum occupancy modeling

Key concepts:

* **Occupancy state** — Boolean map representing active spectrum usage
* **Bin ownership** — Tracks which agent controls each frequency bin
* **Collisions** — Overlapping or invalid spectrum transmissions
* **Dead space** — Unused spectrum available for exploitation

---

# 🧮 Reward Function

Agents are rewarded based on:

* Successful transmission bandwidth
* Collision avoidance
* Spectrum efficiency
* Stable bandwidth selection
* Stable center frequency selection
* Dead-space utilization efficiency

The reward function is configurable through `config.py`.

Important reward parameters include:

* `collision_ratio`
* `beta`
* `transmission_weight`
* `collision_weight`
* `bandwidth_distortion`
* `center_distortion`
* `deadspace_penalty_scale`

---

# 🧠 Learning Algorithms

Implemented learning frameworks include:

## PPO

* On-policy actor-critic reinforcement learning
* Generalized Advantage Estimation (GAE)
* Clipped policy updates

## DQN

* Value-based reinforcement learning
* Experience replay memory
* Target network updates

## DPG

* Deterministic policy gradient optimization

## MFOS

* Evolutionary/meta-learning optimization
* Population-based policy mutation
* Elite selection and exploration

## Rule-Based Agents

* SAA heuristic control
* Random baseline policies

---

# 📊 Outputs

The framework produces multiple forms of evaluation data.

## 1. Spectrum Visualization

* Time vs frequency occupancy maps
* Agent ownership visualization
* Collision highlighting
* Spectrum utilization analysis

## 2. Performance Metrics

Tracked metrics include:

* Reward over time
* Collision rate
* Bandwidth usage
* Center frequency drift (`ΔCF`)
* Bandwidth adaptation (`ΔBW`)

## 3. Evaluation Summary

Evaluation summaries are exported to Excel files.

Example:

```text
./Output/agent_eval_summarySimMulti.xlsx
```

Reported metrics include:

* Average reward
* Reward variance
* Collision statistics
* Bandwidth utilization
* Frequency stability metrics

---

# ⚙️ Configuration

All runtime configuration is controlled through:

```text
config.py
```

The main execution script (`main.py`) does not accept command-line arguments.

Key configurable settings include:

## General Execution

```python
SIM_MODE = True # True if no live data is to be used
MULTI_AGENT = True # Determines if the agents will be run against each other or just against data
EVAL_MODE = False # Always leave this false, set true automatically when 80% of iterations have passed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42069
```

## Spectrum Selection

```python
DATA_CHOICE = "245"
DATA_CHOICE = "u245"
DATA_CHOICE = "264"
DATA_CHOICE = "u264"
```

Provided spectrum datasets:

* `245` — 1M snapshots recorded at ARRC at OU in 2.4–2.5 GHz band using X310
* `u245` — 4.1M snapshots recorded at Oklahoma Memorial Union in 2.4–2.5 GHz band using X310
* `264` — 1M snapshots recorded at ARRC at OU in 2.59–2.69 GHz band using X310
* `u264` — 2.4M snapshots recorded at Oklahoma Memorial Union in 2.59–2.69 GHz band using X310

## Agent Counts

```python
AGENTS = {
    "static": { # Static agents provide simulation data
        "fat": 3, # wide bandwidth occupancy with large gaps between transmissions
        "skinny": 4, # narrow bandwidth occupancy with equal resting vs transmission
        "pulsed": 5, # narrow bandwidth occupancy that pulses for a long period then waits a long period
        "rectangular": 0 # wide bandwidth occupancy with less modulation for short time and random new transmission locations. Good for simulating 2.59-2.69 GHz bandwidth
    },
    "random_start": 0,
    "saa": 0,
    "ppo": 1,
    "dqn": 1,
    "mfos": 1,
    "dpg": 0,
    "ablated_mfos": 0
}
```

## Runtime Parameters

```python
ITERATIONS = 1_000_000 # Overridden when live data used. This is only used for Sim
SPECTRUM_SAMPLE_SIZE = 30_000
PRINT_INTERVAL = 100_000
TIMESTEP_US = 10.24
```

---

# 📁 Project Structure

```text
MASS/
│
├── Agents/
│   ├── PPO/
│   ├── DQN/
│   ├── DPG/
│   ├── MFOS/
│   └── Control/
│
├── Data/
│   ├── spectrum_245ghz.dat
│   ├── union_spectrum_245ghz.dat
│   ├── spectrum_264ghz.dat
│   └── union_spectrum_264ghz.dat
│
├── DataVisualization/
│
├── Output/
│
├── config.py
├── environment.py
├── rewards.py
├── signal_processing.py
└── main.py
```

---

# ▶️ Running the Simulation

Run the simulation with:

```bash
python main.py
```

All simulation parameters are configured through `config.py`.

---

# 📦 Dependencies

Install required Python packages:

```bash
pip install numpy torch pandas matplotlib openpyxl
```

---

# 🚀 GPU Support

PyTorch-based agents support GPU acceleration when CUDA is available.

Device selection is handled automatically:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

---

# ⚠️ Notes

* This framework is intended for research and experimentation.
* The simulation uses a centralized environment loop.
* Runtime scales significantly with the number of agents and iterations.
* Large-scale experiments may require substantial CPU/GPU resources.
* Real spectrum datasets are loaded from `.dat` and cached `.npz` files.

---

# 📚 Research Focus

MASS is designed for experimentation involving:

* Cognitive radar
* Autonomous spectrum access
* RF coexistence
* Multi-agent reinforcement learning
* Electronic warfare simulation
* Adaptive communications
* Dynamic spectrum management
