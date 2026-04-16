# CrowdNav

CrowdNav is a research framework for training and evaluating socially-aware robot navigation policies in crowded environments. A robot learns to navigate to a goal among multiple humans (controlled by unknown policies such as ORCA) while avoiding collisions and behaving in a socially acceptable way.

The project bundles an OpenAI Gym simulation environment (`crowd_sim/`) with a set of learning/classical navigation policies (`crowd_nav/`), and includes a vendored copy of the Python-RVO2 library used for human motion simulation.

## Project Layout

```
crowdnav-dip/
├── crowd_sim/           # OpenAI Gym simulation environment (humans + robot)
│   └── envs/
├── crowd_nav/           # Policies, training, and testing code
│   ├── configs/         # env.config, policy.config, train.config
│   ├── policy/          # cadrl, lstm_rl, sarl, multi_human_rl
│   ├── utils/           # trainer, explorer, replay memory
│   ├── data/            # trained models and outputs (e.g. data/output_trained)
│   ├── train.py
│   └── test.py
├── Python-RVO2-main/    # Vendored Python-RVO2 library (ORCA solver bindings)
└── setup.py
```

## Available Policies

Implemented in [crowd_nav/policy/](crowd_nav/policy) — see [crowd_sim/README.md](crowd_sim/README.md) for more detail.

- **ORCA** — reciprocal, collision-free velocity (classical baseline)
- **CADRL** — value network predicting action for the most important human
- **LSTM-RL** — LSTM encoder over human states
- **SARL** — pairwise interaction module + self-attention over humans
- **OM-SARL** — SARL extended with a local occupancy map for intra-human interaction

## Setup

> All commands assume the `navigate` conda environment is active after step 1.

0. **System packages** (skip if using the school's server)

   ```bash
   sudo apt update && sudo apt install cmake build-essential ffmpeg
   ```

1. **Create the conda environment** — install Anaconda first from https://www.anaconda.com/download/success

   ```bash
   conda create -n navigate python=3.10
   conda activate navigate
   pip install cython
   ```

2. **Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2)** (vendored in `Python-RVO2-main/`)

   ```bash
   cd Python-RVO2-main
   python setup.py build && python setup.py install
   cd ..
   ```

3. **Install `crowd_sim` and `crowd_nav`** — run from the `crowdnav-dip/` project root

   ```bash
   pip install -e .
   pip install gym==0.15.7
   ```

## Getting Started

The repository is organized in two parts: `crowd_sim/` contains the simulation environment and `crowd_nav/` contains the training and testing code. Simulation framework details: [crowd_sim/README.md](crowd_sim/README.md). Commands below should be run from inside `crowd_nav/`.

### Visualize a test case

```bash
cd crowd_nav
python test.py --policy sarl --model_dir data/output_trained --phase test --visualize --test_case 0
```

Useful `test.py` flags (see [crowd_nav/test.py](crowd_nav/test.py)):

- `--policy {orca, cadrl, lstm_rl, sarl}` — policy to evaluate
- `--model_dir <path>` — directory containing `rl_model.pth` and config files
- `--phase {train, val, test}` — dataset phase
- `--test_case <int>` — specific scenario id to visualize
- `--visualize` — show an animated rollout
- `--video_file <path>` — save the rollout to a video file
- `--traj` — overlay trajectories
- `--gpu` — run on GPU

### Train a policy

```bash
cd crowd_nav
python train.py --policy sarl
```

Useful `train.py` flags (see [crowd_nav/train.py](crowd_nav/train.py)):

- `--policy {cadrl, lstm_rl, sarl}` — policy to train
- `--output_dir <path>` — where checkpoints and logs are written (default `data/output`)
- `--resume` — resume from a previous run in `--output_dir`
- `--gpu` — train on GPU
- `--debug` — enable debug logging

Training parameters live in [crowd_nav/configs/train.config](crowd_nav/configs/train.config); environment parameters (number of humans, scenario type, reward shaping) in [crowd_nav/configs/env.config](crowd_nav/configs/env.config); policy hyperparameters in [crowd_nav/configs/policy.config](crowd_nav/configs/policy.config).

## Troubleshooting

- **`Python-RVO2` build fails** — make sure `cython` is installed in the active env and `cmake`/`build-essential` are on the system.
- **`gym` API errors** — this project pins `gym==0.15.7`; newer versions break the env interface.
- **CUDA/GPU** — omit `--gpu` to run on CPU; all scripts work without a GPU.
