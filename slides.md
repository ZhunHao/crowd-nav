# DIP: Ship Autonomous Navigation — Project Briefing

**Presenter:** Huang Erdong
**Supervisor:** Assoc Prof Jiang Xudong
**Institution:** Centre for Advanced Robotics Technology Innovation
**Date:** 13 Aug 2025

> Cleaned, full-content transcript of the professor's 17-slide briefing.
> Drives the implementation plan (`~/.claude/plans/added-slides-md-which-is-logical-grove.md`),
> the system design in [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md),
> and the work-package breakdown in [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md).

---

## Slide 1 — Title

- DIP: Ship Autonomous Navigation
- Presenter: Huang Erdong
- Supervisor: Assoc Prof Jiang Xudong
- Centre for Advanced Robotics Technology Innovation
- 13-Aug-2025

---

## Slide 2 — Project background: maritime path planning

- Path planning is a **non-deterministic polynomial-time (NP) hard** problem: find a continuous path connecting a system from an initial to a final goal configuration.
- Domain scope: **maritime navigation**.
- Challenges unique to the marine environment:
  - Dynamic environment with moving obstacles and complex environmental factors.
  - Ship turning restrictions — speed, angle, acceleration, etc.
  - Traffic separation rules (**COLREGs** — Convention on the International Regulations for Preventing Collisions at Sea).
- Reference: Qiao, Yuanyuan, et al. *"Survey of deep learning for autonomous surface vehicles in marine environments."* IEEE Transactions on Intelligent Transportation Systems **24.4 (2023): 3678–3701**.

---

## Slide 3 — Data: AIS (Automatic Identification System)

- AIS data used for maritime navigation research.
- **239 files** — one per day, covering **1 Jan 2023 → 30 Sep 2023**. Each set contains all ships' information for one day.
- **Over 8 million records** in total.
- Projections:
  - Mercator Projection.
  - UTM (Universal Transverse Mercator Grid System).

---

## Slide 4 — Path planning algorithm: A\*

**Initialize:**
- Start Node — starting point of the path.
- Goal Node — destination point.
- Open List — priority queue of nodes to be evaluated.
- Closed List — set of nodes already evaluated.

**While Open List is not empty:**
- Select the node with the lowest `f(x) = g(x) + h(x)` value from the Open List.
- Expand the selected node; add its neighbours to the Open List.
- Calculate `f(x)`, `g(x)`, `h(x)` values for each neighbour.
- Update the parent of each neighbour to the selected node.
- Move the selected node to the Closed List.

**Path Reconstruction:**
- Once the Goal Node is reached, trace back through parent nodes to find the optimal path.

---

## Slide 5 — Path planning algorithm: Potential Field

**Initialize:**
- Initialize the navigation environment.
- Start and goal points `qs`, `qg`.
- Potential field gain `ka`, obstacle factor `kr`.
- Calculate attractive and repulsive fields `Ua`, `Ur`.
- Construct the potential field `U` considering both attractive and repulsive factors.

**While goal point not reached:**
- Calculate the gradient descent direction.
- Move the ship and update its current position.

**Path Reconstruction:**
- Once the goal point is reached, construct the whole path.

---

## Slide 6 — Path planning algorithm: Reinforcement Learning

**Environment Setup:**
- Define the environment including the map, obstacles, and other relevant features.
- Specify the state space and action space.

**State Representation:**
- Represent the environment state.
- Common representations: ships' data and environmental factors.

**Reward Design:**
- Define a reward function that guides the agent's behavior.
- Positive rewards for reaching the goal; negative rewards for collisions or inefficiencies.

**Policy Learning:**
- Train an RL agent using techniques like Q-learning, DDPG, or PPO.
- The agent learns a policy (mapping from states to actions) that maximizes expected cumulative reward.

**Training Loop:**
- Iteratively collect experiences `(state, action, reward, next_state)` by interacting with the environment.
- Update the agent's policy using gradient-based optimization.

---

## Slide 7 — Papers and codebase

**Reference papers:**

1. *A Formal Basis for the Heuristic Determination of Minimum Cost Paths.*
2. *Learning in Real-Time Search: A Unifying Framework.*
3. *ARA\*: Anytime A\* with Provable Bounds on Sub-Optimality.*
4. *Optimal and Efficient Path Planning for Partially-Known Environments.*
5. Potential Field lecture notes — https://www.cs.cmu.edu/~motionplanning/lecture/Chap4-Potential-Field_howie.pdf
6. *Rapidly-Exploring Random Trees: A New Tool for Path Planning.*
7. *Survey of deep learning for autonomous surface vehicles in marine environments* (Qiao et al., T-ITS 2023).

**Reference codebase:** https://github.com/zhm-real/PathPlanning#papers — reference implementations of A\*, Bi-A\*, LPA\*, Theta\*, value iteration, etc.

---

## Slide 8 — High-level idea of this project

Two complementary path-planning layers:

| Layer | Algorithms | Characteristics | Output |
|-------|-----------|----------------|--------|
| **Global Path Planning** | A\* and its variants | Optimal, but not fast enough. Does not consider moving obstacles, especially those far away. | Path (waypoints) |
| **Local Path Planning** | Deep Reinforcement Learning (DRL) models | "Short-sighted" (non-optimal), but very fast (real-time). For collision avoidance. | Immediate action |

**Integration of Global and Local Planner:** use the global planner to set **intermediate goals** for the local planner.

---

## Slide 9 — DRL crowd-interaction baseline *(Confidential)*

**General Reinforcement Learning — Deep Reinforcement Learning for Ship Navigation / Crowd Interaction**

DRL navigation baseline consists of three modules:

- **Interaction module** — models ship–ship interactions through coarse-grained local maps.
- **Pooling module** — aggregates the interactions into a fixed-length embedding vector via a self-attention mechanism.
- **Planning module** — estimates the value of the joint state of the robot and crowd for ship navigation.

---

## Slide 10 — Starting point (5 / 8 dynamic obstacles) *(Confidential)*

- General Reinforcement Learning — Deep Reinforcement Learning for Ship Navigation / Crowd Interaction.
- Test scenarios: **dynamic obstacles** (moving ships), **without static obstacles**.
- Obstacle counts shown: **5 obstacles**, **8 obstacles**.

---

## Slide 11 — Starting point (12 / 15 dynamic obstacles) *(Confidential)*

- General Reinforcement Learning — Deep Reinforcement Learning for Ship Navigation / Crowd Interaction.
- Dynamic obstacles (moving ships), without static obstacles.
- Obstacle counts shown: **12 obstacles**, **15 obstacles**.
- Observation: the baseline shows **strong ability for local collision avoidance**.

---

## Slide 12 — Starting point (continuation) *(Confidential)*

- General Reinforcement Learning — Deep Reinforcement Learning for Ship Navigation / Crowd Interaction.
- (Visualisation-only slide — no new textual content.)

---

## Slide 13 — Your Task: Path Planning Algorithms

Four deliverables (ordered by dependency):

1. Installation of coding environment, **re-implementation of the given code**, and **generate simulation videos**.
2. **Modify the local goals** so they are not overlapped in the same region; modify dynamic obstacles for each phase of DRL accordingly.
3. **Create a land map**, and add visual "static obstacles" (i.e., land) onto the visualization output.
4. **Integrate a Global Path Planning algorithm** (e.g., **Theta\***) to automatically set the local goals.

---

## Slide 14 — Your Task: GUI demo (example)

**Buttons** the demo must expose:

- Load static map
- Load DRL model
- Set start point
- Set goal point
- Add static obstacles
- Visualize the path
- Export navigation results
- …

**Visualization:** use your creativity to design an interesting GUI for visualizing the results of maritime navigation.

---

## Slide 15 — Your Task: Report and Presentation

- **Literature reading** on the background and state-of-the-art methods for ship/sea navigation problems.
- **Understand and present** the objective of this project and the specific problems the team is solving.
- **Demonstrate and show** the result of the work.
- **Final Report.**
- **Final Presentation (Competition).**

---

## Slide 16 — Your Task: Project objectives

**Objectives:**

- Understand the background and relevant knowledge about maritime navigation.
- Able to implement the DRL algorithm and generate simulation videos.
- Able to modify the code to produce more realistic DRL navigation simulation.
- Finish a demo program with a GUI to show the navigation results.

**Teams:**

| Group | Size |
|-------|------|
| Group A | 3 members |
| Group B | 3 members |
| Group C | 2 members |

**Literature reading:**
- Conference papers like ICML, etc.
- Journal papers like T-ITS, Ocean Engineering, etc.

**Documents and report writing:**
- User manual for the demo program.
- Final report.

**Presentation:**
- Slides for presentation.
- Demonstrate and show the results.

**Navigation algorithms:**
- Rule-based and DRL-based algorithm.
- PyTorch (recommended), TensorFlow.

**Algorithm implementation:**
- Run the navigation algorithm.
- Program debugging and visualization.

**Experiments:**
- Ablation experiments.
- Experimental results.
- Results for visualization.

**Design of GUI:**
- Interface-oriented programming language, such as PyQt or others.

**GUI programming:**
- Map loading.
- Implementation of functions for buttons and displaying.

**Integration of model and GUI:**
- Export the navigation results.
- Load and call the trained model.
- Display the results.

---

## Slide 17 — Thank you!

*(End of briefing.)*

---

## Cross-references — where each requirement lands in this repo

| Slide content | Where it lives / will live |
|---------------|----------------------------|
| DRL baseline (slide 9) | [crowd_nav/policy/sarl.py](crowd_nav/policy/sarl.py) — SARL / OM-SARL |
| Simulation environment (slides 10–12) | [crowd_sim/envs/crowd_sim.py](crowd_sim/envs/crowd_sim.py) |
| Trained model (slide 13, task 1) | [crowd_nav/data/output_trained/](crowd_nav/data/output_trained) |
| Static-obstacle scaffolding (slide 13, task 3) | [crowd_nav/configs/env.config](crowd_nav/configs/env.config) — `static_obs` flag |
| Global planner (slide 13, task 4) | **TODO** — `crowd_nav/planner/` (to be created) |
| GUI demo (slide 14) | **TODO** — `gui/` (to be created) |
| Architecture & design decisions | [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) |
| Work-package breakdown + group plan | [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) |
