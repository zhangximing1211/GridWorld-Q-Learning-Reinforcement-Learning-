# GridWorld Q-Learning Reinforcement Learning Project Report

## 1. Project Overview

This project implements a reinforcement learning agent based on the **Q-Learning** algorithm. The goal is to learn the optimal path from a start point to a goal in a 20×20 two-dimensional grid world (GridWorld), while avoiding obstacles.

### Tech Stack

| Category | Tool |
|----------|------|
| Programming Language | Python 3 |
| Numerical Computing | NumPy |
| Visualization | PyGame |
| Data Format | CSV |

### Project File Structure

```
524Individual Project/
├── main.py          # Main program: training + visualization entry point
├── qlearning.py     # Q-Learning algorithm implementation
├── env.py           # GridWorld environment class
├── World.py         # Map loading and configuration
├── constants.py     # Constants (tile types, actions)
└── viewer.py        # PyGame visualization renderer
```

---

## 2. Environment Definition

### 2.1 Grid World Map

The map data is loaded from `world03.csv`, a 20×20 integer matrix where each number represents a terrain type:

| Value | Type | Description |
|-------|------|-------------|
| 0 | Empty (EMPTY) | Freely passable |
| 1 | Obstacle (OBSTACLE) | Impassable |
| 2 | Start (START) | Agent's initial position (19, 0) |
| 3 | Goal (GOAL) | Target position (0, 19) |
| 5 | Quicksand (QUICKSAND) | Passable but with extra penalty |

- **Start**: Bottom-left corner (19, 0)
- **Goal**: Top-right corner (0, 19)
- The map contains a complex obstacle layout forming maze-like corridors

### 2.2 Environment Class (`env.py` - GridWorldEnv)

The environment follows the classic RL interaction paradigm: `reset() → step(action) → (next_state, reward, done)`

**State Space**: 2D coordinates (row, col) are flattened into a 1D state index: `state = row × cols + col`, yielding 400 total states.

**Action Space**: 4 discrete actions — UP(0), DOWN(1), LEFT(2), RIGHT(3).

**Reward Mechanism**:

| Event | Reward | Purpose |
|-------|--------|---------|
| Each step taken | -1.0 | Encourages finding the shortest path |
| Hitting an obstacle/boundary | -5.0 | Penalizes invalid moves |
| Stepping into quicksand | -10.0 | Penalizes dangerous areas |
| Reaching the goal | +100.0 | Positive reward for task completion |

**Termination Conditions**: Reaching the goal or exceeding the 400-step limit.

---

## 3. Q-Learning Algorithm Implementation

### 3.1 Algorithm Principles

Q-Learning is a **Model-Free, Off-Policy** Temporal Difference (TD) learning method. The core idea is to maintain a Q-table that records the expected cumulative return for each (state, action) pair.

**Q-Value Update Formula**:

$$Q(s, a) \leftarrow (1 - \alpha) \cdot Q(s, a) + \alpha \cdot \left[ r + \gamma \cdot \max_{a'} Q(s', a') \right]$$

Where:
- $\alpha$ = learning rate (0.2 in this project)
- $\gamma$ = discount factor (0.99 in this project)
- $r$ = immediate reward
- $s'$ = next state

### 3.2 Hyperparameter Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `alpha` | 0.2 | Learning rate, controls Q-value update magnitude |
| `gamma` | 0.99 | Discount factor, emphasizes future rewards |
| `rar` | 0.5 | Initial random action rate (exploration rate) |
| `radr` | 0.99 | Exploration rate decay factor |
| `dyna` | 0 | Dyna-Q iterations (0 = pure Q-Learning) |
| `num_states` | 400 | Number of states (20×20) |
| `num_actions` | 4 | Number of actions |

### 3.3 Exploration Strategy: ε-Greedy

The **ε-Greedy** strategy is used to balance exploration and exploitation:

- With probability ε (`rar`), a random action is selected → **explores** unknown areas
- With probability 1-ε, the action with the highest Q-value is selected → **exploits** existing knowledge
- After each step, ε decays by `radr`: `ε ← ε × 0.99`

This means the agent explores heavily in early training and gradually converges to a greedy policy over time.

### 3.4 Dyna-Q Extension

`qlearning.py` also implements a **Dyna-Q** enhancement (activated when `dyna > 0`):

1. On top of real interactions, it builds a state transition model T and reward model R
2. After each real interaction, it performs `dyna` additional simulated updates (imagination-based planning)
3. It samples from previously visited (s, a) pairs and uses the model to generate virtual experiences for Q-table updates
4. This can significantly accelerate learning convergence

The project defaults to `dyna=0`, using pure Q-Learning.

---

## 4. Training Process

### 4.1 Overall Pipeline (main.py)

```
Load map (world03.csv)
    ↓
Initialize environment (GridWorldEnv)
    ↓
Initialize Q-Learner
    ↓
Training phase (500 Episodes)
    ↓
Policy replay + Visualization (PyGame)
```

### 4.2 Single Episode Training Process

```
Reset environment → Get initial state s₀
    ↓
Q-Learner selects action a (ε-Greedy)
    ↓
Environment executes action → Returns (s', reward, done)
    ↓
Q-Learner updates Q(s, a) and selects next action
    ↓
Repeat until goal is reached or 400 steps exceeded
```

### 4.3 Training Dynamics

| Phase | Episode Range | Behavioral Characteristics |
|-------|--------------|---------------------------|
| Early | 1 – 100 | Heavy random exploration, occasional goal discovery |
| Mid | 100 – 300 | Q-values gradually accumulate, path quality improves |
| Late | 300 – 500 | Policy converges, approaches optimal path |

---

## 5. Visualization System (viewer.py)

After training, `main.py` launches a PyGame window that first displays a **mode selection menu**, offering two interaction modes:

### Mode Selection Menu

The main interface is displayed at startup; the user selects a mode via mouse click:

| Mode | Description |
|------|-------------|
| **Manual (WASD)** | Manual mode — the player controls the agent via keyboard |
| **Auto (Q-Policy)** | Auto mode — the agent navigates using the learned Q-policy |

### Manual Mode

The player uses WASD keys to control the agent's movement in the grid:

| Key | Action |
|-----|--------|
| **W** | Move up (UP) |
| **A** | Move left (LEFT) |
| **S** | Move down (DOWN) |
| **D** | Move right (RIGHT) |

Each key press moves the agent one cell. Players can experience the difficulty of maze navigation firsthand and compare their performance with the Q-Learning policy.

### Auto Mode

The agent uses the trained greedy policy (pure exploitation, no exploration) to automatically navigate step-by-step to the goal.

### General Controls and Display

- **Window size**: 900 × 900 pixels
- **Terrain coloring**: Supports "neon" (cyberpunk style) and "classic" color themes
- **Agent marker**: White circle
- **Path trail**: Previously visited cells are highlighted in a special color
- **HUD info**: Displays current mode, step count, completion status, and agent position at the top
- **Completion prompts**: "GOAL REACHED!" when the goal is reached; "MAX STEPS!" when the step limit is exceeded

**Keyboard Shortcuts**:

| Key | Function |
|-----|----------|
| **M** | Return to mode selection menu |
| **R** | Reset the current game |
| **ESC** | Exit the program |

---

## 6. Core Pipeline Summary

```
┌──────────────────────────────────────────────────────────┐
│                   Data Preparation                        │
│  Load world03.csv → Parse into 20×20 grid → Locate       │
│  start/goal positions                                     │
└───────────────────────┬──────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│                   Environment Setup                       │
│  GridWorldEnv: State space (400) + Action space (4)       │
│  + Reward function                                        │
└───────────────────────┬──────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│                   Model Training                          │
│  Q-Learning: 500 Episodes × up to 400 steps/Episode      │
│  ε-Greedy exploration → Q-table updates → Decay ε        │
└───────────────────────┬──────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│                   Policy Evaluation                       │
│  Greedy policy replay: Pure exploitation of optimal       │
│  Q-table actions to navigate to the goal                  │
└───────────────────────┬──────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│                   Result Visualization                    │
│  PyGame animation with manual/auto interactive modes      │
└──────────────────────────────────────────────────────────┘
```

---

## 7. Key Design Decisions

1. **Pure Q-Learning vs Dyna-Q**: Pure Q-Learning (`dyna=0`) is used by default to maintain algorithmic simplicity, while the Dyna-Q interface is preserved for optional use to accelerate convergence by increasing the `dyna` parameter.

2. **State Representation**: A flattened 1D index is used instead of 2D coordinates to simplify Q-table implementation.

3. **Reward Shaping**: Graduated penalties (step cost < wall collision penalty < quicksand penalty) guide the agent toward learning safe, efficient paths.

4. **Modular Architecture**: The environment (env), algorithm (qlearning), visualization (viewer), and map (World) modules are independent, making the system easy to extend and test.

---

## 8. How to Run

```bash
# Run with the default map
python main.py

# Run with a custom map
python main.py /path/to/custom_world.csv
```

After running `python main.py`, the program first completes 500 training episodes, then opens a PyGame window with a mode selection menu:
1. Click **Manual (WASD)** → Control the agent manually with W/A/S/D keys
2. Click **Auto (Q-Policy)** → Watch the agent navigate automatically using the optimal policy
3. During gameplay, press **M** to return to the menu, **R** to reset, or **ESC** to exit

---

## 9. Conclusion

This project implements a complete reinforcement learning system from scratch:

- **Environment Modeling**: Abstracts the maze navigation problem as a standard MDP (Markov Decision Process)
- **Algorithm Implementation**: Manually implements Q-Learning + Dyna-Q, demonstrating a deep understanding of TD learning mechanisms
- **Policy Learning**: Through the ε-Greedy exploration strategy, the agent learns an efficient path from start to goal within 500 episodes
- **Interactive Visualization**: Provides a PyGame-based visualization with manual and auto dual modes, supporting human-agent comparison experiences

This project demonstrates the application of reinforcement learning to path planning problems and reflects a thorough understanding and practical implementation of core Q-Learning concepts (value functions, exploration-exploitation trade-off, temporal difference updates).
