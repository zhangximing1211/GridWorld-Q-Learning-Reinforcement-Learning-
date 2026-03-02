"""
Main script: train the Q-Learning agent on the 20×20 GridWorld and
produce two figures:

1. rewards_curve.png  – episode reward over training
2. optimal_path.png   – the learned greedy path on the grid
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from grid_world import GridWorld, OBSTACLE
from q_learning import QLearningAgent

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
GRID_SIZE = 20
N_EPISODES = 3000
MAX_STEPS = 500
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
SEED = 42

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def smooth(data, window: int = 50):
    """Return a simple moving average of *data*."""
    if len(data) < window:
        return np.array(data, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def plot_rewards(rewards, path: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    smoothed = smooth(rewards)
    offset = len(rewards) - len(smoothed)
    ax.plot(rewards, alpha=0.3, color="steelblue", label="Episode reward")
    ax.plot(range(offset, offset + len(smoothed)), smoothed, color="orange",
            linewidth=2, label="Moving avg (window=50)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Q-Learning Training – Reward Curve")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_path(env: GridWorld, path, success: bool, path_out: str):
    grid = env.grid
    n = env.grid_size

    # Build colour matrix
    cmap_data = np.zeros((n, n, 3))
    for r in range(n):
        for c in range(n):
            if grid[r, c] == OBSTACLE:
                cmap_data[r, c] = [0.2, 0.2, 0.2]   # dark grey
            else:
                cmap_data[r, c] = [0.95, 0.95, 0.95]  # light grey

    # Mark path
    for r, c in path:
        cmap_data[r, c] = [0.4, 0.7, 1.0]  # blue

    # Start / Goal
    sr, sc = env.start
    gr, gc = env.goal
    cmap_data[sr, sc] = [0.2, 0.8, 0.2]   # green
    cmap_data[gr, gc] = [1.0, 0.3, 0.3]   # red

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cmap_data, origin="upper")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(range(n), fontsize=6)
    ax.set_yticklabels(range(n), fontsize=6)
    ax.grid(True, color="white", linewidth=0.5)

    status = "SUCCESS" if success else "FAILED (no path found)"
    ax.set_title(f"Greedy Path after Training – {status}\nPath length: {len(path)} steps")

    patches = [
        mpatches.Patch(color=[0.2, 0.8, 0.2], label="Start"),
        mpatches.Patch(color=[1.0, 0.3, 0.3], label="Goal"),
        mpatches.Patch(color=[0.4, 0.7, 1.0], label="Path"),
        mpatches.Patch(color=[0.2, 0.2, 0.2], label="Obstacle"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path_out, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path_out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== GridWorld Q-Learning ===")
    print(f"Grid size : {GRID_SIZE}x{GRID_SIZE}")
    print(f"Episodes  : {N_EPISODES}")

    env = GridWorld(grid_size=GRID_SIZE, seed=SEED)
    agent = QLearningAgent(
        env,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
    )

    print("Training ...")
    rewards = agent.train(n_episodes=N_EPISODES, max_steps=MAX_STEPS)
    print(f"Training complete. Last-100-episode avg reward: {np.mean(rewards[-100:]):.2f}")

    path, success = agent.get_greedy_path(max_steps=MAX_STEPS)
    print(f"Greedy path: {'reached goal' if success else 'did NOT reach goal'} in {len(path)} steps")

    plot_rewards(rewards, os.path.join(OUTPUT_DIR, "rewards_curve.png"))
    plot_path(env, path, success, os.path.join(OUTPUT_DIR, "optimal_path.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
