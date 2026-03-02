"""
GridWorld Environment for Q-Learning Reinforcement Learning.

A 20x20 grid where an agent navigates from a start cell to a goal cell
while avoiding obstacle cells.
"""

import numpy as np

# Cell type constants
EMPTY = 0
OBSTACLE = 1
START = 2
GOAL = 3

# Action constants (index -> (row_delta, col_delta))
ACTIONS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}
NUM_ACTIONS = len(ACTIONS)

# Rewards
REWARD_GOAL = 100
REWARD_OBSTACLE = -10
REWARD_STEP = -1


class GridWorld:
    """A 20x20 GridWorld environment."""

    def __init__(self, grid_size: int = 20, seed: int = 42):
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed)
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.start = (0, 0)
        self.goal = (grid_size - 1, grid_size - 1)
        self._place_obstacles()
        self.agent_pos = self.start

    # ------------------------------------------------------------------
    # Environment setup
    # ------------------------------------------------------------------

    def _place_obstacles(self):
        """Randomly place obstacles (≈20 % of cells), leaving start/goal clear."""
        n_obstacles = int(self.grid_size * self.grid_size * 0.20)
        placed = 0
        while placed < n_obstacles:
            r = int(self.rng.integers(0, self.grid_size))
            c = int(self.rng.integers(0, self.grid_size))
            if (r, c) in (self.start, self.goal):
                continue
            if self.grid[r, c] == EMPTY:
                self.grid[r, c] = OBSTACLE
                placed += 1

    # ------------------------------------------------------------------
    # Gym-like interface
    # ------------------------------------------------------------------

    def reset(self):
        """Reset the agent to the start position and return the initial state."""
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action: int):
        """
        Apply *action* and return ``(next_state, reward, done)``.

        Parameters
        ----------
        action : int
            One of the keys in ``ACTIONS``.

        Returns
        -------
        next_state : tuple[int, int]
        reward : float
        done : bool
        """
        dr, dc = ACTIONS[action]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc

        # Boundary check – stay in place if the action would leave the grid
        if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
            return self.agent_pos, REWARD_OBSTACLE, False

        if self.grid[nr, nc] == OBSTACLE:
            return self.agent_pos, REWARD_OBSTACLE, False

        self.agent_pos = (nr, nc)

        if self.agent_pos == self.goal:
            return self.agent_pos, REWARD_GOAL, True

        return self.agent_pos, REWARD_STEP, False

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def is_obstacle(self, r: int, c: int) -> bool:
        return self.grid[r, c] == OBSTACLE

    def render(self) -> str:
        """Return a plain-text representation of the current grid state."""
        lines = []
        for r in range(self.grid_size):
            row = []
            for c in range(self.grid_size):
                if (r, c) == self.agent_pos:
                    row.append("A")
                elif (r, c) == self.goal:
                    row.append("G")
                elif (r, c) == self.start:
                    row.append("S")
                elif self.grid[r, c] == OBSTACLE:
                    row.append("#")
                else:
                    row.append(".")
            lines.append(" ".join(row))
        return "\n".join(lines)
