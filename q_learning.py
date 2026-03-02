"""
Q-Learning agent for the GridWorld environment.

The agent maintains a Q-table of shape (grid_size, grid_size, num_actions)
and updates it using the standard Q-Learning (off-policy TD) rule:

    Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
"""

import numpy as np
from grid_world import GridWorld, NUM_ACTIONS


class QLearningAgent:
    """
    Tabular Q-Learning agent.

    Parameters
    ----------
    env : GridWorld
    alpha : float
        Learning rate (0 < alpha <= 1).
    gamma : float
        Discount factor (0 <= gamma <= 1).
    epsilon : float
        Initial exploration probability for ε-greedy policy.
    epsilon_min : float
        Minimum exploration probability.
    epsilon_decay : float
        Multiplicative decay applied to ε after every episode.
    """

    def __init__(
        self,
        env: GridWorld,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: int = None,
    ):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        n = env.grid_size
        self.q_table = np.zeros((n, n, NUM_ACTIONS))

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def choose_action(self, state: tuple) -> int:
        """Return an action using an ε-greedy policy."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, NUM_ACTIONS))
        r, c = state
        return int(np.argmax(self.q_table[r, c]))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self, state, action, reward, next_state):
        """Apply one Q-Learning update step."""
        r, c = state
        nr, nc = next_state
        best_next = np.max(self.q_table[nr, nc])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[r, c, action]
        self.q_table[r, c, action] += self.alpha * td_error

    def decay_epsilon(self):
        """Decay ε after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, n_episodes: int = 2000, max_steps: int = 500):
        """
        Train the agent for *n_episodes* episodes.

        Returns
        -------
        rewards : list[float]
            Total reward collected in each episode.
        """
        rewards = []
        for _ in range(n_episodes):
            state = self.env.reset()
            total_reward = 0.0
            for _ in range(max_steps):
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.update(state, action, reward, next_state)
                state = next_state
                total_reward += reward
                if done:
                    break
            self.decay_epsilon()
            rewards.append(total_reward)
        return rewards

    # ------------------------------------------------------------------
    # Greedy path extraction
    # ------------------------------------------------------------------

    def get_greedy_path(self, max_steps: int = 500):
        """
        Follow the greedy policy from start to goal.

        Returns
        -------
        path : list[tuple[int, int]]
            Sequence of visited cells (including start and goal).
        success : bool
            Whether the goal was reached within *max_steps*.
        """
        state = self.env.reset()
        path = [state]
        for _ in range(max_steps):
            r, c = state
            action = int(np.argmax(self.q_table[r, c]))
            next_state, _, done = self.env.step(action)
            if next_state == state and not done:
                # Stuck (all moves lead to obstacles/boundaries)
                break
            path.append(next_state)
            state = next_state
            if done:
                return path, True
        return path, False
