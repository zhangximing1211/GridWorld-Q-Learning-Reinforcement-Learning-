"""
Unit tests for the GridWorld environment and Q-Learning agent.
"""

import numpy as np
import pytest

from grid_world import (
    GridWorld,
    ACTIONS,
    NUM_ACTIONS,
    OBSTACLE,
    REWARD_GOAL,
    REWARD_OBSTACLE,
    REWARD_STEP,
)
from q_learning import QLearningAgent


# ---------------------------------------------------------------------------
# GridWorld tests
# ---------------------------------------------------------------------------

class TestGridWorld:

    def setup_method(self):
        self.env = GridWorld(grid_size=20, seed=42)

    def test_grid_shape(self):
        assert self.env.grid.shape == (20, 20)

    def test_start_and_goal_not_obstacles(self):
        sr, sc = self.env.start
        gr, gc = self.env.goal
        assert self.env.grid[sr, sc] != OBSTACLE
        assert self.env.grid[gr, gc] != OBSTACLE

    def test_reset_returns_start(self):
        self.env.agent_pos = (5, 5)
        state = self.env.reset()
        assert state == self.env.start

    def test_step_reach_goal(self):
        """Place agent next to the goal and step into it."""
        env = GridWorld(grid_size=5, seed=0)
        # Manually clear the grid so we can control the test
        env.grid[:] = 0
        env.start = (0, 0)
        env.goal = (4, 4)
        env.agent_pos = (4, 3)  # one step left of goal
        _, reward, done = env.step(3)  # action 3 = right
        assert done is True
        assert reward == REWARD_GOAL

    def test_step_into_obstacle(self):
        """Hitting an obstacle should return a negative reward and stay put."""
        env = GridWorld(grid_size=5, seed=0)
        env.grid[:] = 0
        env.start = (0, 0)
        env.goal = (4, 4)
        env.grid[0, 1] = OBSTACLE
        env.agent_pos = (0, 0)
        state, reward, done = env.step(3)  # try to move right into obstacle
        assert state == (0, 0)
        assert reward == REWARD_OBSTACLE
        assert done is False

    def test_step_out_of_bounds(self):
        """Hitting a boundary should return a negative reward and stay put."""
        env = GridWorld(grid_size=5, seed=0)
        env.grid[:] = 0
        env.start = (0, 0)
        env.goal = (4, 4)
        env.agent_pos = (0, 0)
        state, reward, done = env.step(0)  # try to move up out of grid
        assert state == (0, 0)
        assert reward == REWARD_OBSTACLE
        assert done is False

    def test_step_normal_move(self):
        env = GridWorld(grid_size=5, seed=0)
        env.grid[:] = 0
        env.start = (0, 0)
        env.goal = (4, 4)
        env.agent_pos = (2, 2)
        state, reward, done = env.step(1)  # down
        assert state == (3, 2)
        assert reward == REWARD_STEP
        assert done is False

    def test_num_actions(self):
        assert NUM_ACTIONS == 4

    def test_render_contains_agent(self):
        self.env.reset()
        rendered = self.env.render()
        assert "A" in rendered

    def test_obstacle_ratio_approx_20_percent(self):
        env = GridWorld(grid_size=20, seed=1)
        obstacle_count = np.sum(env.grid == OBSTACLE)
        total = 20 * 20
        ratio = obstacle_count / total
        assert 0.15 <= ratio <= 0.25


# ---------------------------------------------------------------------------
# QLearningAgent tests
# ---------------------------------------------------------------------------

class TestQLearningAgent:

    def setup_method(self):
        self.env = GridWorld(grid_size=20, seed=42)
        self.agent = QLearningAgent(self.env)

    def test_q_table_shape(self):
        n = self.env.grid_size
        assert self.agent.q_table.shape == (n, n, NUM_ACTIONS)

    def test_q_table_initialised_to_zero(self):
        assert np.all(self.agent.q_table == 0)

    def test_choose_action_valid(self):
        state = (0, 0)
        for _ in range(20):
            action = self.agent.choose_action(state)
            assert 0 <= action < NUM_ACTIONS

    def test_update_changes_q_value(self):
        state = (0, 0)
        action = 1
        reward = -1.0
        next_state = (1, 0)
        before = self.agent.q_table[0, 0, action]
        self.agent.update(state, action, reward, next_state)
        after = self.agent.q_table[0, 0, action]
        assert after != before

    def test_epsilon_decay(self):
        initial_eps = self.agent.epsilon
        self.agent.decay_epsilon()
        assert self.agent.epsilon < initial_eps

    def test_epsilon_does_not_go_below_min(self):
        self.agent.epsilon = self.agent.epsilon_min
        self.agent.decay_epsilon()
        assert self.agent.epsilon == self.agent.epsilon_min

    def test_train_returns_rewards_list(self):
        rewards = self.agent.train(n_episodes=10, max_steps=50)
        assert len(rewards) == 10
        assert all(isinstance(r, float) for r in rewards)

    def test_train_improves_reward(self):
        """Average reward in the last quarter should exceed the first quarter."""
        rewards = self.agent.train(n_episodes=500, max_steps=300)
        first_avg = np.mean(rewards[:125])
        last_avg = np.mean(rewards[-125:])
        assert last_avg > first_avg

    def test_greedy_path_on_small_clear_grid(self):
        """On a small obstacle-free grid the agent should reliably find the goal."""
        env = GridWorld(grid_size=5, seed=99)
        env.grid[:] = 0  # clear all obstacles
        env.start = (0, 0)
        env.goal = (4, 4)
        agent = QLearningAgent(env, epsilon_decay=0.99)
        agent.train(n_episodes=1000, max_steps=100)
        _, success = agent.get_greedy_path(max_steps=200)
        assert success
