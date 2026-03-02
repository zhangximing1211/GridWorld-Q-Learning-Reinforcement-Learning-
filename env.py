import numpy as np
from typing import Tuple, Set
from constants import OBSTACLE, QUICKSAND, QS_STEPPED
from constants import UP, DOWN, LEFT, RIGHT
from World import WorldMeta

def rc_to_state(r: int, c: int, cols: int) -> int:
    return r * cols + c


def state_to_rc(s: int, cols: int) -> Tuple[int, int]:
    return s // cols, s % cols


class GridWorldEnv:
    def __init__(
        self,
        base_grid: np.ndarray,
        meta: 'WorldMeta',
        *,
        step_cost: float = -1.0,
        obstacle_cost: float = -5.0,
        quicksand_cost: float = -10.0,
        goal_reward: float = 100.0,
        max_steps: int = 400,
        record_trail: bool = True,
    ):
        self.base_grid = np.array(base_grid, copy=True)
        self.meta = meta
        self.rows = meta.rows
        self.cols = meta.cols

        self.step_cost = step_cost
        self.obstacle_cost = obstacle_cost
        self.quicksand_cost = quicksand_cost
        self.goal_reward = goal_reward
        self.max_steps = max_steps
        self.record_trail = record_trail

        self.pos: Tuple[int, int] = meta.start
        self.steps: int = 0
        self.done: bool = False
        self.trail: Set[Tuple[int, int]] = set()

    def reset(self) -> int:
        self.pos = self.meta.start
        self.steps = 0
        self.done = False
        self.trail = set()
        if self.record_trail:
            self.trail.add(self.pos)
        return rc_to_state(self.pos[0], self.pos[1], self.cols)

    def _is_obstacle(self, r: int, c: int) -> bool:
        return int(self.base_grid[r, c]) == OBSTACLE

    def _is_quicksand(self, r: int, c: int) -> bool:
        return int(self.base_grid[r, c]) in (QUICKSAND, QS_STEPPED)

    def _is_goal(self, r: int, c: int) -> bool:
        return (r, c) == self.meta.goal

    def step(self, action: int):
        if self.done:
            s = rc_to_state(self.pos[0], self.pos[1], self.cols)
            return s, 0.0, True, {"pos": self.pos}

        r, c = self.pos
        dr, dc = 0, 0
        if action == UP:
            dr = -1
        elif action == DOWN:
            dr = 1
        elif action == LEFT:
            dc = -1
        elif action == RIGHT:
            dc = 1

        nr, nc = r + dr, c + dc
        self.steps += 1

        # out of bounds -> treat like obstacle
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            reward = self.obstacle_cost
            nr, nc = r, c
        elif self._is_obstacle(nr, nc):
            reward = self.obstacle_cost
            nr, nc = r, c
        else:
            reward = self.step_cost
            if self._is_quicksand(nr, nc):
                reward += self.quicksand_cost
            if self._is_goal(nr, nc):
                reward += self.goal_reward
                self.done = True

        self.pos = (nr, nc)
        if self.record_trail:
            self.trail.add(self.pos)

        if self.steps >= self.max_steps and not self.done:
            self.done = True

        s_prime = rc_to_state(nr, nc, self.cols)
        return s_prime, float(reward), bool(self.done), {"pos": self.pos}
