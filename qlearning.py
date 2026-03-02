import random
from typing import Set, Tuple, List
import numpy as np

class QLearning:
    def __init__(
        self,
        num_states=100,
        num_actions=4,
        alpha=0.2,
        gamma=0.99,
        dyna=0,
        rar=0.5,
        radr=0.99,
        verbose=False,
    ):
        self.verbose = verbose
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna

        self.s = 0
        self.a = 0

        self.Q = np.zeros((num_states, num_actions), dtype=float)

        # Dyna model
        self.T = np.zeros((num_states, num_actions, num_states), dtype=float)
        self.Tc = np.full((num_states, num_actions, num_states), 0.00001, dtype=float)
        self.R = np.zeros((num_states, num_actions), dtype=float)

    def querysetstate(self, s: int) -> int:
        """设置当前状态，选择一个动作（epsilon-greedy），不更新Q表"""
        self.s = s
        if random.random() < self.rar:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = int(np.argmax(self.Q[s]))
        self.a = action
        if self.verbose:
            print("querysetstate: s =", s, "a =", action)
        return action

    def query(self, s_prime: int, r: float) -> int:
        """根据 (s, a, r, s') 更新 Q，并返回下一步动作（epsilon-greedy）"""
        if not hasattr(self, "_seen_sa_set"):
            self._seen_sa_set: Set[Tuple[int, int]] = set()
            self._seen_sa_list: List[Tuple[int, int]] = []
        # Q update
        self.Q[self.s, self.a] = (1 - self.alpha) * self.Q[self.s, self.a] + self.alpha * (
            r + self.gamma * np.max(self.Q[s_prime])
        )

        # Dyna updates
        if self.dyna > 0:
            self.Tc[self.s, self.a, s_prime] += 1
            self.T[self.s, self.a, :] = self.Tc[self.s, self.a, :] / np.sum(self.Tc[self.s, self.a, :])
            self.R[self.s, self.a] = (1 - self.alpha) * self.R[self.s, self.a] + self.alpha * r

                # 把真实经历过的 (s,a) 存起来（只存一次）
            key = (self.s, self.a)
            if key not in self._seen_sa_set:
                self._seen_sa_set.add(key)
                self._seen_sa_list.append(key)

            for _ in range(self.dyna):
                # 只从见过的 (s,a) 抽样，确保 T 分布有效
                s_sim, a_sim = random.choice(self._seen_sa_list)

                probs = self.T[s_sim, a_sim, :]
                p_sum = probs.sum()

                # 理论上 p_sum 应该 > 0，但做个保险，避免极端情况下仍报错
                if not np.isfinite(p_sum) or p_sum <= 0:
                    s_next = random.randint(0, self.Q.shape[0] - 1)
                else:
                    probs = probs / p_sum
                    s_next = int(np.random.choice(np.arange(self.Q.shape[0]), p=probs))

                r_sim = self.R[s_sim, a_sim]

                old_value_sim = self.Q[s_sim, a_sim]
                future_value_sim = np.max(self.Q[s_next])
                self.Q[s_sim, a_sim] = (1 - self.alpha) * old_value_sim + self.alpha * (
                    r_sim + self.gamma * future_value_sim
                )
  

        # choose next action
        if random.random() < self.rar:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = int(np.argmax(self.Q[s_prime]))

        # decay exploration
        self.rar *= self.radr

        self.s = s_prime
        self.a = action

        if self.verbose:
            print("query: s' =", s_prime, "a =", action, "r =", r)

        return action

    def greedy_action(self, s: int) -> int:
        return int(np.argmax(self.Q[s]))

    def save_q(self, path: str):
        np.save(path, self.Q)

    def load_q(self, path: str):
        self.Q = np.load(path)
