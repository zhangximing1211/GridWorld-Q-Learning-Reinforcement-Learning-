import os
import sys
from World import load_world
from World import WorldMeta
from env import GridWorldEnv
from qlearning import QLearning
from viewer import run_policy_pygame

def train_agent(env: GridWorldEnv, learner: QLearning, episodes: int = 800) -> None:
    for ep in range(episodes):
        s = env.reset()
        a = learner.querysetstate(s)
        done = False
        while not done:
            s_prime, r, done, _ = env.step(a)
            a = learner.query(s_prime, r)

DEFAULT_CSV_PATH = "/Users/zhangximing/Desktop/world03.csv"  # <- 改成你的 CSV 路径

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_CSV_PATH
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            f"请修改 DEFAULT_CSV_PATH 或用命令行传参：python rl_grid_demo.py /path/to/world03.csv"
        )

    grid, meta = load_world(csv_path, theme="neon", expected_shape=(20, 20), strict=True)
    print(meta.title)
    print("start:", meta.start, "goal:", meta.goal)
    print("num obstacles:", len(meta.obstacles))
    print("cwd:", os.getcwd())

    # env：训练时可以不记录 trail（省一点点），回放时记录 trail（更好看）
    train_env = GridWorldEnv(grid, meta, record_trail=False, max_steps=400)
    replay_env = GridWorldEnv(grid, meta, record_trail=True, max_steps=400)

    num_states = meta.rows * meta.cols
    learner = QLearning(
        num_states=num_states,
        num_actions=4,
        alpha=0.2,
        gamma=0.99,
        dyna=0,      # 0=纯Q-learning；>0=Dyna-Q 更快
        rar=0.5,
        radr=0.99,
        verbose=False,
    )

    # train
    train_agent(train_env, learner, episodes=500)

    # replay
    run_policy_pygame(
        grid,
        meta,
        replay_env,
        learner,
        greedy=True,
        step_every_n_frames=6,
        fps=30,
        window=(900, 900),
        pad=14,
        gap=2,
        radius=6,
    )