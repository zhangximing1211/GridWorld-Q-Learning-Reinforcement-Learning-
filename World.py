import sys
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from constants import EMPTY, OBSTACLE, START, GOAL, QUICKSAND, QS_STEPPED, TRAIL

@dataclass(frozen=True)
class WorldMeta:
    rows: int
    cols: int
    start: Tuple[int, int]
    goal: Tuple[int, int]
    obstacles: List[Tuple[int, int]]
    quicksand: List[Tuple[int, int]]
    palette: Dict[int, Tuple[int, int, int]]
    checker_a: Tuple[int, int, int]
    checker_b: Tuple[int, int, int]
    grid_line: Tuple[int, int, int]
    title: str


def load_world(
    csv_path: str,
    *,
    expected_shape: Optional[Tuple[int, int]] = (20, 20),
    strict: bool = True,
    theme: str = "neon",
) -> Tuple[np.ndarray, WorldMeta]:
    """读取 CSV -> grid + UI meta（主题配色、障碍列表等）"""

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = []
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            str_values = line.split(",")
            try:
                int_values = [int(v) for v in str_values]
            except ValueError:
                raise ValueError(f"Invalid integer value at line {line_no} in {csv_path}")
            rows.append(int_values)

    grid = np.array(rows, dtype=int)
    if grid.ndim != 2:
        raise ValueError(f"Grid must be 2D, got shape {grid.shape} in {csv_path}")

    r, c = grid.shape

    # shape 校验
    if expected_shape is not None and (r, c) != expected_shape:
        msg = f"Expected shape {expected_shape}, got {grid.shape} in {csv_path}"
        if strict:
            raise ValueError(msg)
        else:
            print("Warning:", msg, file=sys.stderr)

    # tile values 校验
    allowed = {EMPTY, OBSTACLE, START, GOAL}
    unique = set(np.unique(grid).tolist())
    illegal = sorted(list(unique - allowed))
    if illegal:
        msg = f"Illegal tile values {illegal} found in {csv_path}"
        if strict:
            raise ValueError(msg)
        else:
            print("Warning:", msg, file=sys.stderr)

    # start / goal
    start_pos = np.argwhere(grid == START)
    goal_pos = np.argwhere(grid == GOAL)
    if len(start_pos) != 1 or len(goal_pos) != 1:
        raise ValueError(
            f"Expected exactly one START and one GOAL in {csv_path}, "
            f"found {len(start_pos)} START and {len(goal_pos)} GOAL"
        )
    start = tuple(map(int, start_pos[0]))
    goal = tuple(map(int, goal_pos[0]))

    obstacles = [tuple(map(int, pos)) for pos in np.argwhere(grid == OBSTACLE)]
    quicksand = [tuple(map(int, pos)) for pos in np.argwhere((grid == QUICKSAND) | (grid == QS_STEPPED))]

    # theme palettes
    if theme == "neon":
        palette = {
            EMPTY: (18, 20, 28),
            OBSTACLE: (255, 64, 129),
            START: (0, 224, 255),
            GOAL: (0, 255, 140),
            TRAIL: (120, 120, 140),
            QUICKSAND: (255, 214, 0),
            QS_STEPPED: (255, 165, 0),
        }
        checker_a = (14, 16, 22)
        checker_b = (22, 24, 34)
        grid_line = (35, 38, 54)
        title = "Neon Themed Grid World"
    elif theme == "classic":
        palette = {
            EMPTY: (245, 245, 245),
            OBSTACLE: (60, 60, 60),
            START: (80, 160, 255),
            GOAL: (80, 200, 120),
            TRAIL: (180, 180, 180),
            QUICKSAND: (240, 220, 160),
            QS_STEPPED: (230, 190, 120),
        }
        checker_a = (250, 250, 250)
        checker_b = (238, 238, 238)
        grid_line = (210, 210, 210)
        title = "Classic Themed Grid World"
    else:
        raise ValueError(f"Unknown theme '{theme}'")

    meta = WorldMeta(
        rows=r,
        cols=c,
        start=start,
        goal=goal,
        obstacles=obstacles,
        quicksand=quicksand,
        palette=palette,
        checker_a=checker_a,
        checker_b=checker_b,
        grid_line=grid_line,
        title=title,
    )
    return grid, meta