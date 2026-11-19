"""Medium Maze difficulty-based task sampler."""

from collections import deque
import os
import numpy as np

from evals.simu_env_planning.envs.pointmaze_env.maze_model import (
    MEDIUM_MAZE, parse_maze, EMPTY, GOAL
)


def shortest_path(start, goal, free):
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        current, path = queue.popleft()
        if current == goal:
            return path

        r, c = current
        for nxt in (
            (r + 1, c), (r - 1, c),
            (r, c + 1), (r, c - 1)
        ):
            if nxt in free and nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, path + [nxt]))

    return None


def count_turns(path):
    directions = [
        (b[0] - a[0], b[1] - a[1])
        for a, b in zip(path[:-1], path[1:])
    ]
    return sum(
        a != b for a, b in zip(directions[:-1], directions[1:])
    )


def build_tasks():
    maze = parse_maze(MEDIUM_MAZE)

    free = {
        (r, c)
        for r in range(maze.shape[0])
        for c in range(maze.shape[1])
        if maze[r, c] in (EMPTY, GOAL)
    }

    tasks = {"straight": [], "corner": [], "detour": []}

    for start in sorted(free):
        for goal in sorted(free):
            if start == goal:
                continue

            path = shortest_path(start, goal, free)
            if path is None:
                continue

            distance = len(path) - 1
            turns = count_turns(path)

            if 2 <= distance <= 3 and turns == 0:
                level = "straight"
            elif 3 <= distance <= 4 and turns == 1:
                level = "corner"
            elif 5 <= distance <= 9 and turns >= 2:
                level = "detour"
            else:
                continue

            tasks[level].append((start, goal, distance, turns))

    return tasks


TASKS = build_tasks()


def sample_medium_task(seed, level=None):
    """Sample one deterministic, collision-free grid-centre task.

    Explicit maze-medium-{straight,corner,detour} task names take priority.
    JEPA_MAZE_LEVEL remains supported for the legacy maze-medium config.
    """
    if level is None:
        level = os.environ.get("JEPA_MAZE_LEVEL", "detour")
    level = level.lower()

    if level not in TASKS:
        raise ValueError(f"Invalid difficulty: {level}")

    candidates = TASKS[level]
    if not candidates:
        raise RuntimeError(f"No tasks for {level}")

    rng = np.random.RandomState(seed)
    start, goal, distance, turns = candidates[
        rng.randint(len(candidates))
    ]

    def to_state(cell):
        # Official maze_model.py: wall centre is (r+1, c+1), while the
        # particle body is initially at (1.2, 1.2). The slider qpos that
        # puts the particle at a free tile centre is therefore tile-0.2.
        # This 0.2 offset must not be changed to tile+1.
        return np.array(
            [cell[0] - 0.2, cell[1] - 0.2, 0.0, 0.0],
            dtype=np.float64
        )

    print(
        f"[MazeCurriculum] level={level} seed={seed} "
        f"start={start} goal={goal} "
        f"BFS={distance} turns={turns}"
    )

    # Check that our grid-to-qpos mapping stays in the free cells.
    for cell in (start, goal):
        qpos = to_state(cell)[:2]
        world_pos = qpos + np.array([1.2, 1.2])
        assert np.allclose(world_pos, np.asarray(cell) + 1.0)

    return to_state(start), to_state(goal)


if __name__ == "__main__":
    for level, candidates in TASKS.items():
        print(f"{level}: {len(candidates)} valid tasks")
