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


def sample_medium_task(seed):
    level = os.environ.get("JEPA_MAZE_LEVEL", "detour").lower()

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
        return np.array(
            [cell[0] - 0.2, cell[1] - 0.2, 0.0, 0.0],
            dtype=np.float64
        )

    print(
        f"[MazeCurriculum] level={level} seed={seed} "
        f"start={start} goal={goal} "
        f"BFS={distance} turns={turns}"
    )

    return to_state(start), to_state(goal)


if __name__ == "__main__":
    for level, candidates in TASKS.items():
        print(f"{level}: {len(candidates)} valid tasks")
