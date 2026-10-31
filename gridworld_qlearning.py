"""Hello world of reinforcement learning: a grid world solved with Q-learning.

The agent starts in the top-left cell and has to reach the bottom-right.
Every move costs 1, so the return of an episode is minus the number of steps.
Q-learning fills in a table Q[state, action] — how good each move is from each
cell — using only the rewards it actually receives.

    Q(s, a) <- Q(s, a) + alpha * (target - Q(s, a))
    target   = reward                     if the episode ended
             = reward + gamma * max_a' Q(s', a')   otherwise

With gamma = 1 and a cost of 1 per step, a learned Q value is the negative of
the number of steps still to go. No Gym and no neural network: the table is
the whole agent.

Run:
    python3 gridworld_qlearning.py
"""

from __future__ import annotations

import random
from typing import Iterable

# Actions are row/column deltas. Row 0 is the top of the printed grid.
ACTIONS: dict[int, tuple[int, int]] = {
    0: (-1, 0),  # up
    1: (0, 1),   # right
    2: (1, 0),   # down
    3: (0, -1),  # left
}
ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}


class GridWorld:
    """Deterministic grid. Walking into a wall or the edge costs a step and stays put."""

    def __init__(
        self,
        size: int = 4,
        start: tuple[int, int] = (0, 0),
        goal: tuple[int, int] = (3, 3),
        walls: Iterable[tuple[int, int]] = ((1, 1), (1, 2)),
    ) -> None:
        self.size = size
        self.start = start
        self.goal = goal
        self.walls = frozenset(walls)
        self.pos = start

    def reset(self) -> tuple[int, int]:
        self.pos = self.start
        return self.pos

    def step(self, action: int) -> tuple[tuple[int, int], float, bool]:
        dr, dc = ACTIONS[action]
        nxt = (self.pos[0] + dr, self.pos[1] + dc)
        if self._inside(nxt) and nxt not in self.walls:
            self.pos = nxt
        # Reward is -1 on every transition, including the one that enters the goal.
        # The episode return is therefore minus the path length.
        reward = -1.0
        done = self.pos == self.goal
        return self.pos, reward, done

    def _inside(self, cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < self.size and 0 <= c < self.size


class QLearningAgent:
    """Tabular Q-learning with epsilon-greedy action choice."""

    def __init__(self, alpha: float = 0.5, gamma: float = 1.0, epsilon: float = 1.0) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = tuple(ACTIONS)
        self.q: dict[tuple[tuple[int, int], int], float] = {}

    def q_value(self, state: tuple[int, int], action: int) -> float:
        return self.q.get((state, action), 0.0)

    def best_action(self, state: tuple[int, int]) -> int:
        # Lowest action index wins ties, so the printed arrows stay stable.
        return max(self.actions, key=lambda a: (self.q_value(state, a), -a))

    def choose_action(self, state: tuple[int, int]) -> int:
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return self.best_action(state)

    def update(
        self,
        state: tuple[int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int],
        done: bool,
    ) -> None:
        # A finished episode has no next state worth bootstrapping from.
        future = 0.0 if done else max(self.q_value(next_state, a) for a in self.actions)
        target = reward + self.gamma * future
        old = self.q_value(state, action)
        self.q[(state, action)] = old + self.alpha * (target - old)


def epsilon_schedule(episode: int, episodes: int) -> float:
    """1.0 on the first episode, 0.05 on the last. Linear."""
    progress = episode / max(episodes - 1, 1)
    return 1.0 + progress * (0.05 - 1.0)


def train(
    env: GridWorld,
    agent: QLearningAgent,
    episodes: int = 400,
    max_steps: int = 100,
) -> list[float]:
    """Return the total reward of each episode."""
    returns: list[float] = []
    for episode in range(episodes):
        # Explore almost at random at first, then mostly trust the table.
        agent.epsilon = epsilon_schedule(episode, episodes)

        state = env.reset()
        total = 0.0
        for _ in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total += reward
            state = next_state
            if done:
                break
        returns.append(total)
    return returns


def greedy_return(env: GridWorld, agent: QLearningAgent, max_steps: int = 100) -> float:
    """Walk the learned policy with no exploration."""
    agent.epsilon = 0.0
    state = env.reset()
    total = 0.0
    for _ in range(max_steps):
        action = agent.best_action(state)
        state, reward, done = env.step(action)
        total += reward
        if done:
            break
    return total


def render_policy(env: GridWorld, agent: QLearningAgent) -> str:
    rows: list[str] = []
    for r in range(env.size):
        cells: list[str] = []
        for c in range(env.size):
            cell = (r, c)
            if cell == env.goal:
                cells.append("G")
            elif cell in env.walls:
                cells.append("#")
            else:
                cells.append(ARROWS[agent.best_action(cell)])
        rows.append(" ".join(cells))
    return "\n".join(rows)


def render_values(env: GridWorld, agent: QLearningAgent) -> str:
    """V(s) = max_a Q(s, a). With these rewards, V(s) is minus steps-to-go."""
    rows: list[str] = []
    for r in range(env.size):
        cells: list[str] = []
        for c in range(env.size):
            cell = (r, c)
            if cell in env.walls:
                cells.append("  #")
            else:
                value = max(agent.q_value(cell, a) for a in agent.actions)
                cells.append(f"{value:4.0f}")
        rows.append(" ".join(cells))
    return "\n".join(rows)


def main() -> None:
    random.seed(0)
    env = GridWorld()
    agent = QLearningAgent(alpha=0.5, gamma=1.0)
    returns = train(env, agent, episodes=400)

    print("Grid: S is top-left, G is the goal, # is a wall. Every move costs 1.\n")
    print("Return while training. -6 is a shortest path; more negative means a longer walk.")
    episodes = len(returns)
    for episode in (0, 50, 100, 200, episodes - 1):
        eps = epsilon_schedule(episode, episodes)
        print(f"  episode {episode:3d}   return {returns[episode]:6.0f}   epsilon {eps:.2f}")

    window = returns[-50:]
    print(f"\nMean return over the last 50 episodes: {sum(window) / len(window):.1f}")
    print("\nGreedy policy (follow the arrows):")
    print(render_policy(env, agent))
    print("\nState values, V(s) = max Q. A value of -6 means six steps remain:")
    print(render_values(env, agent))

    score = greedy_return(env, agent)
    optimal = -6.0  # shortest path around the two walls is 6 moves
    print(f"\nGreedy rollout return: {score:.0f}  (shortest path is {optimal:.0f})")
    if score != optimal:
        raise SystemExit("Policy is not optimal yet. Try more episodes.")


if __name__ == "__main__":
    main()
