"""Tic-tac-toe solved by self-play with tabular Q-learning.

Two seats share one table. On each turn the board is rewritten so the player
about to move is always +1 and the other player is always -1. A transition is
not "I placed a mark." It is "I placed a mark, then the other seat answered."

    Q(s, a) <- Q(s, a) + alpha * (target - Q(s, a))
    target   = reward                     if the game ended
             = reward + gamma * max_a' Q(s', a')   otherwise

Rewards are 0 until the game ends, then +1 for a win, 0 for a draw, -1 for a
loss. With gamma = 1, Q(s, a) is the expected final result of that move against
the opponent the agent has actually been playing. No Gym and no neural network.

Run:
    python3 tictactoe_qlearning.py
"""

from __future__ import annotations

import random
from typing import Callable

# Cells are numbered like this when a board is printed:
#   0 1 2
#   3 4 5
#   6 7 8
# Absolute boards store 0 empty, 1 X, 2 O.
# Boards passed to the agent store 0 empty, +1 me, -1 opponent.
Board = tuple[int, ...]
LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)
GLYPH = {0: ".", 1: "X", 2: "O"}
EMPTY: Board = (0, 0, 0, 0, 0, 0, 0, 0, 0)

# Enough games for the greedy policy to draw with perfect play from both seats.
EPISODES = 100_000
ALPHA = 0.2
EPSILON_END = 0.05
RANDOM_GAMES = 1000


def legal_cells(board: Board) -> list[int]:
    return [i for i, cell in enumerate(board) if cell == 0]


def place(board: Board, cell: int, mark: int) -> Board:
    cells = list(board)
    cells[cell] = mark
    return tuple(cells)


def outcome(board: Board) -> int | None:
    """1 or 2 if that player has three in a row, 0 if the board is full, else None."""
    for a, b, c in LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    if 0 in board:
        return None
    return 0


def from_my_side(board: Board, mark: int) -> Board:
    """Rewrite an absolute board so `mark` is +1 and the opponent is -1."""
    return tuple(0 if cell == 0 else (1 if cell == mark else -1) for cell in board)


def render(board: Board) -> str:
    rows = []
    for row in range(3):
        rows.append(" ".join(GLYPH[board[3 * row + col]] for col in range(3)))
    return "\n".join(rows)


class QLearningAgent:
    """Tabular Q-learning. Missing entries are 0, a guess of "this will draw"."""

    def __init__(self, alpha: float = ALPHA, gamma: float = 1.0, epsilon: float = 1.0) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q: dict[tuple[Board, int], float] = {}

    def q_value(self, state: Board, action: int) -> float:
        return self.q.get((state, action), 0.0)

    def best_action(self, state: Board) -> int:
        actions = legal_cells(state)
        # Lowest cell index wins ties, so a greedy game is stable.
        return max(actions, key=lambda a: (self.q_value(state, a), -a))

    def choose_action(self, state: Board) -> int:
        actions = legal_cells(state)
        if random.random() < self.epsilon:
            return random.choice(actions)
        return self.best_action(state)

    def update(
        self,
        state: Board,
        action: int,
        reward: float,
        next_state: Board,
        done: bool,
    ) -> None:
        if done:
            future = 0.0
        else:
            future = max(self.q_value(next_state, a) for a in legal_cells(next_state))
        target = reward + self.gamma * future
        old = self.q_value(state, action)
        self.q[(state, action)] = old + self.alpha * (target - old)


def epsilon_schedule(episode: int, episodes: int) -> float:
    """1.0 on the first episode, EPSILON_END on the last. Linear."""
    progress = episode / max(episodes - 1, 1)
    return 1.0 + progress * (EPSILON_END - 1.0)


def self_play(agent: QLearningAgent, choose: Callable[[Board], int] | None = None) -> int:
    """Play one game, both seats using this agent, and return 1, 2, or 0 for a draw.

    The previous seat's (state, action) is held until the reply arrives. Only
    then is there a next state where that same seat is to move again. A finished
    game closes both the move that just ended it and the move before it.
    """
    if choose is None:
        choose = agent.choose_action

    board = EMPTY
    pending: tuple[Board, int, int] | None = None
    mark = 1
    while True:
        state = from_my_side(board, mark)
        action = choose(state)
        board = place(board, action, mark)
        result = outcome(board)

        if result is not None:
            reward_me = 1.0 if result == mark else 0.0
            agent.update(state, action, reward_me, state, True)
            if pending is not None:
                prev_state, prev_action, _prev_mark = pending
                reward_them = -1.0 if result == mark else 0.0
                agent.update(prev_state, prev_action, reward_them, prev_state, True)
            return result

        if pending is not None:
            prev_state, prev_action, prev_mark = pending
            next_state = from_my_side(board, prev_mark)
            agent.update(prev_state, prev_action, 0.0, next_state, False)
        pending = (state, action, mark)
        mark = 3 - mark


def train(agent: QLearningAgent, episodes: int) -> list[int]:
    """Return the winner of each episode: 1, 2, or 0."""
    results: list[int] = []
    for episode in range(episodes):
        agent.epsilon = epsilon_schedule(episode, episodes)
        results.append(self_play(agent))
    return results


Chooser = Callable[[Board, int], int]


def greedy_game(agent: QLearningAgent) -> tuple[int, Board, list[int]]:
    """One game, both seats greedy. Returns the result, the final board, and the cells played."""
    board = EMPTY
    moves: list[int] = []
    mark = 1
    while True:
        cell = agent.best_action(from_my_side(board, mark))
        moves.append(cell)
        board = place(board, cell, mark)
        result = outcome(board)
        if result is not None:
            return result, board, moves
        mark = 3 - mark


def play_pair(play_x: Chooser, play_o: Chooser) -> int:
    """One game with no learning. Each chooser sees the absolute board and whose turn it is."""
    board = EMPTY
    mark = 1
    players = {1: play_x, 2: play_o}
    while True:
        board = place(board, players[mark](board, mark), mark)
        result = outcome(board)
        if result is not None:
            return result
        mark = 3 - mark


def random_move(board: Board, mark: int) -> int:
    return random.choice(legal_cells(board))


def minimax(board: Board, mark: int, memo: dict[Board, tuple[int, int]]) -> tuple[int, int]:
    """Best result and a best cell for the player about to move.

    The result is +1 win, 0 draw, -1 loss, assuming both sides play perfectly
    from here. The lower cell index wins ties.
    """
    cached = memo.get(board)
    if cached is not None:
        return cached

    result = outcome(board)
    if result is not None:
        # A finished board: the player who would have moved has already lost,
        # unless the board is full and nobody won.
        value = 0 if result == 0 else -1
        memo[board] = (value, -1)
        return memo[board]

    best_value = -2
    best_cell = -1
    for cell in legal_cells(board):
        opp_value, _opp_cell = minimax(place(board, cell, mark), 3 - mark, memo)
        value = -opp_value
        if value > best_value or (value == best_value and cell < best_cell):
            best_value = value
            best_cell = cell
    memo[board] = (best_value, best_cell)
    return memo[board]


def perfect_move(memo: dict[Board, tuple[int, int]]) -> Chooser:
    def move(board: Board, mark: int) -> int:
        _value, cell = minimax(board, mark, memo)
        return cell

    return move


def agent_move(agent: QLearningAgent) -> Chooser:
    def move(board: Board, mark: int) -> int:
        return agent.best_action(from_my_side(board, mark))

    return move


def action_value(board: Board, mark: int, cell: int, memo: dict[Board, tuple[int, int]]) -> int:
    """Perfect-play result of putting `mark` in `cell`, for the player about to move."""
    opp_value, _cell = minimax(place(board, cell, mark), 3 - mark, memo)
    return -opp_value


def reachable() -> list[tuple[Board, int]]:
    """Every legal position where somebody still has to move, in play order."""
    found: dict[Board, int] = {}

    def walk(board: Board, mark: int) -> None:
        if outcome(board) is not None or board in found:
            return
        found[board] = mark
        for cell in legal_cells(board):
            walk(place(board, cell, mark), 3 - mark)

    walk(EMPTY, 1)
    return list(found.items())


def policy_report(
    agent: QLearningAgent, memo: dict[Board, tuple[int, int]]
) -> tuple[int, int, tuple[Board, int] | None]:
    """Count reachable positions whose greedy move perfect play would accept.

    Also return the shortest position where it would not, if there is one.
    """
    correct = 0
    total = 0
    missed: list[tuple[Board, int]] = []
    for board, mark in reachable():
        total += 1
        cell = agent.best_action(from_my_side(board, mark))
        best, _best_cell = minimax(board, mark, memo)
        if action_value(board, mark, cell, memo) == best:
            correct += 1
        else:
            missed.append((board, mark))
    return correct, total, simplest(missed)


def tally(play_x: Chooser, play_o: Chooser, games: int) -> tuple[int, int, int]:
    """Wins for X, wins for O, draws."""
    x_wins = o_wins = draws = 0
    for _ in range(games):
        result = play_pair(play_x, play_o)
        if result == 1:
            x_wins += 1
        elif result == 2:
            o_wins += 1
        else:
            draws += 1
    return x_wins, o_wins, draws


def window_line(results: list[int], end: int, episodes: int, width: int = 2000) -> str:
    start = max(0, end + 1 - width)
    chunk = results[start : end + 1]
    n = len(chunk)
    x_wins = sum(result == 1 for result in chunk)
    o_wins = sum(result == 2 for result in chunk)
    draws = n - x_wins - o_wins
    eps = epsilon_schedule(end, episodes)
    return (
        f"  episode {end:6d}   "
        f"X {100 * x_wins / n:5.1f}%   "
        f"O {100 * o_wins / n:5.1f}%   "
        f"draw {100 * draws / n:5.1f}%   "
        f"epsilon {eps:.2f}"
    )


def render_opening_q(agent: QLearningAgent) -> str:
    rows = []
    state = from_my_side(EMPTY, 1)
    for row in range(3):
        cells = [f"{agent.q_value(state, 3 * row + col):+5.2f}" for col in range(3)]
        rows.append(" ".join(cells))
    return "\n".join(rows)


def simplest(candidates: list[tuple[Board, int]]) -> tuple[Board, int] | None:
    if not candidates:
        return None
    return min(candidates, key=lambda item: (sum(cell != 0 for cell in item[0]), item[0]))


def example_positions(
    memo: dict[Board, tuple[int, int]],
) -> tuple[tuple[Board, int] | None, tuple[Board, int] | None]:
    """A short win-in-one position, and a short position where a blunder loses."""
    wins: list[tuple[Board, int]] = []
    blocks: list[tuple[Board, int]] = []
    for board, mark in reachable():
        cells = legal_cells(board)
        immediate = [cell for cell in cells if outcome(place(board, cell, mark)) == mark]
        values = [action_value(board, mark, cell, memo) for cell in cells]
        if immediate and min(values) < 1:
            wins.append((board, mark))
            continue
        opponent = 3 - mark
        threatened = any(outcome(place(board, cell, opponent)) == opponent for cell in cells)
        if threatened and not immediate and max(values) >= 0 and min(values) < max(values):
            blocks.append((board, mark))
    return simplest(wins), simplest(blocks)


def print_position(
    agent: QLearningAgent,
    board: Board,
    mark: int,
    memo: dict[Board, tuple[int, int]],
) -> None:
    state = from_my_side(board, mark)
    print(render(board))
    print(f"{GLYPH[mark]} to move. Learned Q, then the perfect-play result of that cell:")
    for cell in legal_cells(board):
        perfect = action_value(board, mark, cell, memo)
        print(f"  cell {cell}   Q {agent.q_value(state, cell):+.2f}   perfect {perfect:+d}")
    print(f"Greedy cell: {agent.best_action(state)}")


def check_update_timing() -> None:
    """One forced game. The winning move must land on +1 and the move that allowed it on -1."""
    agent = QLearningAgent(alpha=1.0, gamma=1.0, epsilon=0.0)
    script = [0, 3, 1, 4, 2]

    def choose(_state: Board) -> int:
        return script.pop(0)

    # X plays 0, 1, 2 down the top row. O plays 3, then 4, and X wins.
    if self_play(agent, choose) != 1 or script:
        raise RuntimeError("scripted top-row game did not end as a win for X")

    boards = [EMPTY]
    board = EMPTY
    for cell, mark in zip((0, 3, 1, 4, 2), (1, 2, 1, 2, 1)):
        board = place(board, cell, mark)
        boards.append(board)

    win_state = from_my_side(boards[4], 1)
    blunder_state = from_my_side(boards[3], 2)
    if agent.q_value(win_state, 2) != 1.0 or agent.q_value(blunder_state, 4) != -1.0:
        raise RuntimeError("terminal rewards were written onto the wrong moves")


def check_perfect_play(memo: dict[Board, tuple[int, int]]) -> None:
    if minimax(EMPTY, 1, memo)[0] != 0:
        raise RuntimeError("perfect play from an empty board should be a draw")
    perfect = perfect_move(memo)
    if play_pair(perfect, perfect) != 0:
        raise RuntimeError("two perfect players should draw")


def seat_record(wins: int, draws: int, losses: int) -> str:
    games = wins + draws + losses
    return (
        f"wins {wins:4d} ({100 * wins / games:5.1f}%)   "
        f"draws {draws:4d} ({100 * draws / games:5.1f}%)   "
        f"losses {losses:4d} ({100 * losses / games:5.1f}%)"
    )


def main() -> None:
    check_update_timing()
    random.seed(0)
    memo: dict[Board, tuple[int, int]] = {}
    check_perfect_play(memo)
    perfect = perfect_move(memo)

    agent = QLearningAgent(alpha=ALPHA, gamma=1.0)
    results = train(agent, EPISODES)
    agent.epsilon = 0.0
    learned = agent_move(agent)

    print("Tic-tac-toe. X moves first. Both seats share one Q-table and learn by playing.")
    print("Reward is 0 until the end, then +1 win, 0 draw, -1 loss.\n")
    print(f"Training games: {EPISODES}. alpha {ALPHA}. epsilon falls from 1.00 to {EPSILON_END:.2f}.")
    print("Each line is the previous 2000 games, exploration included.\n")
    checkpoints = sorted({EPISODES // 20 - 1, EPISODES // 4 - 1, EPISODES // 2 - 1, EPISODES - 1})
    for episode in checkpoints:
        print(window_line(results, episode, EPISODES))

    print(f"\nTable entries: {len(agent.q)}")
    print("\nCells:\n0 1 2\n3 4 5\n6 7 8")
    print("\nLearned Q for X's first move. Perfect play draws from every cell,")
    print("so a value above 0 is profit this opponent still gives away:\n")
    print(render_opening_q(agent))

    win_example, block_example = example_positions(memo)
    if win_example is not None:
        print("\nA position one move from a win:")
        print_position(agent, win_example[0], win_example[1], memo)
    if block_example is not None:
        print("\nA position where one of the cells throws the game away:")
        print_position(agent, block_example[0], block_example[1], memo)

    result, final, moves = greedy_game(agent)
    names = {1: "X wins", 2: "O wins", 0: "draw"}
    print(f"\nGreedy self-play, no exploration: {names[result]}")
    print(f"Cells played, X first: {' '.join(str(cell) for cell in moves)}\n")
    print(render(final))

    as_x = play_pair(learned, perfect)
    as_o = play_pair(perfect, learned)
    correct, total, missed = policy_report(agent, memo)
    print(f"\nGreedy X against perfect O: {names[as_x]}")
    print(f"Perfect X against greedy O: {names[as_o]}")
    print(f"Greedy move is perfect in {correct} of {total} reachable positions.")
    if missed is not None:
        print(f"\n{total - correct} positions are still wrong. Shortest one:")
        print("Greedy picks a cell that did well against this same agent.")
        print("Perfect play punishes that cell. Training rarely did.")
        print_position(agent, missed[0], missed[1], memo)

    learned_x = tally(learned, random_move, RANDOM_GAMES)
    learned_o = tally(random_move, learned, RANDOM_GAMES)
    perfect_x = tally(perfect, random_move, RANDOM_GAMES)
    perfect_o = tally(random_move, perfect, RANDOM_GAMES)
    print(f"\nAgainst a random opponent, {RANDOM_GAMES} games each seat.")
    print("Learned as X:", seat_record(learned_x[0], learned_x[2], learned_x[1]))
    print("Learned as O:", seat_record(learned_o[1], learned_o[2], learned_o[0]))
    print("Perfect as X:", seat_record(perfect_x[0], perfect_x[2], perfect_x[1]))
    print("Perfect as O:", seat_record(perfect_o[1], perfect_o[2], perfect_o[0]))

    losses = learned_x[1] + learned_o[0]
    if as_x != 0 or as_o != 0 or losses != 0:
        raise SystemExit("Greedy play failed the exam. It should draw with perfect play and lose none of the random games.")


if __name__ == "__main__":
    main()
