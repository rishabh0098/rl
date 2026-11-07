# Reinforcement learning

Two tabular Q-learning programs. Each one is a single Python file plus a page that defines the words as they come up. Python 3 is enough. There are no extra packages.

The same update is used in both:

```text
Q(s, a) = Q(s, a) + alpha * (target - Q(s, a))
target  = reward                          if the episode ended
        = reward + gamma * max Q(s', a')  otherwise
```

The grid world is the place to learn what that line means. One agent walks a fixed board, every step costs 1, and a learned value is minus the steps still to go. Tic-tac-toe keeps the update and changes the world: the other seat is the environment, and the two seats learn by playing each other.

## Grid world

A 4×4 grid. The agent starts at the top-left and learns to reach the bottom-right. Every move costs 1, so a shortest path has return −6.

```bash
python3 gridworld/gridworld_qlearning.py
```

[How the grid world learns](gridworld/README.md)

## Tic-tac-toe

X and O share one Q-table. On each turn the board is rewritten so the player about to move is `+1` and the other player is `−1`. A step includes the opponent's reply. The reward is 0 until the game ends, then +1, 0, or −1. Perfect play draws, and training takes a few seconds.

```bash
python3 tictactoe/tictactoe_qlearning.py
```

[How the tic-tac-toe agent learns](tictactoe/README.md)
