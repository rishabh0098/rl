# Tic-tac-toe Q-learning

Two seats play tic-tac-toe and share one Q-table. Each learns from the games it plays against the other. With perfect play the game is a draw, so a trained pair that has stopped exploring should draw.

```bash
python3 tictactoe/tictactoe_qlearning.py
```

Python 3 is enough. There are no extra packages. Training plays 100,000 games and takes a few seconds. `random.seed(0)` makes a run repeat.

The update rule is the same one as in the [grid world](../README.md). That page is the longer introduction to Q-learning. This page is about what a state, a next state, and a reward are when the opponent is the other seat.

## What is different from the grid

In the grid, the walls never move. A number learned on episode 10 is still the right target on episode 400.

Here the opponent is part of the transition. "I placed a mark" is not a finished step. The step is "I placed a mark, then the other seat answered, and now it is my turn again." While that other seat is learning, the same move can start producing different replies. A Q value that was right against a beginner is wrong once the opponent starts blocking. The table has to keep moving. That is why `alpha` stays at 0.2 for the whole run, instead of shrinking toward zero: a shrinking step size would freeze the early, lucky numbers in place.

## The world

**Environment.** The board rules at the top of `tictactoe_qlearning.py`: `place`, `outcome`, and `legal_cells`. Nine cells. X moves first. Three in a row wins. A full board with no three in a row is a draw.

**Agent.** `QLearningAgent`. One object plays both seats. Both read and write `agent.q`.

**State.** A board the agent can use as a dictionary key. Empty is 0, the player about to move is +1, the opponent is −1. `from_my_side` rewrites an absolute board (0 empty, 1 X, 2 O) into that form before every choice and every update.

After X opens in the center, the absolute board and O's view of it are:

```text
. . .          0  0  0
. X .          0 -1  0
. . .          0  0  0
absolute       O to move
```

O's center cell is −1, because the center belongs to the other player. X and O never store "an X mark" and "an O mark" as different things. They store "mine" and "theirs." A tactic learned from either seat is waiting for the other seat the next time the pattern appears.

**Action.** A cell index in `0 .. 8`:

```text
0 1 2
3 4 5
6 7 8
```

`choose_action` only offers empty cells. An occupied cell is not an action, so the table never spends games learning the rules of placement.

**Reward.** 0 on every move that leaves the game going. When the game ends, the player who just moved gets +1 for a win and 0 for a draw. The other player gets −1 for that loss and 0 for that draw. There is no prize for taking the center, and no penalty for a slow win. The only signal is the result.

**Episode.** One game, at most nine moves.

**Return.** The sum of one seat's rewards. Because every earlier reward is 0, the return is just that seat's final +1, 0, or −1.

## When a move is updated

The previous seat's `(state, action)` is kept in `pending` until the reply exists. Only then is there a next state in which that same seat is to move. `self_play` does this for both seats.

A short game, the one `check_update_timing` forces before training. X plays 0, then 1, then 2. O plays 3, then 4. X completes the top row.

```text
X . .        X X .
O . .        O O .
. . .        . . .
after O's    X plays 2
second move  and wins
```

Two writes happen on that last move:

- X's state, action 2, reward +1, and the episode is over, so there is no future term. With `alpha = 1` this writes `Q = 1`. The real run uses `alpha = 0.2`, so one visit covers a fifth of the gap from 0 toward 1, and later visits finish the job.
- O's stored move, action 4, reward −1. That is the move that allowed the top row. It is pulled toward −1.

Moves earlier in the game were updated when the reply arrived, with reward 0. Their target was `gamma` times the best number then stored in the resulting position. At the start of training those numbers are 0, so the early moves stay near 0 until a finished game has written a +1 or a −1 further down the line. The result spreads backward through the table, one reply at a time, the same way the step cost spreads backward across the grid.

The next state is always a board where this seat is to move, after the opponent's reply. The board right after your own mark, before the reply, is the opponent's turn. Taking `max Q` there would mean "what if I moved twice."

## The learning rule

One resolved move changes one number. `QLearningAgent.update`:

```text
future = 0                          if the game is over
future = best number stored at s'   otherwise

target = reward + gamma * future
Q(s, a) = Q(s, a) + alpha * (target - Q(s, a))
```

A missing entry is read as 0. For this game, 0 means "I expect a draw," which is the right guess for perfect play from the opening.

**Learning rate (alpha).** `alpha = 0.2`. Each visit moves a fifth of the way from the stored number to the target. It does not shrink. The opponent is changing, so the table has to be willing to revise an old number late in training.

**Discount factor (gamma).** `gamma = 1`. A win next move counts the same as a win on the last move. With rewards only at the end, `Q(s, a)` is the expected final result of that cell against the opponent these games actually contain.

**Bootstrap.** `future` is read from the table. The agent does not wait to add up a return for the non-terminal update. It trusts the best number already written at `s'`. On a finished game that number is defined to be 0.

**Why the best next action.** `future` uses the largest Q at `s'`, whichever cell that is. The cell tried on the following visit can be an exploratory one. The number written for this move still assumes that, once it is this seat's turn again, the seat will take the best cell the table knows.

## How actions are chosen

**Exploration.** A uniform draw among the empty cells, so a seat can still find a better cell than the one the table currently likes.

**Exploitation.** The empty cell with the largest Q. If two cells tie, the lower index wins, so a greedy game repeats.

**Epsilon.** The probability of exploring on this move. It starts at 1, so the first games are random, and falls in a straight line to 0.05. The 0.05 that remains is load-bearing. If both seats become fully greedy, they repeat one draw forever and stop producing the positions the table still has wrong.

Both seats share epsilon. During a training game, either seat can be the one exploring.

## How to read the run

The printed windows are 2,000 training games ending at that episode. Exploration is still on, so X wins a lot early (X moves first, and random play favors the first move) and draws climb once both seats have learned to block. The last window is mostly draws and is not 100% draws, because 5% of moves are still random.

Then epsilon is set to 0 and the table is examined.

**Opening values.** Nine numbers for X's first move. Perfect play draws from every opening cell, so every one of those cells has perfect-play value 0. The learned numbers sit a little above 0. The center sits highest. An opponent who still explores loses more often after a center opening, and the table recorded that.

**A win in one move.** The program finds a short position where some empty cell ends the game, and prints each empty cell's learned Q next to the result perfect play assigns to that cell. The winning cell should be near +1, and greedy should take it.

**A cell that throws the game away.** A short position where the opponent is threatening a line, and at least one empty cell loses while another does not. Greedy should take a cell whose perfect-play value is the best in the list. Some losing cells can still show a positive Q. Those games ended well because the other seat did not take the win it was offered. The number is the result against the opponent in the training games. It is not a promise about a perfect opponent.

**Greedy self-play.** Both seats take `best_action`. With no exploration the whole game is determined, so this is one game, not a sample. It should be a draw. The cells and the final board are printed.

**Perfect opponent.** `minimax` scores a position as +1, 0, or −1 for the player about to move, assuming both sides play perfectly from there. The value of a cell is the opposite of the value the opponent can then force. From the empty board that value is 0: perfect play draws. The agent never trains against this player. Minimax is the exam. Greedy X against perfect O should draw, and perfect X against greedy O should draw. If either game is a loss, the process exits with an error.

**Reachable positions.** Every legal board in which somebody still has to move. For each one, the exam asks whether the greedy cell's perfect-play value equals the best cell's value. Several ties are fine: perfect play often has more than one drawing cell, and the lowest index is only a tie-break. The run prints how many positions pass.

A few dozen do not. Self-play can leave them that way. Both seats share the table, so they share the blind spot. A cell that loses to a perfect reply keeps a high Q when the other seat almost never plays that reply, and a saving or winning cell that is rarely tried stays near 0. More games of the same kind do not fix it: against this opponent the bad cell really does score better. The run prints the shortest such position, with Q and the perfect-play value side by side.

**Random opponent.** 1,000 games with the learned seat as X and 1,000 as O, and the same for perfect play, as a reference. A perfect player never loses to a random one. The learned player should not either, in this sample; a loss fails the run.

Win rates will not match the perfect player's win rates. Perfect play, given several cells that all draw against a perfect opponent, takes the lowest index. The table takes the cell that won more often against the opponent it trained against, who still blunders. That is visible in O's win rate against the random player: the learned seat converts more of those games into wins than the index tie-break does. X's win rate sits just below the perfect player's, which is the other side of the same comparison. The failures-to-convert are draws, not losses.
