# Grid world Q-learning

A 4×4 grid. The agent starts at the top-left and learns to reach the bottom-right. Every move costs 1, so a shortest path has return −6.

```bash
python3 gridworld/gridworld_qlearning.py
```

Python 3 is enough. There are no extra packages.

One episode looks like this: `reset` puts the agent at `(0, 0)`. The agent picks an action. `step` returns the next cell, a reward of −1, and whether the goal was reached. `update` changes one entry in the Q-table. That repeats until the goal, or until 100 steps.

## What Q-learning is

Q-learning is a way for an agent to learn which action is best in each situation, using only the rewards it receives while it acts.

It keeps a table. Each row is a state (here, a cell). Each column is an action (up, right, down, left). The number in a cell of that table is called `Q(s, a)`. It is the agent's current guess of the total reward it will collect if it takes action `a` in state `s` and, from then on, always chooses the best action it knows.

The guess starts at 0, which is wrong. Each time the agent actually moves, it sees a real reward and a real next cell, and it nudges that one number toward what the move just taught it. The formula is in [The learning rule](#the-learning-rule). After many episodes the guesses settle. The agent then looks at the current cell and takes the action with the largest number. Those choices are the arrows printed at the end.

A short example of how a number gets its meaning: the agent is one step left of the goal and steps right. The reward is −1 and the episode ends, so the target for that action is −1. From a cell two steps away, the target is −1 plus the best number in the next cell. Once the neighbor has learned "one step left, worth −1", this cell learns "two steps left, worth −2". The path length spreads backward through the table, one cell at a time. Early episodes wander. Later episodes are short because the numbers already point the way.

This file is **tabular** Q-learning: the table is an ordinary dictionary, `agent.q`. Nothing is approximated.

## When it is used

Use Q-learning when all of these are true:

- You can name the states and the actions. A cell on a small grid, a square on a board, or a reading from a handful of sensors.
- The agent learns by trying. It does not need a written map of "if I step here, I will land there." It discovers that by calling `step`.
- What you want at the end is the best action in each state, judged by the total reward over the whole episode, not by the next reward alone.
- There are few enough states and actions that a table fits in memory and every pair gets visited often enough to learn. This 4×4 grid has 14 open cells and 4 actions, so 56 numbers.

That covers a lot of first problems: mazes, tic-tac-toe against a fixed opponent, and simple control tasks where the state is a short list of discrete values. The world is allowed to be noisy. If the same action sometimes leads to different next states, repeated updates average that in. This particular grid is deterministic, which makes the learned numbers easy to read (they become minus the steps remaining).

Reach for a different method when the problem leaves that list:

- **Too many states for a table.** A camera image, or a position that can be any real number, cannot be one row per state. The same update rule can sit on top of a neural network that guesses `Q(s, a)`. That is deep Q-learning, and it is a later program than this one.
- **You already know every outcome.** If the map, the walls, and the reward of every move are known up front, the numbers can be computed by sweeping the grid, with no random walking. Q-learning is the method for when those outcomes have to be sampled by acting.
- **You want the score of the behavior you are actually following,** including its random exploratory moves. Q-learning scores the best next action even while training still explores. An update that scores the action the agent will really take next is a different algorithm.

## The world

**Environment.** `GridWorld`. It owns the rules: the walls, what a move does, and the reward for that move. The agent only calls `reset` and `step`.

**Agent.** `QLearningAgent`. It chooses actions and stores what it has learned. It does not know where the walls are. It only sees the cell it is in, the reward it just received, and the cell it landed in.

**State.** A cell, written `(row, column)`. Row 0 is the top of the printed grid. The start state is `(0, 0)`. The goal state is `(3, 3)`.

**Action.** One of four moves, numbered in `ACTIONS`:

| Action | Move | Delta `(row, column)` |
| --- | --- | --- |
| 0 | up | `(-1, 0)` |
| 1 | right | `(0, 1)` |
| 2 | down | `(1, 0)` |
| 3 | left | `(0, -1)` |

**Transition.** One call to `step(action)`. From the current state and that action, the environment returns the next state, the reward, and `done`.

**Deterministic.** The same state and the same action always produce the same next state and the same reward. The only randomness is in the agent's exploration, controlled by `random.seed(0)` so a run of this file repeats.

**Wall and edge.** Cells `(1, 1)` and `(1, 2)` are walls. A move that would leave the 4×4 grid, or step onto a wall, leaves the agent where it is. The reward is still −1, because a step was spent.

**Reward.** The scalar the environment pays for one transition. In this grid it is always −1, including the transition that lands on the goal. There is no separate prize for arriving. The goal is valuable because the episode ends and the agent stops paying for more steps.

**Terminal state.** `done` is true only when the agent is on the goal. The episode ends, and the update does not look beyond that state.

**Episode.** One attempt: from `reset` until the goal, or until `max_steps` (100) if the goal is still unreached. Training runs 400 episodes.

**Return.** The sum of rewards collected in one episode. With a reward of −1 per step, the return is minus the number of steps. −6 is a shortest path. −14 is a walk of 14 steps. An episode that never finds the goal stops at 100 steps and has return −100.

**Step cap.** `max_steps` cuts off a lost episode so training cannot loop forever. Hitting the cap does not set `done`. Only standing on the goal does. The last update of a capped episode still includes a next-state estimate.

**Optimal.** A policy whose greedy walk from the start has return −6, the length of a shortest path around the two walls.

## What is stored

**Q-value.** `Q(s, a)` is the number stored for one state and one action: how good it is to take that action in that state and continue from there. The table `agent.q` starts empty. `q_value` treats a missing entry as 0.

**Tabular method.** The learner is that table, one number per `(state, action)`, not a function approximator. This grid has few cells, so every value can be printed.

**State value.** `V(s) = max` over actions of `Q(s, a)`, computed in `render_values`. It is the best the table currently claims for cell `s`. After learning, with a step cost of 1 and `gamma` of 1, `V(s)` is minus the steps still to go. The start cell prints as −6. The goal prints as 0, because no transition is updated out of the goal.

**Policy.** A choice of action for each state. The arrows from `render_policy` are the greedy policy. `G` is the goal and `#` is a wall.

**Greedy action.** The action with the largest Q-value in that state, from `best_action`. If two actions tie, the lower action index wins (up before right before down before left), so the printed arrows stay put.

**Greedy rollout.** `greedy_return` sets epsilon to 0 and walks once, using only `best_action`. The return of that walk is the score of the learned policy. `main` checks that it equals −6.

## The learning rule

One move changes one number. The agent was in cell `s`, took action `a`, received `reward`, and landed in cell `s'`. `update` rewrites `Q(s, a)` and leaves every other entry alone.

It does that in three lines, which are `QLearningAgent.update`:

```text
future = 0                          if this move reached the goal
future = best number stored at s'   otherwise

target = reward + gamma * future
Q(s, a) = Q(s, a) + alpha * (target - Q(s, a))
```

`future` is the agent's guess of what the rest of the path is worth, read from the table at the cell it just entered. Reaching the goal ends the episode, so the rest of the path is worth 0. Any other landing cell still has a path ahead, so `future` is the largest Q-value stored there: the best action the table currently knows in `s'`.

`target` is what this one move says `Q(s, a)` should be. The move itself paid `reward` (−1). The rest of the trip is `future`. Add them.

The last line does not jump straight to `target`. It takes one step from the old number toward `target`. How big that step is depends on `alpha`.

**A move onto the goal.** The agent is in the cell just left of `G` and steps right. Reward is −1, the episode ends, so `future` is 0 and `target` is −1. The table still has 0 for that move, and `alpha` is 0.5:

```text
new Q = 0 + 0.5 * (−1 − 0) = −0.5
```

The same move again, now starting from −0.5:

```text
new Q = −0.5 + 0.5 * (−1 − (−0.5)) = −0.75
```

Each visit covers half the remaining distance to −1. After enough visits that entry sits at −1, which means "this action costs one step and then the episode is over."

**A move one cell further away.** The agent steps right into that cell beside the goal. Suppose the best number stored in the landing cell is already −1. Then:

```text
target = −1 + gamma * (−1)
```

This program uses `gamma = 1`, so `target = −2`. The move costs 1, and the best continuation from the next cell costs another 1. The same halfway step pulls the stored number toward −2. That is how "six steps remain" at the start gets built: each cell learns its number from the cell it steps into, working backward from the goal.

**Learning rate (alpha).** `alpha = 0.5` means each visit moves halfway from the old number to the target. `alpha = 1` would replace the old number with the target on the first visit. `alpha = 0` would leave the table unchanged.

**Discount factor (gamma).** The weight on `future` inside the target. `gamma = 1` counts a later step the same as the step just taken, which is why a learned value equals minus the number of steps left. A smaller gamma would count far-away steps less.

**Bootstrap.** `future` is read from the table's current guess at `s'`. The agent does not wait to finish the episode and add up every later reward. It trusts the number already written in the next cell. On the goal that guess is defined to be 0, because there is no next cell.

**Temporal-difference error.** The gap `target − Q(s, a)`. In the first example the gap was −1 − 0 = −1, and the update added `alpha` times that gap. A gap of 0 means the stored number already matches what this move just showed, so the entry stays as it is.

**Why the best next action.** `future` uses the largest Q-value in `s'`, whichever action that is. The action the agent happens to try on the following step can be a random wander. The number written for this step still assumes that, once there, the agent will take the best action the table knows. The table is therefore learning the arrows you see at the end, even during episodes that still wander.

## How actions are chosen

The update above only runs after a move. Choosing the move is separate, in `choose_action`.

**Exploration.** Picking an action at random, so the agent can still find a better path than the one the table currently likes.

**Exploitation.** Picking the action with the largest Q-value and trusting the table.

**Epsilon (ε).** The probability of exploring on this step. With probability epsilon, `choose_action` draws one of the four actions at random. Otherwise it takes the best action.

**Epsilon schedule.** `epsilon_schedule` sets epsilon to 1.0 on episode 0, so every action is random, and lowers it in a straight line to 0.05 on the last episode, so almost every action follows the table. The 0.05 that remains keeps a little exploration through the end of training.
