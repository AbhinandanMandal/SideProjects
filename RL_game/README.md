# Q-Learning Maze Game

A small browser-based maze game that demonstrates reinforcement learning with Q-learning.

The agent starts at `S`, tries to reach `G`, and updates a Q-table after every move.

## Reward design

- `+100` for reaching the goal.
- Extra speed bonus for finishing with fewer steps.
- `-1` for every normal move, encouraging shorter paths.
- `-8` for hitting a wall, discouraging invalid moves.

## Run

Open `index.html` in a browser.

## Controls

- `Train 500 episodes`: runs many Q-learning episodes quickly.
- `Watch learned path`: follows the best learned action from each cell.
- `Step once`: advances one learning step manually.
- `Reset learning`: clears the Q-table.

The blue strokes on the board show each cell's current best learned move once its Q-values become positive.
