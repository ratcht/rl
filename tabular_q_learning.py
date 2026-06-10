from dataclasses import dataclass
from enum import Enum

import numpy as np
from tqdm import tqdm

rng = np.random.default_rng(seed=0)


class Action(Enum):
  Up = 0
  Right = 1
  Left = 2
  Down = 3


DIRECTIONS = {
  Action.Up: (0, 1),
  Action.Right: (1, 0),
  Action.Left: (-1, 0),
  Action.Down: (0, -1),
}

n_actions = 4

type State = list[int]
type Reward = float


class Environment:
  def __init__(self, dim: tuple[int, int]):
    self.dim = dim
    self.prev_pos: State = [0, 0]
    self.pos: State = [0, 0]
    self.goal = [dim[0] - 1, dim[1] - 1]

  def _has_reached_goal(self) -> bool:
    return self.pos == self.goal

  def step(self, action: Action) -> tuple[Reward, bool]:
    dx, dy = DIRECTIONS[action]

    new_pos_0 = max(0, min(self.pos[0] + dx, self.dim[0]))
    new_pos_1 = max(0, min(self.pos[1] + dy, self.dim[1]))

    # update history
    self.pos[0], self.prev_pos[0] = new_pos_0, self.pos[0]
    self.pos[1], self.prev_pos[1] = new_pos_1, self.pos[1]

    return (0.0, True) if self._has_reached_goal() else (-1.0, False)  # reward

  def reset(self) -> None:
    self.pos = [0, 0]

  def render(self) -> None:
    for y in range(self.dim[1]):
      row = ""
      for x in range(self.dim[0]):
        if [x, y] == self.pos == self.goal:
          row += "AG "
        elif [x, y] == self.pos:
          row += "A "
        elif [x, y] == self.goal:
          row += "G "
        else:
          row += ". "
      print(row)


@dataclass
class Parameters:
  alpha: float  # step size
  epsilon: float  # e-greedy
  gamma: float  # discount factor


def select_action(q_table: np.ndarray, state: State, epsilon: float) -> Action:
  # using e-greedy method

  q: np.ndarray = q_table[*state]
  assert q.shape == (n_actions,)

  # exploratory
  if rng.random() < epsilon:
    a = rng.integers(0, n_actions)
  else:
    noise = rng.random(n_actions) * 1e-8  # so argmax randomly picks
    a = int(np.argmax(q + noise))

  return Action(a)


def update(
  q_table: np.ndarray,
  state: State,
  new_state: State,
  action: Action,
  reward: Reward,
  done: bool,
  params: Parameters,
):
  q_next = (1 - done) * np.max(q_table[*new_state])
  td_error = reward + params.gamma * q_next - q_table[*state]
  q_table[*state, action.value] += params.alpha * td_error


env = Environment(dim=(4, 4))

# init q table
q_table = np.zeros((*env.dim, 4))
params = Parameters(alpha=0.5, epsilon=0.05, gamma=1)
num_episodes = 50
max_steps = 1000


pbar_ep = tqdm(range(num_episodes), desc="Episodes")
for ep in range(num_episodes):
  env.reset()
  total_reward = 0

  for step in tqdm(
    range(max_steps), desc="Steps", leave=False
  ):  # loop until terminal state
    action = select_action(q_table, env.pos, params.epsilon)
    reward, done = env.step(action)
    update(q_table, env.prev_pos, env.pos, action, reward, done, params)

    total_reward += reward

    if done:
      break

  pbar_ep.set_postfix(reward=total_reward, eps=params.epsilon)
