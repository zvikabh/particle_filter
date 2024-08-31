import dataclasses
import random
import time

import matplotlib.pyplot as plt

SCREEN_X = 500
SCREEN_Y = 500
MAX_V = 5
OBSERVATION_NOISE = 10


@dataclasses.dataclass
class State:
  x: int
  y: int
  vx: int
  vy: int


@dataclasses.dataclass
class Location:
  x: int
  y: int


def choose_player_locations(num_players: int) -> list[Location]:
  """Select random initial locations for each of the players."""
  player_locations = []
  for i in range(num_players):
    player_locations.append(Location(x=random.randint(0, SCREEN_X), y=random.randint(0, SCREEN_Y)))
  return player_locations


def choose_initial_state() -> State:
  s = State(x=random.randint(0, SCREEN_X), y=random.randint(0, SCREEN_Y), vx=random.randint(-MAX_V, MAX_V),
            vy=random.randint(-MAX_V, MAX_V))
  return s


def is_ball_near_player(state: State, player_locations: list[Location]) -> bool:
  for player_location in player_locations:
    if abs(state.x - player_location.x) < 20 and abs(state.y - player_location.y) < 20:
      return True
  return False


def is_ball_on_x_edge(state) -> bool:
  if state.x <= 0 or state.x >= SCREEN_X:
    return True
  return False


def is_ball_on_y_edge(state) -> bool:
  if state.y <= 0 or state.y >= SCREEN_Y:
    return True
  return False


def update_state(state: State, player_locations: list[Location]) -> None:
  if is_ball_near_player(state, player_locations):
    state.vx = random.randint(-MAX_V, MAX_V)
    state.vy = random.randint(-MAX_V, MAX_V)
  if is_ball_on_x_edge(state):
    state.vx = -state.vx
  if is_ball_on_y_edge(state):
    state.vy = -state.vy
  state.x += state.vx
  state.y += state.vy


def get_observations(state: State) -> Location:
  loc = Location(x=state.x + random.randint(-OBSERVATION_NOISE, OBSERVATION_NOISE),
                 y=state.y + random.randint(-OBSERVATION_NOISE, OBSERVATION_NOISE))
  return loc


def main():
  player_locations = choose_player_locations(num_players=4)
  state = choose_initial_state()
  obs = get_observations(state)

  plt.ion()
  fig, ax = plt.subplots()
  ax.plot([0, SCREEN_X, SCREEN_X, 0, 0], [0, 0, SCREEN_Y, SCREEN_Y, 0], color='grey')
  ax.plot([loc.x for loc in player_locations], [loc.y for loc in player_locations], 'o', color='red')
  state_plot, = ax.plot([state.x], [state.y], '.', color='blue')
  obs_plot, = ax.plot([obs.x], [obs.y], 'o', color='green')
  fig.canvas.draw()
  fig.canvas.flush_events()

  time.sleep(0.01)
  for i in range(1000):
    update_state(state, player_locations)
    obs = get_observations(state)
    obs_plot.set_xdata([obs.x])
    obs_plot.set_ydata([obs.y])
    state_plot.set_xdata([state.x])
    state_plot.set_ydata([state.y])
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.01)


if __name__ == '__main__':
  main()
