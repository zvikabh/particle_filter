import dataclasses
import math
import random
import time

import matplotlib.pyplot as plt

# State and observation parameters.
SCREEN_X = 500
SCREEN_Y = 500
MAX_V = 5
OBSERVATION_NOISE = 10
NUM_PLAYERS = 4

# Particle filter parameters.
NUM_PARTICLES = 100


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


def choose_player_locations() -> list[Location]:
  """Select random initial locations for each of the players."""
  player_locations = []
  for i in range(NUM_PLAYERS):
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


def calc_weight(particle_state: State, obs: Location) -> float:
  x1 = particle_state.x
  x2 = obs.x
  y1 = particle_state.y
  y2 = obs.y
  distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
  weight = 1/(1 + 1/10*distance)
  return weight


def main():
  player_locations = choose_player_locations()
  state = choose_initial_state()
  obs = get_observations(state)
  particles = [choose_initial_state() for i in range(NUM_PARTICLES)]
  particle_weights = [1 for i in range(NUM_PARTICLES)]
  particle_player_locations = [choose_player_locations() for i in range(NUM_PARTICLES)]

  plt.ion()
  fig, ax = plt.subplots()
  ax.plot([0, SCREEN_X, SCREEN_X, 0, 0], [0, 0, SCREEN_Y, SCREEN_Y, 0], color='grey')
  ax.plot([loc.x for loc in player_locations], [loc.y for loc in player_locations], 'o', color='red')
  state_plot, = ax.plot([state.x], [state.y], '.', color='blue')
  obs_plot, = ax.plot([obs.x], [obs.y], 'o', color='green')
  particle_plot, = ax.plot([particle.x for particle in particles], [particle.y for particle in particles], '.', color='black')
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

    # Update the particles
    for i in range(NUM_PARTICLES):
      update_state(particles[i], particle_player_locations[i])
      particle_weights[i] = calc_weight(particles[i], obs)

    # Find the best particle
    max = 0
    max_index = None
    for i in range(NUM_PARTICLES):
      if particle_weights[i] > max:
        max = particle_weights[i]
        max_index = i
    best_particle = particles[max_index]

    # Kill particles more than 100 pixels from the observation, and replace them with particles based on the best one.
    for i in range(NUM_PARTICLES):
      if particle_weights[i] < 0.1:
        particles[i].x = best_particle.x
        particles[i].y = best_particle.y

    particle_plot.set_xdata([particle.x for particle in particles])
    particle_plot.set_ydata([particle.y for particle in particles])
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.01)


if __name__ == '__main__':
  main()
