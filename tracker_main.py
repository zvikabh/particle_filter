# Particle filter for tracking a bouncing ball.
# Joint work with Naama.

import dataclasses
import math
import random
import time

import matplotlib.pyplot as plt

# State and observation parameters.
SCREEN_X = 500
SCREEN_Y = 500
MAX_V = 10
OBSERVATION_NOISE = 10
NUM_PLAYERS = 7

# Particle filter parameters.
NUM_PARTICLES = 100
DETECTION_THRESHOLD = 16


@dataclasses.dataclass
class State:
  x: int
  y: int
  vx: int
  vy: int


@dataclasses.dataclass
class Location:
  x: float
  y: float


def choose_player_locations() -> list[Location]:
  """Select random initial locations for each of the players."""
  player_locations = []
  for i in range(NUM_PLAYERS):
    player_locations.append(Location(x=random.randint(50, SCREEN_X-50), y=random.randint(50, SCREEN_Y-50)))
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


def update_state(state: State, player_locations: list[Location]) -> bool:
  """Returns whether you bounced off a player."""
  ret_value = False
  if is_ball_near_player(state, player_locations):
    state.vx = random.randint(-MAX_V, MAX_V)
    state.vy = random.randint(-MAX_V, MAX_V)
    ret_value = True
  if is_ball_on_x_edge(state):
    state.vx = -state.vx
  if is_ball_on_y_edge(state):
    state.vy = -state.vy
  state.x += state.vx
  state.y += state.vy
  return ret_value


def get_observations(state: State) -> Location:
  loc = Location(x=state.x + random.randint(-OBSERVATION_NOISE, OBSERVATION_NOISE),
                 y=state.y + random.randint(-OBSERVATION_NOISE, OBSERVATION_NOISE))
  return loc


def calc_weight(particle_state: State, obs: Location) -> float:
  distance = calc_distance(particle_state, obs)
  weight = 1 / (1 + 1 / 10 * distance)
  return weight


def calc_distance(pt1: Location | State, pt2: Location | State) -> float:
  distance = math.sqrt((pt1.x - pt2.x) ** 2 + (pt1.y - pt2.y) ** 2)
  return distance


def update_particles(particles: list[State], particle_weights: list[float],
                     particle_player_locations: list[list[Location]], obs: Location) -> None:
  # Update the particles
  for i in range(NUM_PARTICLES):
    update_state(particles[i], particle_player_locations[i])
    particle_weights[i] = calc_weight(particles[i], obs)

  # Kill particles more than X pixels from the observation, and replace them with particles based on the best one.
  best_particle = find_best_particle(particles, particle_weights)
  for i in range(NUM_PARTICLES):
    if particle_weights[i] < 0.3:
      particles[i].x = best_particle.x + random.randint(-5, 5)
      particles[i].y = best_particle.y + random.randint(-5, 5)
      particles[i].vx = best_particle.vx + random.randint(-2, 2)
      particles[i].vy = best_particle.vy + random.randint(-2, 2)


def find_best_particle(particles, particle_weights):
  # Find the best particle
  max = 0
  max_index = None
  for i in range(NUM_PARTICLES):
    if particle_weights[i] > max:
      max = particle_weights[i]
      max_index = i
  best_particle = particles[max_index]
  return best_particle


def calc_average(sequence: list[float]) -> float:
  sum = 0
  for num in sequence:
    sum += num
  avg = sum/len(sequence)
  return avg


def main():
  player_locations = choose_player_locations()
  state = choose_initial_state()
  obs = get_observations(state)
  particles = [choose_initial_state() for i in range(NUM_PARTICLES)]
  particle_weights = [1 for i in range(NUM_PARTICLES)]
  particle_player_locations = [choose_player_locations() for i in range(NUM_PARTICLES)]
  avg_distances = []

  plt.ion()
  fig, ax = plt.subplots()
  ax.plot([0, SCREEN_X, SCREEN_X, 0, 0], [0, 0, SCREEN_Y, SCREEN_Y, 0], color='grey')
  ax.plot([loc.x for loc in player_locations], [loc.y for loc in player_locations], 'o', color='red')
  particle_plot, = ax.plot([particle.x for particle in particles], [particle.y for particle in particles], '.',
                           color=('black', 0.2))
  state_plot, = ax.plot([state.x], [state.y], 'o', color='orange')
  obs_plot, = ax.plot([obs.x], [obs.y], 'o', color='green')
  fig.canvas.draw()
  fig.canvas.flush_events()

  time.sleep(0.01)
  bounce_times = []
  suspected_players = []
  suspected_bounce_times = []
  for i in range(1000):
    if update_state(state, player_locations):
      bounce_times.append(i)
    obs = get_observations(state)
    obs_plot.set_xdata([obs.x])
    obs_plot.set_ydata([obs.y])
    state_plot.set_xdata([state.x])
    state_plot.set_ydata([state.y])

    update_particles(particles, particle_weights, particle_player_locations, obs)

    avg_distance = calc_average([calc_distance(particle, state) for particle in particles])
    avg_distances.append(avg_distance)

    if avg_distance >= DETECTION_THRESHOLD and i > 50:
      # suspected_players.append(Location(
      #   x=calc_average([particle.x for particle in particles]),
      #   y=calc_average([particle.y for particle in particles])
      # ))
      best_particle = find_best_particle(particles, particle_weights)
      suspected_players.append(Location(x=best_particle.x, y=best_particle.y))
      suspected_bounce_times.append(i)


    particle_plot.set_xdata([particle.x for particle in particles])
    particle_plot.set_ydata([particle.y for particle in particles])
    ax.set_title(f'Avg distance: {avg_distance:.0f} pixels')
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.01)

  plt.close(fig)
  plt.ioff()
  fig2, ax2 = plt.subplots()
  ax2.plot(bounce_times, [0]*len(bounce_times), 'o', color='red')
  ax2.plot(suspected_bounce_times, [DETECTION_THRESHOLD]*len(suspected_bounce_times), 'o', color='orange')
  ax2.plot(avg_distances, color='blue')
  plt.xlabel('Time')
  plt.ylabel('Average particle error (pixels)')
  plt.show()

  fig3, ax3 = plt.subplots()
  ax3.plot([loc.x for loc in player_locations], [loc.y for loc in player_locations], 'o', color='red')
  ax3.plot([loc.x for loc in suspected_players], [loc.y for loc in suspected_players], 'x', color='blue')
  plt.show()



if __name__ == '__main__':
  main()
