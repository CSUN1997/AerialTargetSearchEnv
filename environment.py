import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, List
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class DroneSearchEnv(gym.Env):
    """
    Multi-agent environment for drone target search missions.
    Each agent receives an egocentric grid observation and has a continuous action space.
    """

    def __init__(
            self,
            num_agents: int = 3,
            grid_size: int = 20,
            observation_radius: int = 5,
            num_targets: int = 3,
            obstacle_density: float = 0.1,
            max_steps: int = 500,
            comm_fail=0,
            num_agents_adjusted=None
    ):
        super().__init__()

        self.num_agents = num_agents
        self.grid_size = grid_size
        self.observation_radius = observation_radius
        self.num_targets = num_targets
        self.obstacle_density = obstacle_density
        self.max_steps = max_steps
        self.comm_fail = comm_fail
        self.num_agents_adjusted = num_agents_adjusted

        # Action space: [delta_x, delta_y] for each agent
        # Continuous values between -1 and 1
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        obs_shape = (observation_radius * 2 + 1, observation_radius * 2 + 1)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=obs_shape,
            dtype=np.float32
        )

        # Initialize state variables
        self.positions = None
        self.agent_trajectories = None
        self.velocities = None
        self.obstacles = None
        self.targets = None
        self.visited = None
        self.self_visited = None
        self.steps = 0

        # Visualization setup
        self.fig = None
        self.ax = None
        self.agent_colors = plt.cm.Set3(np.linspace(0, 1, num_agents))

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[dict] = None
    ):
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Generate random obstacle positions
        self.obstacles = self.np_random.random((self.grid_size, self.grid_size)) < self.obstacle_density

        # Initialize agent positions randomly in free cells
        self.positions = []
        free_cells = np.argwhere(~self.obstacles)
        agent_indices = self.np_random.choice(len(free_cells), size=self.num_agents, replace=False)
        for idx in agent_indices:
            self.positions.append(free_cells[idx].astype(float))
        self.positions = np.array(self.positions)

        self.agent_trajectories = [[] for _ in range(self.num_agents)]
        for i, pos in enumerate(self.positions):
            self.agent_trajectories[i].append(pos.copy())

        self.velocities = np.zeros((self.num_agents, 2))

        # Place targets randomly in free cells
        remaining_cells = [i for i in range(len(free_cells)) if i not in agent_indices]
        target_indices = self.np_random.choice(remaining_cells, size=self.num_targets, replace=False)
        self.targets = free_cells[target_indices]

        # global visited cells
        self.visited = np.zeros((self.grid_size, self.grid_size))
        # agent individual visited cells
        self.self_visited = [np.zeros((self.grid_size, self.grid_size)) for _ in range(self.num_agents)]
        for i, pos in enumerate(self.positions):
            self._mark_visited(i, pos)

        self.steps = 0

        # Reset visualization
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

        # Get initial observations for each agent
        neighbors = self.discover_neighbors()
        observations = [self._get_agent_observation(neighbors[i], i) for i in range(self.num_agents)]

        return observations

    def step(self, actions):
        """Execute one time step with given actions."""
        self.steps += 1

        # Update positions and velocities
        rewards = [None for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            action = actions[i]
            action = np.clip(action, -1, 1)

            # Update velocity (with some inertia)
            self.velocities[i] = action

            # Update position
            old_pos = self.positions[i].copy()
            new_pos = old_pos + self.velocities[i]
            self.positions[i] = new_pos

            # Check bounds and obstacles
            if self._is_valid_position(new_pos):
                reward = self._calculate_reward(i, old_pos, new_pos)
                self._mark_visited(i, self.positions[i])
            else:
                reward = -0.5  # Penalty for invalid move

            rewards[i] = reward

        # Update agent trajectories
        for i, pos in enumerate(self.positions):
            self.agent_trajectories[i].append(pos.copy())

        neighbors = self.discover_neighbors()

        # Get observations
        observations = [self._get_agent_observation(neighbors[i], i) for i in range(self.num_agents)]
        observations_from_neighbors = [[observations[j] for j in neighbors[i]] for i in range(self.num_agents)]

        # Check truncation conditions
        truncated = [self.steps >= self.max_steps for _ in range(self.num_agents)]

        return observations, observations_from_neighbors, rewards, truncated

    def discover_neighbors(self) -> List[List[int]]:
        """
        We used the Bernoulli communication model, where the communication between any two agents fail with a probability p
        """
        adj_mat = np.ones([self.num_agents, self.num_agents])
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if np.random.rand() < self.comm_fail:
                    adj_mat[i, j] = 0
                    adj_mat[j, i] = 0
        neighbors = []
        for i in range(self.num_agents):
            n = np.nonzero(adj_mat[i, :])[0]
            neighbors.append(n.tolist())
        return neighbors

    def _get_agent_observation(self, neighbors: List[int], agent_idx: int) -> np.ndarray:
        """
        Get egocentric grid observation for given agent.
        The gird is a square, with each entry indicating different status of the cell as follows:
        - 0: unvisited and obstacle free
        - 1: visited before
        - 2: obstacle or out of boundary
        """
        pos = self.positions[agent_idx].astype(int)
        rad = self.observation_radius

        # Initialize observation channels
        obs = np.zeros((rad * 2 + 1, rad * 2 + 1))

        # Fill in observable region
        for i in range(-rad, rad + 1):
            for j in range(-rad, rad + 1):
                x, y = pos[0] + i, pos[1] + j
                obs_x, obs_y = i + rad, j + rad
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    if self.obstacles[x, y] != 0:
                        obs[obs_x, obs_y] = 2
                    else:
                        visited_by_others = False
                        for neighbor in neighbors:
                            # cell visited by neighboring agents
                            if self.self_visited[neighbor][x, y] == 1:
                                visited_by_others = True
                                break
                        if visited_by_others or self.self_visited[agent_idx][x, y] != 0:
                            obs[obs_x, obs_y] = 1
                else:
                    # Outside grid bounds - use padding
                    obs[obs_x, obs_y] = 2  # Treat out-of-bounds as obstacles

        obs = np.concatenate(
            [[(self.positions[agent_idx][0]) / self.grid_size, (20 - self.positions[agent_idx][0]) / self.grid_size,
              (self.positions[agent_idx][1]) / self.grid_size, (20 - self.positions[agent_idx][1]) / self.grid_size],
             (self.positions[agent_idx] - 10) / self.grid_size,
             obs.flatten()])
        # obs = obs.flatten()
        return obs

    def _calculate_reward(self, agent_idx: int, old_pos: np.ndarray, new_pos: np.ndarray) -> float:
        """Calculate reward for an agent's action."""
        reward = 0.0

        # Reward for exploring new cells
        new_pos_int = new_pos.astype(int)
        if self.visited[new_pos_int[0], new_pos_int[1]] == 0:
            reward += 1.0
        else:
            reward -= 0.3

        # Penalty for overlapping with other agents
        for i, pos in enumerate(self.positions):
            if i != agent_idx and np.linalg.norm(pos - new_pos) < 1.0:
                reward -= 0.5

        # Reward for finding targets
        # for target in self.targets:
        #     if tuple(new_pos_int) == tuple(target):
        #         reward += 5.0

        return reward

    def inter_agent_collision(self):
        collision = [0 for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            pos_i = self.positions[i]
            for j in range(i + 1, self.num_agents):
                pos_j = self.positions[j]
                if j != i and np.linalg.norm(pos_i - pos_j) < 0.2:
                    collision[i] = 1
                    collision[j] = 1
        return collision

    def _is_valid_position(self, pos: np.ndarray) -> bool:
        """Check if a position is valid (within bounds and not in obstacle)."""
        if not (0 <= pos[0] < self.grid_size - 1 and 0 <= pos[1] < self.grid_size - 1):
            return False

        # Check if position intersects with obstacles
        pos_int = pos.astype(int)
        return not self.obstacles[pos_int[0], pos_int[1]]

    def _mark_visited(self, agent_idx, pos: np.ndarray):
        if self.num_agents_adjusted and agent_idx >= self.num_agents_adjusted:
            return
        """Mark cells as visited around the given position."""
        pos_int = pos.astype(int)
        self.visited[pos_int[0], pos_int[1]] = 1
        self.self_visited[agent_idx][pos_int[0], pos_int[1]] = 1

    def render(self):
        """
        Render the environment using matplotlib.
        Shows obstacles, agents, targets, and visited areas.
        """
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            plt.ion()

        self.ax.clear()

        # Plot obstacles
        obstacle_map = self.obstacles.copy()
        obstacle_map = np.ma.masked_where(obstacle_map == 0, obstacle_map)
        # self.ax.imshow(obstacle_map.T, cmap='Greys', alpha=0.7, origin='lower')
        self.ax.imshow(obstacle_map.T)

        for i in range(self.num_agents):
            if len(self.agent_trajectories[i]) > 1:  # Need at least 2 points to plot a line
                trajectory = np.array(self.agent_trajectories[i])
                self.ax.plot(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    color=self.agent_colors[i],
                    # alpha=0.3,
                    linewidth=1,
                    linestyle=':'
                )

        # Plot agents with their observation radius
        for i, pos in enumerate(self.positions):
            # Draw observation radius
            circle = Circle(
                (pos[0], pos[1]),
                self.observation_radius,
                fill=False,
                linestyle='--',
                color=self.agent_colors[i],
                alpha=0.3
            )
            self.ax.add_patch(circle)

            # Draw agent
            self.ax.scatter(
                pos[0],
                pos[1],
                color=self.agent_colors[i],
                s=100,
                label=f'Agent {i}'
            )

        # Plot targets
        self.ax.scatter(
            self.targets[:, 0],
            self.targets[:, 1],
            color='red',
            marker='*',
            s=200,
            label='Targets'
        )

        # Set plot properties
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_title(f'Step {self.steps}')
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

    def close(self):
        """Close the visualization window."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


if __name__ == "__main__":
    env = DroneSearchEnv(num_agents=2)
    obs = env.reset()
    actions = [np.random.rand(2) for _ in range(2)]
    next_obs, obs_others, reward, done = env.step(actions)
