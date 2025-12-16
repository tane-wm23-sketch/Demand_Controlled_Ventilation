import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional

class VentilationEnvironment(gym.Env):
    """
    Gymnasium environment for Occupancy-Based Demand-Controlled Ventilation.

    Objectives:
    1. Maintain CO2 close to outdoor levels (406 ppm)
    2. Minimize energy consumption
    3. Avoid frequent fan speed changes
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 96,
        dt: float = 5.0,
        room_area: float = 50.0,
        room_height: float = 3.0,
        co2_setpoint: float = 1500.0,
        co2_outdoor: float = 406.0,
        k_iaq: float = 5.0,
        k_energy: float = 0.01,
        k_switch: float = 0.5,
        max_fan_power: float = 325.0,
        max_fan_air_volume: float = 800.0,
        min_hold_steps: int = 3,
        discrete_actions_count: int = 5,
        min_meeting_duration: float = 30.0,
        max_meeting_duration: float = 120.0,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.dt = dt
        self.room_area = room_area
        self.room_height = room_height
        self.room_volume = self.room_area * self.room_height
        ashrae_occupancy_per_sqm = 50 / 100
        self.max_occupancy = max(1, int(np.floor(self.room_area * ashrae_occupancy_per_sqm)))

        self.co2_outdoor = co2_outdoor
        self.co2_setpoint = co2_setpoint+ self.co2_outdoor
        self.k_iaq = k_iaq
        self.k_energy = k_energy
        self.k_switch = k_switch
        self.co2_per_person = 0.3125

        self.min_meeting_duration = min_meeting_duration
        self.max_meeting_duration = max_meeting_duration
        self.is_meeting_active = False
        self.meeting_occupancy_count = 0
        self.meeting_remaining_steps = 0

        self.max_fan_power = max_fan_power
        self.max_fan_air_volume = max_fan_air_volume
        self.max_ventilation_rate_m3_per_min = self.max_fan_air_volume / 60.0

        # 4D observation space: [CO2, fan_speed, light, occupancy]
        self.observation_space = spaces.Box(
            low=np.array([406.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([5000.0, 1.0, 1.0, float(self.max_occupancy)], dtype=np.float32),
            dtype=np.float32
        )

        self.discrete_actions_count = discrete_actions_count
        self.action_space = spaces.Discrete(self.discrete_actions_count)
        self.discrete_actions = np.linspace(0.0, 1.0, self.discrete_actions_count)

        self.current_step = 0
        self.co2_level = 406.0
        self.fan_speed = 0.0
        self.previous_fan_speed = 0.0
        self.light_level = 0.0
        self.occupancy = 0
        self.min_hold_steps = 6
        self.hold_counter = 0

        self.total_reward = 0.0
        self.iaq_violations = 0
        self.energy_consumed = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.co2_level = 406.0
        self.fan_speed = 0.0
        self.previous_fan_speed = 0.0
        self.light_level = 0.0
        self.occupancy = 0
        self.is_meeting_active = False
        self.meeting_occupancy_count = 0
        self.meeting_remaining_steps = 0
        self.total_reward = 0.0
        self.iaq_violations = 0
        self.energy_consumed = 0.0
        self.hold_counter = 0

        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action_idx: int):
        target_fan_speed  = self.discrete_actions[action_idx]

        if target_fan_speed != self.fan_speed:
            if self.hold_counter >= self.min_hold_steps:
                self.fan_speed = target_fan_speed
                self.hold_counter = 0
            else:
                self.hold_counter += 1
        else:
            self.hold_counter += 1


        self._update_occupancy_and_light()
        self._update_co2_level(self.fan_speed)
        reward = float(self._calculate_reward(self.fan_speed))
        self.total_reward += reward

        if self.co2_level > self.co2_setpoint:
            self.iaq_violations += 1

        self.energy_consumed += (self.fan_speed * self.max_fan_power) * (self.dt / 60.0)
        self.previous_fan_speed = self.fan_speed

        self.current_step += 1
        terminated = self.co2_level > self.co2_setpoint
        truncated = self.current_step >= self.max_steps
        observation = self._get_observation()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        return np.array([
            self.co2_level,
            self.fan_speed,
            self.light_level,
            float(self.occupancy)
        ], dtype=np.float32)

    def _get_info(self):
        return {
            'step': self.current_step,
            'co2_level': self.co2_level,
            'fan_speed': self.fan_speed,
            'light_level': self.light_level,
            'occupancy': self.occupancy,
            'total_reward': self.total_reward,
            'iaq_violations': self.iaq_violations,
            'iaq_compliance_rate': 1.0 - (self.iaq_violations / max(1, self.current_step)),
            'energy_consumed': self.energy_consumed,
            'is_meeting_active': self.is_meeting_active
        }

    def _update_occupancy_and_light(self):
        current_sim_minutes = self.current_step * self.dt
        current_sim_hour = current_sim_minutes / 60.0
        absolute_hour = 8 + current_sim_hour

        if self.is_meeting_active:
            self.occupancy = self.meeting_occupancy_count
            self.light_level = 1.0
            self.meeting_remaining_steps -= 1
            if self.meeting_remaining_steps <= 0:
                self.is_meeting_active = False
                self.meeting_occupancy_count = 0
        else:
            meeting_probability = self._get_meeting_probability(absolute_hour)
            if self.np_random.random() < meeting_probability:
                meeting_duration_minutes = self.np_random.uniform(
                    self.min_meeting_duration, self.max_meeting_duration
                )
                self.meeting_remaining_steps = max(1, int(meeting_duration_minutes / self.dt))
                self.meeting_occupancy_count = self.np_random.integers(2, self.max_occupancy + 1)
                self.is_meeting_active = True
                self.occupancy = self.meeting_occupancy_count
                self.light_level = 1.0
            else:
                if self.np_random.random() < 0.7:
                    self.occupancy = 0
                    self.light_level = 0.0
                else:
                    max_partial = min(3, self.max_occupancy)
                    self.occupancy = self.np_random.integers(1, max_partial + 1)
                    self.light_level = 1.0

    def _get_meeting_probability(self, hour: float):
        if 9 <= hour < 11:
            return 0.6
        elif 14 <= hour < 16:
            return 0.7
        elif 11 <= hour < 14:
            return 0.3
        else:
            return 0.4

    def _update_co2_level(self, fan_speed: float):
      co2_generation = self.occupancy * self.co2_per_person

      ventilation_rate = fan_speed * self.max_ventilation_rate_m3_per_min
      co2_removal = ventilation_rate * (self.co2_level - self.co2_outdoor) / 1000.0
      net_co2_change = ((co2_generation - co2_removal) / self.room_volume) * 1000.0
      self.co2_level += net_co2_change * self.dt
      self.co2_level = np.clip(self.co2_level, self.co2_outdoor, 5000.0)

    def _calculate_reward(self, fan_speed: float):
        # 1. Air-quality satisfaction with adjustable weight

        r_iaq = self.k_iaq * (1 - min(abs(self.co2_level - self.co2_outdoor) / 1000.0, 1.0))

        # 2. Energy penalty
        r_energy = - self.k_energy * fan_speed * self.max_fan_power

        # 3. Switching penalty: penalize any fan speed change
        fan_change = abs(fan_speed - self.previous_fan_speed)
        r_switch = - self.k_switch * fan_change
        total_reward = r_iaq + r_energy + r_switch
        return total_reward

    def render(self):
        if self.render_mode == 'human':
            print(f"\n{'='*60}")
            print(f"Step: {self.current_step} | Time: {(self.current_step * self.dt / 60):.1f} hours")
            print(f"{'='*60}")
            print(f"CO2 Level:      {self.co2_level:.1f} ppm (Target: {self.co2_outdoor:.0f})")
            print(f"CO2 Distance:   {abs(self.co2_level - self.co2_outdoor):.1f} ppm")
            print(f"Fan Speed:      {self.fan_speed*100:.1f}% (Prev: {self.previous_fan_speed*100:.1f}%)")
            print(f"Light Level:    {'ON' if self.light_level > 0.5 else 'OFF'}")
            print(f"Occupancy:      {self.occupancy} people (Max: {self.max_occupancy})")
            print(f"Meeting:        {'Yes' if self.is_meeting_active else 'No'}")
            print(f"{'='*60}")
            print(f"Total Reward:   {self.total_reward:.2f}")
            print(f"IAQ Violations: {self.iaq_violations}")
            print(f"Energy Used:    {self.energy_consumed:.2f} Wh")
            print(f"{'='*60}\n")

    def close(self):
        pass


# #Exploration Strategies


import numpy as np
import math
from collections import defaultdict
from abc import ABC, abstractmethod


# ==================== UNIFIED EXPLORATION STRATEGIES ====================

class ExplorationStrategy(ABC):
    """Base class for exploration strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_action(self, q_values, step: int = 0) -> int:
        """
        Select action based on Q-values.

        Args:
            q_values: Either dict or np.ndarray of Q-values
            step: Current step (for strategies that need it)

        Returns:
            Selected action index
        """
        pass

    @abstractmethod
    def update(self, episode: int = 0, state: tuple = None, action: int = None, reward: float = None):
        """
        Update exploration parameters.

        Args:
            episode: Current episode number
            state: Current state (for UCB tracking)
            action: Taken action (for UCB tracking)
            reward: Received reward
        """
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """Return current parameters as dictionary."""
        pass


class RandomExploration(ExplorationStrategy):
    """Pure random exploration - selects actions uniformly at random."""

    def __init__(self, n_actions: int = 5):
        super().__init__("Random")
        self.n_actions = n_actions

    def select_action(self, q_values, step: int = 0) -> int:
        # Handle both dict and array formats
        if isinstance(q_values, dict):
            return np.random.choice(list(q_values.keys()))
        else:
            return np.random.randint(len(q_values))

    def update(self, episode: int = 0, state: tuple = None, action: int = None, reward: float = None):
        pass  # Random strategy doesn't need updates

    def get_params(self) -> dict:
        return {}


class EpsilonGreedy(ExplorationStrategy):
    """
    Epsilon-greedy exploration strategy.
    With probability epsilon, select random action; otherwise select greedy action.
    """

    def __init__(self, epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995):
        super().__init__("Epsilon-Greedy")
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.initial_epsilon = epsilon

    def select_action(self, q_values, step: int = 0) -> int:
        # Explore with probability epsilon
        if np.random.random() < self.epsilon:
            if isinstance(q_values, dict):
                return np.random.choice(list(q_values.keys()))
            else:
                return np.random.randint(len(q_values))

        # Exploit: select best action
        if isinstance(q_values, dict):
            max_q = max(q_values.values())
            max_actions = [a for a, q in q_values.items() if q == max_q]
            return np.random.choice(max_actions)
        else:
            return int(np.argmax(q_values))

    def update(self, episode: int = 0, state: tuple = None, action: int = None, reward: float = None):
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_params(self) -> dict:
        return {
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay
        }

    def reset(self):
        """Reset epsilon to initial value."""
        self.epsilon = self.initial_epsilon


class UCB(ExplorationStrategy):
    """
    Upper Confidence Bound (UCB) exploration.
    Balances exploitation and exploration using confidence bounds.
    """

    def __init__(self, c: float = 2.0, n_actions: int = 5):
        super().__init__("UCB")
        self.c = c
        self.n_actions = n_actions

        # For dict-based Q-tables (SARSA)
        self.state_action_counts = defaultdict(lambda: defaultdict(int))
        self.state_total_counts = defaultdict(int)

        # For array-based Q-tables (Dyna-Q)
        self.action_counts = defaultdict(int)
        self.total_steps = 0

    def select_action(self, q_values, step: int = 0) -> int:
        if isinstance(q_values, dict):
            return self._select_action_dict(q_values)
        else:
            return self._select_action_array(q_values)

    def _select_action_dict(self, q_values: dict) -> int:
        """Select action for dict-based Q-values (SARSA)."""
        state_key = tuple(sorted(q_values.items()))

        # Try each action at least once
        for action in q_values.keys():
            if self.state_action_counts[state_key][action] == 0:
                return action

        # Calculate UCB values
        total = self.state_total_counts[state_key]
        ucb_values = {}

        for action, q_value in q_values.items():
            count = self.state_action_counts[state_key][action]
            exploration_bonus = self.c * np.sqrt(np.log(total + 1) / (count + 1e-5))
            ucb_values[action] = q_value + exploration_bonus

        return max(ucb_values, key=ucb_values.get)

    def _select_action_array(self, q_values: np.ndarray) -> int:
        """Select action for array-based Q-values (Dyna-Q)."""
        self.total_steps += 1

        # Try each action at least once
        for a in range(len(q_values)):
            if self.action_counts[a] == 0:
                self.action_counts[a] += 1
                return a

        # Calculate UCB values
        ucb_values = np.zeros(len(q_values))
        for a in range(len(q_values)):
            exploration_bonus = self.c * math.sqrt(
                math.log(self.total_steps) / self.action_counts[a]
            )
            ucb_values[a] = q_values[a] + exploration_bonus

        action = int(np.argmax(ucb_values))
        self.action_counts[action] += 1
        return action

    def update(self, episode: int = 0, state: tuple = None, action: int = None, reward: float = None):
        # Update counts for dict-based Q-tables (SARSA)
        if state is not None and action is not None:
            self.state_action_counts[state][action] += 1
            self.state_total_counts[state] += 1

        # Reset for new episode (Dyna-Q style)
        if episode > 0:
            self.action_counts = defaultdict(int)
            self.total_steps = 0

    def get_params(self) -> dict:
        return {
            'c': self.c,
            'total_steps': self.total_steps
        }


class Softmax(ExplorationStrategy):
    """
    Softmax (Boltzmann) exploration.
    Selects actions probabilistically based on their Q-values.
    """

    def __init__(self, temperature: float = 1.0, temp_min: float = 0.01,
                 temp_decay: float = 0.995):
        super().__init__("Softmax")
        self.temperature = temperature
        self.temp_min = temp_min
        self.temp_decay = temp_decay
        self.initial_temperature = temperature

    def select_action(self, q_values, step: int = 0) -> int:
        if isinstance(q_values, dict):
            actions = list(q_values.keys())
            q_vals = np.array([q_values[a] for a in actions])
        else:
            actions = list(range(len(q_values)))
            q_vals = q_values

        # Numerical stability: subtract max before exponentiating
        q_vals_shifted = q_vals - np.max(q_vals)
        exp_values = np.exp(q_vals_shifted / self.temperature)
        probabilities = exp_values / np.sum(exp_values)

        return np.random.choice(actions, p=probabilities)

    def update(self, episode: int = 0, state: tuple = None, action: int = None, reward: float = None):
        # Decay temperature
        self.temperature = max(self.temp_min, self.temperature * self.temp_decay)

    def get_params(self) -> dict:
        return {
            'temperature': self.temperature,
            'temp_min': self.temp_min,
            'temp_decay': self.temp_decay
        }

    def reset(self):
        """Reset temperature to initial value."""
        self.temperature = self.initial_temperature


# ==================== STRATEGY FACTORY ====================

def create_exploration_strategy(strategy_name: str, n_actions: int = 5,
                               **params) -> ExplorationStrategy:
    """
    Factory function to create exploration strategies.

    Args:
        strategy_name: One of 'random', 'epsilon_greedy', 'ucb', 'softmax'
        n_actions: Number of actions in the environment
        **params: Strategy-specific parameters

    Returns:
        ExplorationStrategy instance

    Example:
        >>> strategy = create_exploration_strategy('epsilon_greedy',
        ...                                        epsilon=1.0, epsilon_decay=0.995)
    """
    strategies = {
        'random': RandomExploration,
        'epsilon_greedy': EpsilonGreedy,
        'ucb': UCB,
        'softmax': Softmax
    }

    if strategy_name not in strategies:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Choose from {list(strategies.keys())}"
        )

    # Create strategy with appropriate parameters
    if strategy_name == 'random':
        return RandomExploration(n_actions=n_actions)

    elif strategy_name == 'epsilon_greedy':
        return EpsilonGreedy(
            epsilon=params.get('epsilon', 1.0),
            epsilon_min=params.get('epsilon_min', 0.01),
            epsilon_decay=params.get('epsilon_decay', 0.995)
        )

    elif strategy_name == 'ucb':
        return UCB(
            c=params.get('c', 2.0),
            n_actions=n_actions
        )

    elif strategy_name == 'softmax':
        return Softmax(
            temperature=params.get('temperature', 1.0),
            temp_min=params.get('temp_min', 0.01),
            temp_decay=params.get('temp_decay', 0.995)
        )

# # Dyna-Q


import numpy as np
from collections import defaultdict, deque
from typing import Tuple, Optional, List
import pickle
import math
from abc import ABC, abstractmethod

# ==================== DYNA-Q AGENT ====================

class DynaQAgent:
    """
    Dyna-Q agent enhanced with Experience Replay.

    Combines model-based planning (Dyna-Q) with experience replay for improved
    sample efficiency and learning stability.
    """

    def __init__(
        self,
        n_actions: int = 5,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        planning_steps: int = 5,
        state_discretization: Tuple[int, int, int, int] = (50, 10, 2, 20),
        exploration_strategy: str = "epsilon_greedy",
        # Experience Replay parameters
        use_replay: bool = True,
        replay_buffer_size: int = 10000,
        replay_batch_size: int = 32,
        replay_frequency: int = 1,  # How often to sample from replay buffer
        min_replay_size: int = 100,  # Minimum experiences before replay
        use_prioritized_replay: bool = False,
        **strategy_params
    ):
        """
        Initialize Dyna-Q agent with Experience Replay.

        Args:
            n_actions: Number of discrete actions
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            planning_steps: Number of planning steps per real experience
            state_discretization: Bins for discretizing state space
            exploration_strategy: 'random', 'epsilon_greedy', 'ucb', or 'softmax'
            use_replay: Whether to use experience replay
            replay_buffer_size: Maximum size of replay buffer
            replay_batch_size: Number of experiences to sample per replay
            replay_frequency: How many steps between replay updates
            min_replay_size: Minimum buffer size before starting replay
            use_prioritized_replay: Use prioritized experience replay
            **strategy_params: Parameters for the exploration strategy
        """
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.planning_steps = planning_steps

        # State discretization parameters
        self.co2_bins = state_discretization[0]
        self.fan_bins = state_discretization[1]
        self.light_bins = state_discretization[2]
        self.occupancy_bins = state_discretization[3]

        # Q-table and Model
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.model = {}
        self.visited_states = set()

        # Experience Replay
        self.use_replay = use_replay
        self.replay_batch_size = replay_batch_size
        self.replay_frequency = replay_frequency
        self.min_replay_size = min_replay_size
        self.step_counter = 0

        if self.use_replay:
            if use_prioritized_replay:
                self.replay_buffer = PrioritizedExperienceReplay(
                    capacity=replay_buffer_size
                )
            else:
                self.replay_buffer = ExperienceReplayBuffer(
                    capacity=replay_buffer_size
                )
        else:
            self.replay_buffer = None

        # Initialize exploration strategy
        self.exploration_strategy = self._create_strategy(
            exploration_strategy, n_actions, **strategy_params
        )

        # Statistics
        self.episode_rewards = []
        self.episode_steps = []
        self.replay_updates = 0

        # Action statistics
        self.action_counts = np.zeros(n_actions)
        self.episode_action_counts = []
        self.current_episode_actions = []

    def _create_strategy(self, strategy_name: str, n_actions: int, **params) -> ExplorationStrategy:
        """Create exploration strategy based on name."""
        strategies = {
            'random': RandomExploration,
            'epsilon_greedy': EpsilonGreedy,
            'ucb': UCB,
            'softmax': Softmax
        }
        if strategy_name not in strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}. Choose from {list(strategies.keys())}")
        if strategy_name == 'random':
            return strategies[strategy_name](n_actions=n_actions)
        elif strategy_name == 'epsilon_greedy':
            return strategies[strategy_name](
                epsilon=params.get('epsilon', 1.0),
                epsilon_min=params.get('epsilon_min', 0.01),
                epsilon_decay=params.get('epsilon_decay', 0.995)
            )
        elif strategy_name == 'ucb':
            return strategies[strategy_name](
                c=params.get('c', 2.0),
                n_actions=n_actions
            )
        elif strategy_name == 'softmax':
            return strategies[strategy_name](
                temperature=params.get('temperature', 1.0),
                temp_min=params.get('temp_min', 0.01),
                temp_decay=params.get('temp_decay', 0.995)
            )


    def discretize_state(self, observation: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Discretize continuous observation into discrete state.

        Args:
            observation: [CO2, fan_speed, light_level, occupancy]

        Returns:
            Discrete state tuple
        """
        co2, fan_speed, light, occupancy = observation

        # Discretize CO2 (406-5000 ppm range)
        co2_discrete = int(np.clip((co2 - 406) / (5000 - 406) * self.co2_bins,
                                   0, self.co2_bins - 1))

        # Discretize fan speed (0-1 range)
        fan_discrete = int(np.clip(fan_speed * self.fan_bins, 0, self.fan_bins - 1))

        # Discretize light (binary: 0 or 1)
        light_discrete = 1 if light > 0.5 else 0

        # Discretize occupancy
        occupancy_discrete = int(np.clip(occupancy, 0, self.occupancy_bins - 1))

        return (co2_discrete, fan_discrete, light_discrete, occupancy_discrete)

    def select_action(self, state: Tuple, training: bool = True) -> int:
        """Select action using the exploration strategy."""
        q_values = self.q_table[state]

        if training:
            action = self.exploration_strategy.select_action(q_values)
        else:
            # During evaluation, always select best action
            action = int(np.argmax(q_values))

        # Record action statistics
        self.action_counts[action] += 1
        self.current_episode_actions.append(action)

        return action

    def update_q_value(self, state: Tuple, action: int, reward: float,
                       next_state: Tuple, done: bool) -> float:
        """
        Update Q-value using Q-learning rule.

        Returns:
            TD error (for prioritized replay)
        """
        current_q = self.q_table[state][action]

        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * max_next_q

        td_error = target_q - current_q
        self.q_table[state][action] = current_q + self.alpha * td_error

        return td_error

    def update_model(self, state: Tuple, action: int, reward: float, next_state: Tuple):
        """Update environment model."""
        self.model[(state, action)] = (reward, next_state)
        self.visited_states.add(state)

    def planning(self):
        """Perform Dyna-Q planning using learned model."""
        if len(self.model) == 0:
            return

        for _ in range(self.planning_steps):
            state_action_pairs = list(self.model.keys())
            state, action = state_action_pairs[np.random.randint(len(state_action_pairs))]

            reward, next_state = self.model[(state, action)]
            self.update_q_value(state, action, reward, next_state, done=False)

    def experience_replay(self):
        """
        Perform experience replay updates.
        Sample from replay buffer and update Q-values.
        """
        if not self.use_replay or self.replay_buffer is None:
            return

        if not self.replay_buffer.is_ready(self.min_replay_size):
            return

        # Sample batch from replay buffer
        if isinstance(self.replay_buffer, PrioritizedExperienceReplay):
            batch, weights, indices = self.replay_buffer.sample(self.replay_batch_size)
            td_errors = []

            for i, (state, action, reward, next_state, done) in enumerate(batch):
                td_error = self.update_q_value(state, action, reward, next_state, done)
                td_errors.append(td_error)

            # Update priorities
            self.replay_buffer.update_priorities(indices, np.array(td_errors))

        else:
            batch = self.replay_buffer.sample(self.replay_batch_size)

            for state, action, reward, next_state, done in batch:
                self.update_q_value(state, action, reward, next_state, done)

        self.replay_updates += 1

    def train_episode(self, env, render: bool = False) -> Tuple[float, int]:
        """Train agent for one episode."""
        observation, _ = env.reset()
        state = self.discretize_state(observation)

        total_reward = 0.0
        steps = 0
        done = False

        # Reset current episode action tracking
        self.current_episode_actions = []

        while not done:
            action = self.select_action(state, training=True)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = self.discretize_state(next_observation)

            # Q-learning update
            td_error = self.update_q_value(state, action, reward, next_state, done)

            # Store experience in replay buffer
            if self.use_replay and self.replay_buffer is not None:
                if isinstance(self.replay_buffer, PrioritizedExperienceReplay):
                    self.replay_buffer.add(state, action, reward, next_state, done, td_error)
                else:
                    self.replay_buffer.add(state, action, reward, next_state, done)

            # Model learning (Dyna-Q)
            self.update_model(state, action, reward, next_state)

            # Planning (Dyna-Q)
            self.planning()

            # Experience Replay
            self.step_counter += 1
            if self.step_counter % self.replay_frequency == 0:
                self.experience_replay()

            state = next_state
            total_reward += reward
            steps += 1

            if render:
                env.render()

        # Store episode action counts
        episode_action_count = np.zeros(self.n_actions)
        for action in self.current_episode_actions:
            episode_action_count[action] += 1
        self.episode_action_counts.append(episode_action_count)

        return total_reward, steps

    def evaluate_episode(self, env, render: bool = False) -> Tuple[float, int, dict]:
        """Evaluate agent without exploration."""
        observation, _ = env.reset()
        state = self.discretize_state(observation)

        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action = self.select_action(state, training=False)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_state = self.discretize_state(next_observation)

            state = next_state
            total_reward += reward
            steps += 1

            if render:
                env.render()

        return total_reward, steps, info

    def update_exploration(self, episode: int):
        """Update exploration strategy parameters."""
        self.exploration_strategy.update(episode)

    def get_statistics(self) -> dict:
        """Get comprehensive agent statistics."""
        stats = {
            'action_stats': self.get_action_statistics(),
            'q_table_size': len(self.q_table),
            'model_size': len(self.model),
            'visited_states': len(self.visited_states),
            'replay_updates': self.replay_updates,
            'total_steps': self.step_counter
        }

        if self.replay_buffer is not None:
            stats['replay_buffer'] = self.replay_buffer.get_statistics()

        return stats

    def get_action_statistics(self) -> dict:
        """Get action selection statistics."""
        total_actions = np.sum(self.action_counts)

        stats = {
            'action_counts': self.action_counts.copy(),
            'action_percentages': (self.action_counts / total_actions * 100) if total_actions > 0 else np.zeros(self.n_actions),
            'total_actions': int(total_actions),
            'most_selected_action': int(np.argmax(self.action_counts)),
            'least_selected_action': int(np.argmin(self.action_counts))
        }

        return stats

    def print_statistics(self):
        """Print comprehensive statistics."""
        stats = self.get_statistics()

        print("\n" + "="*60)
        print("Agent Statistics")
        print("="*60)
        print(f"Q-table size: {stats['q_table_size']}")
        print(f"Model size: {stats['model_size']}")
        print(f"Visited states: {stats['visited_states']}")
        print(f"Total steps: {stats['total_steps']}")
        print(f"Replay updates: {stats['replay_updates']}")

        if 'replay_buffer' in stats:
            rb = stats['replay_buffer']
            print(f"\nReplay Buffer:")
            print(f"  Size: {rb['current_size']}/{rb['capacity']}")
            print(f"  Utilization: {rb['utilization']:.1f}%")
            print(f"  Total added: {rb['total_added']}")

        print("\nAction Distribution:")
        action_stats = stats['action_stats']
        for action in range(self.n_actions):
            count = action_stats['action_counts'][action]
            percentage = action_stats['action_percentages'][action]
            bar = "█" * int(percentage / 2)
            print(f"  Action {action}: {int(count):6d} ({percentage:5.2f}%) {bar}")
        print("="*60 + "\n")

    def save(self, filepath: str):
        """Save agent."""
        data = {
            'q_table': dict(self.q_table),
            'model': self.model,
            'exploration_strategy_name': self.exploration_strategy.name,
            'exploration_params': self.exploration_strategy.get_params(),
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps,
            'action_counts': self.action_counts,
            'episode_action_counts': self.episode_action_counts,
            'replay_updates': self.replay_updates,
            'step_counter': self.step_counter
        }

        # Save replay buffer if it exists
        if self.replay_buffer is not None:
            data['replay_buffer'] = list(self.replay_buffer.buffer)

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Agent saved to {filepath}")

    def load(self, filepath: str):
        """Load agent."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.q_table = defaultdict(lambda: np.zeros(self.n_actions), data['q_table'])
        self.model = data['model']
        self.episode_rewards = data['episode_rewards']
        self.episode_steps = data['episode_steps']
        self.action_counts = data.get('action_counts', np.zeros(self.n_actions))
        self.episode_action_counts = data.get('episode_action_counts', [])
        self.replay_updates = data.get('replay_updates', 0)
        self.step_counter = data.get('step_counter', 0)

        # Load replay buffer if it exists
        if 'replay_buffer' in data and self.replay_buffer is not None:
            for experience in data['replay_buffer']:
                self.replay_buffer.add(*experience)

        print(f"Agent loaded from {filepath}")


# Experience Replay Buffer Classes (inline for completeness)

class ExperienceReplayBuffer:
    """Simple experience replay buffer."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.total_added = 0

    def add(self, state: Tuple, action: int, reward: float,
            next_state: Tuple, done: bool):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        self.total_added += 1

    def sample(self, batch_size: int) -> List[Tuple]:
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        return batch

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, min_size: int = 1) -> bool:
        return len(self.buffer) >= min_size

    def get_statistics(self) -> dict:
        return {
            'current_size': len(self.buffer),
            'capacity': self.capacity,
            'total_added': self.total_added,
            'utilization': len(self.buffer) / self.capacity * 100
        }


class PrioritizedExperienceReplay(ExperienceReplayBuffer):
    """Prioritized experience replay buffer."""

    def __init__(self, capacity: int = 10000, alpha: float = 0.6,
                 beta: float = 0.4, beta_increment: float = 0.001):
        super().__init__(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0

    def add(self, state: Tuple, action: int, reward: float,
            next_state: Tuple, done: bool, td_error: float = None):
        super().add(state, action, reward, next_state, done)
        priority = self.max_priority if td_error is None else abs(td_error) + 1e-6
        self.priorities.append(priority)

    def sample(self, batch_size: int) -> Tuple[List[Tuple], np.ndarray, List[int]]:
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        priorities = np.array(self.priorities, dtype=np.float64)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size,
                                  p=probabilities, replace=False)
        batch = [self.buffer[idx] for idx in indices]

        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        return batch, weights, indices.tolist()

    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-6
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

# ## Train
# 


# Create environment
env = VentilationEnvironment(max_steps=96, discrete_actions_count=5)

# Create agent with chosen exploration strategy
DynaQAgent_train = DynaQAgent(
    n_actions=5,
    learning_rate=0.1,
    discount_factor=0.99,
    planning_steps=5,
    state_discretization=(100, 10, 2, 5),
    exploration_strategy="softmax",  # Choose: random, epsilon_greedy, ucb, softmax
    temperature= 1.0,
    temp_min= 0.01,
    temp_decay= 0.999,
    use_replay=True,
    replay_buffer_size=10000,
    replay_batch_size=32,
    replay_frequency=1,  # Replay every step
    min_replay_size=100  # Start replay after 100 experiences

)

print(f"\nTraining with {DynaQAgent_train.exploration_strategy.name}")
print("=" * 60)

n_episodes = 1000
eval_interval = 50

for episode in range(1, n_episodes + 1):
    reward, steps = DynaQAgent_train.train_episode(env)
    DynaQAgent_train.episode_rewards.append(reward)
    DynaQAgent_train.episode_steps.append(steps)

    DynaQAgent_train.update_exploration(episode)

    if episode % eval_interval == 0:
        eval_reward, eval_steps, info = DynaQAgent_train.evaluate_episode(env)

        avg_train_reward = np.mean(DynaQAgent_train.episode_rewards[-eval_interval:])

        print(f"\nEpisode {episode}/{n_episodes}")
        print(f"  Strategy: {DynaQAgent_train.exploration_strategy.name}")
        print(f"  Params: {DynaQAgent_train.exploration_strategy.get_params()}")
        print(f"  Avg Train Reward: {avg_train_reward:.2f}")
        print(f"  Eval Reward: {eval_reward:.2f}")
        print(f"  IAQ Compliance: {info['iaq_compliance_rate']*100:.1f}%")
        print(f"  Energy Used: {info['energy_consumed']:.2f} Wh")
        print(f"  Q-table size: {len(DynaQAgent_train.q_table)}")

print("\n" + "=" * 60)
print("Training Complete!")

# Print action statistics
DynaQAgent_train.get_action_statistics()

# Save agent
#DynaQAgent_train.save(f"dynaq_{DynaQAgent_train.exploration_strategy.name.lower().replace(' ', '_')}_agent.pkl")