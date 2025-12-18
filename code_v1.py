# Generated from: asssignment v1.ipynb
# Converted at: 2025-12-18T07:05:48.495Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Environment


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
        k_iaq: float = 1.0,
        k_energy: float = 0.0015,
        k_switch: float = 1.0,
        max_fan_power: float = 325.0,
        max_fan_air_volume: float = 800.0,
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

        self.total_reward = 0.0
        self.iaq_violations = 0
        self.energy_consumed = 0.0
        self.fan_switch_count = 0


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
        self.fan_switch_count = 0
        self.previous_occupancy = 0

        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action_idx: int):
        fan_speed = self.discrete_actions[action_idx]
        self.fan_speed = fan_speed

        self._update_occupancy_and_light()
        self._update_co2_level(fan_speed)
        reward = float(self._calculate_reward(fan_speed))
        self.total_reward += reward

        if self.co2_level > self.co2_setpoint:
            self.iaq_violations += 1

        self.energy_consumed += (fan_speed * self.max_fan_power) * (self.dt / 60.0)
        if fan_speed != self.previous_fan_speed:
          self.fan_switch_count += 1
        self.previous_fan_speed = fan_speed
        self.previous_occupancy = self.occupancy

        self.current_step += 1
        terminated = False
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
            'is_meeting_active': self.is_meeting_active,
            'fan_switch_count': self.fan_switch_count
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
      co2_generation_L_per_min = self.occupancy * 0.3125  # L/min

    # Convert to ppm/min
      generation_ppm_per_min = (co2_generation_L_per_min / (self.room_volume * 1000)) * 1e6
      ventilation_rate_m3_per_min = fan_speed * self.max_ventilation_rate_m3_per_min

      if ventilation_rate_m3_per_min > 0 and self.co2_level > self.co2_outdoor:
          air_change_rate = ventilation_rate_m3_per_min / self.room_volume  # 1/min ✓
          removal_ppm_per_min = air_change_rate * (self.co2_level - self.co2_outdoor)
          # Units: (1/min) * (ppm) = ppm/min ✓
      else:
          removal_ppm_per_min = 0

      # NET CHANGE (both terms in ppm/min now!)
      net_change_ppm_per_min = generation_ppm_per_min - removal_ppm_per_min
      # Units: ppm/min - ppm/min = ppm/min ✓

      # UPDATE
      self.co2_level += net_change_ppm_per_min * self.dt
      # Units: ppm + (ppm/min * min) = ppm ✓
      self.co2_level = np.clip(self.co2_level, self.co2_outdoor, 5000.0)

    def _calculate_reward(self, fan_speed: float):
        # 1. Air-quality satisfaction with adjustable weight

        r_iaq = self.k_iaq * (1 - min(abs(self.co2_level - self.co2_outdoor) / 1000.0, 1.0))

        # 2. Energy penalty (can remove /100 scaling if desired)
        r_energy = - self.k_energy * fan_speed * self.max_fan_power

        # 3. Switching penalty: penalize any fan speed change
        fan_change = abs(fan_speed - self.previous_fan_speed)
        r_switch = - self.k_switch * fan_change

        if hasattr(self, 'previous_occupancy'):
          occupancy_change = abs(self.occupancy - self.previous_occupancy)

          # If occupancy changes but fan speed changes only slightly, a penalty will be imposed.
          if occupancy_change > 0 and fan_change < 0.1:
            # The magnitude of the punishment is directly proportional to the change in occupancy.
            r_occupancy_response = -0.3 * (occupancy_change / self.max_occupancy)
          else:
              r_occupancy_response = 0
        else:
          r_occupancy_response = 0

        total_reward = r_iaq + r_energy + r_switch  + r_occupancy_response
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

# # sarsa


import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from collections import defaultdict
import pickle
from datetime import datetime

class SarsaAgent:
    """
    SARSA agent for the VentilationEnvironment with configurable exploration strategies.
    修正了参数透传 (strategy_params) 和探索策略初始化逻辑。
    """
    def __init__(
        self,
        env: gym.Env,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        state_discretization: tuple = (100, 5, 2, 25),
        exploration_strategy: str = "epsilon_greedy",
        **strategy_params # 接收 temperature, epsilon, c 等参数
    ):
        self.env = env
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = discount_factor

        # 提取策略参数，设置默认值
        self.epsilon = strategy_params.get('epsilon', 1.0)
        self.epsilon_decay = strategy_params.get('epsilon_decay', 0.995)
        self.min_epsilon = strategy_params.get('epsilon_min', 0.01)

        # 保存所有参数用于工厂函数
        self.strategy_params = strategy_params

        # Initialize Q-table
        # Changed to use a picklable method as default_factory
        self.q_table = defaultdict(self._create_default_q_values)

        # Episode tracking
        self.episode_rewards = []
        self.episode_steps = []
        self.action_counts = defaultdict(int)

        # Define discretization bins (保留你原有的逻辑)
        num_co2_bins, num_fan_speed_bins, num_light_bins, num_occupancy_bins = state_discretization

        self.co2_bins = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num_co2_bins + 1)
        self.obs_fan_speed_bins = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num_fan_speed_bins + 1)

        occupancy_high = int(env.observation_space.high[3])
        self.occupancy_bins = np.linspace(0, occupancy_high, num_occupancy_bins + 1)

        # 修正后的工厂函数调用，透传 strategy_params
        self.exploration_strategy = self._create_exploration_strategy(
            exploration_strategy, **strategy_params
        )

    def _create_default_q_values(self):
        """Helper method to create default Q-values for a new state."""
        return {action_idx: 0.0 for action_idx in range(self.n_actions)}

    def _create_exploration_strategy(self, strategy_name: str, **params):
        """修正后的工厂方法：直接匹配外部 create_exploration_strategy 的逻辑"""
        # 注意：这里调用的是你代码库中定义的通用类（RandomExploration, Softmax 等）
        if strategy_name == "random":
            return RandomExploration(n_actions=self.n_actions)
        elif strategy_name == "epsilon_greedy":
            return EpsilonGreedy(
                epsilon=params.get('epsilon', 1.0),
                epsilon_min=params.get('epsilon_min', 0.01),
                epsilon_decay=params.get('epsilon_decay', 0.995)
            )
        elif strategy_name == "ucb":
            return UCB(c=params.get('c', 2.0), n_actions=self.n_actions)
        elif strategy_name == "softmax":
            return Softmax(
                temperature=params.get('temperature', 1.0), # 接收 temperature
                temp_min=params.get('temp_min', 0.01),
                temp_decay=params.get('temp_decay', 0.995)
            )
        else:
            raise ValueError(f"Unknown exploration strategy: {strategy_name}")

    def discretize_state(self, observation: np.ndarray) -> tuple:
        """保持原有的 np.digitize 离散化逻辑"""
        co2_level, fan_speed_obs, light_level, occupancy = observation

        co2_bin = np.digitize(co2_level, self.co2_bins) - 1
        co2_bin = np.clip(co2_bin, 0, len(self.co2_bins) - 2)

        fan_speed_obs_bin = np.digitize(fan_speed_obs, self.obs_fan_speed_bins) - 1
        fan_speed_obs_bin = np.clip(fan_speed_obs_bin, 0, len(self.obs_fan_speed_bins) - 2)

        light_level_discrete = int(round(light_level))

        occupancy_bin = np.digitize(occupancy, self.occupancy_bins) - 1
        occupancy_bin = np.clip(occupancy_bin, 0, len(self.occupancy_bins) - 2)

        return (int(co2_bin), int(fan_speed_obs_bin), int(light_level_discrete), int(occupancy_bin))

    def select_action(self, state: tuple, training: bool = True) -> int:
        """使用 Q 字典进行动作选择"""
        q_dict = self.q_table[state]

        if training:
            # 兼容字典格式的探索策略调用
            action = self.exploration_strategy.select_action(q_dict)
        else:
            # 评估时使用贪婪策略
            max_q = max(q_dict.values())
            max_actions = [a for a, q in q_dict.items() if q == max_q]
            action = np.random.choice(max_actions)

        if training:
            self.action_counts[action] += 1
        return action

    def learn(self, state: tuple, action: int, reward: float, next_state: tuple, next_action: int, done: bool):
        """标准的 SARSA 更新规则"""
        current_q = self.q_table[state][action]
        next_q = 0.0 if done else self.q_table[next_state][next_action]

        # SARSA TD 更新
        td_target = reward + self.gamma * next_q
        self.q_table[state][action] += self.alpha * (td_target - current_q)

        # 注意：这里不再调用 self.exploration_strategy.update
        # 因为 update 通常是 Episode 级别的，我们在外层循环调用。

    def train_episode(self, env):
        """SARSA 训练循环：S -> A -> R -> S' -> A'"""
        observation, info = env.reset()
        state = self.discretize_state(observation)
        action = self.select_action(state)

        total_reward = 0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = self.discretize_state(next_observation)

            done = terminated or truncated
            # SARSA 关键：在更新当前 Q 之前先选出下一个动作
            next_action = self.select_action(next_state) if not done else None

            self.learn(state, action, reward, next_state, next_action, done)

            state = next_state
            action = next_action
            total_reward += reward
            steps += 1

        return total_reward, steps

    def evaluate_episode(self, env):
        observation, info = env.reset()
        state = self.discretize_state(observation)
        episode_reward = 0
        episode_steps = 0
        done = False
        while not done:
            action = self.select_action(state, training=False)
            obs, reward, term, trun, info = env.step(action)
            state = self.discretize_state(obs)
            episode_reward += reward
            episode_steps += 1
            done = term or trun
        return episode_reward, episode_steps, info

    def update_exploration(self, episode: int):
        """统一调用探索策略的更新逻辑"""
        self.exploration_strategy.update(episode)

    def print_action_statistics(self):
        print("\n" + "=" * 60)
        print("Action Selection Statistics")
        print("=" * 60)
        total_actions = sum(self.action_counts.values())
        if total_actions == 0: return
        for action_idx in range(self.n_actions):
            count = self.action_counts[action_idx]
            percentage = (count / total_actions * 100)
            bar = "█" * int(percentage / 2)
            print(f"  Action {action_idx}: {count:6d} ({percentage:5.2f}%) {bar}")
        print("=" * 60)

    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str):
        with open(filename, 'rb') as f:
            return pickle.load(f)


# ## Train
# 


# env = VentilationEnvironment(max_steps=96, discrete_actions_count=5)

# # Create agent with chosen exploration strategy
# SarsaAgent_train = SarsaAgent(
#     env=env,
#     n_actions=5,
#     learning_rate=0.1,
#     discount_factor=0.999,
#     state_discretization=(50, 5, 2, 5),
#     exploration_strategy="epsilon_greedy",  # Choose: random, epsilon_greedy, ucb, softmax
#     epsilon=1.0,
#     epsilon_decay=0.999
# )

# print(f"\nTraining with {SarsaAgent_train.exploration_strategy.name}")
# print("=" * 60)

# n_episodes = 3000
# eval_interval = 50

# for episode in range(1, n_episodes + 1):
#     reward, steps = SarsaAgent_train.train_episode(env)
#     SarsaAgent_train.episode_rewards.append(reward)
#     SarsaAgent_train.episode_steps.append(steps)

#     SarsaAgent_train.update_exploration(episode)

#     if episode % eval_interval == 0:
#         eval_reward, eval_steps, info = SarsaAgent_train.evaluate_episode(env)

#         avg_train_reward = np.mean(SarsaAgent_train.episode_rewards[-eval_interval:])

#         print(f"\nEpisode {episode}/{n_episodes}")
#         print(f"  Strategy: {SarsaAgent_train.exploration_strategy.name}")
#         print(f"  Params: {SarsaAgent_train.exploration_strategy.get_params()}")
#         print(f"  Avg Train Reward: {avg_train_reward:.2f}")
#         print(f"  Eval Reward: {eval_reward:.2f}")
#         print(f"  IAQ Compliance: {info.get('iaq_compliance_rate', 0)*100:.1f}%")
#         print(f"  Energy Used: {info.get('energy_consumed', 0):.2f} Wh")
#         print(f"  Q-table size: {len(SarsaAgent_train.q_table)}")

# print("\n" + "=" * 60)
# print("Training Complete!")

# # Print action statistics

# # Save agent
# SarsaAgent_train.save(f"sarsa_{SarsaAgent_train.exploration_strategy.name.lower().replace(' ', '_')}_agent.pkl")