from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple, Union

from mlagents_envs.environment import UnityEnvironment  # type: ignore
from gym_unity.envs import UnityToGymWrapper  # type: ignore
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import RenderFrame


class AutoDRIVEWrapper(gym.Wrapper):
    def __init__(self) -> None:
        if sys.platform == "darwin":
            binary_path = "autodrive_macos_build/autodrive.app"
        elif sys.platform == "win32":
            binary_path = r"autodrive_windows_build/autodrive/AutoDRIVE Simulator.exe"
        else:
            binary_path = "autodrive_linux_build/autodrive.x86_64"

        unity_env = UnityEnvironment(binary_path, no_graphics=True)
        self.env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)

        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(55,), dtype=np.float32),
            }
        )
        self.action_space = spaces.MultiDiscrete([3, 3])
        self.reward_range = (-np.inf, np.inf)
        self._render_mode: str = "rgb_array"
        self._metadata = {"render_fps": 60}

    @property
    def render_mode(self) -> str:
        return self._render_mode

    def _convert_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        return {"state": obs[0]}

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        return self._convert_obs(obs), reward, done, False, info

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        return self._convert_obs(obs), {}

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        return self.env.render()

    def close(self):
        self.env.close()
