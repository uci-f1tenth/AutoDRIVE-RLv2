from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple, Union

import gymnasium as gym
import numpy as np
import requests
from gym_unity.envs import UnityToGymWrapper
from gymnasium import spaces
from gymnasium.core import RenderFrame
from mlagents_envs.environment import UnityEnvironment


class AutoDRIVEWrapper(gym.Wrapper):
    def __init__(self) -> None:
        self.url = "http://127.0.0.1:9000"

        if os.environ.get("UNITY_EDITOR", "").lower() in ["1", "true", "t"]:
            unity_env = UnityEnvironment()
        else:
            match sys.platform:
                case "darwin":
                    binary_path = "autodrive_macos_build/autodrive.app"
                case "win32":
                    binary_path = r"autodrive_windows_build/autodrive/AutoDRIVE Simulator.exe"
                case _:
                    binary_path = "autodrive_linux_build/autodrive.x86_64"
            unity_env = UnityEnvironment(binary_path, no_graphics=True)

        self.env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )  # (0, 1) for throttle and (-1, 1) for steering
        self.reward_range = (-np.inf, np.inf)
        self._render_mode: str = "rgb_array"
        self._metadata = {"render_fps": 60}

    def __del__(self) -> None:
        self.close()

    @property
    def render_mode(self) -> str:
        return self._render_mode

    def _convert_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        return {"state": obs[0]}

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)  # type: ignore
        try:
            requests.post(self.url, json={"command": "publish_pose", "data": obs[0].tolist()})
        except Exception as e:
            print(f"Failed to send publish command: {e}")
        return self._convert_obs(obs), reward, done, False, info  # type: ignore

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs = self.env.reset()
        try:
            requests.post(self.url, json={"command": "reset"})
        except Exception as e:
            print(f"Failed to send reset command: {e}")
        return self._convert_obs(obs), {}  # type: ignore

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        return self.env.render()  # type: ignore

    def close(self):
        try:
            self.env.close()
        except Exception as e:
            print(f"Failed to close environment: {e}")
