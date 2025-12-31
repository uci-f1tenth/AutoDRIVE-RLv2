from __future__ import annotations

import os
import sys
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
        if sys.platform == "darwin":
            binary_path = "autodrive_macos_build/autodrive.app"
        elif sys.platform == "win32":
            binary_path = r"autodrive_windows_build/autodrive/AutoDRIVE Simulator.exe"
        else:
            binary_path = "autodrive_linux_build/autodrive.x86_64"

        if os.environ.get("UNITY_EDITOR", "").lower() in ["1", "true", "t"]:
            unity_env = UnityEnvironment()
        else:
            unity_env = UnityEnvironment(binary_path, no_graphics=True)

        self.env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)

        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(53,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )  # (0, 1) for throttle and (-1, 1) for steering
        self.reward_range = (-np.inf, np.inf)
        self._render_mode: str = "rgb_array"
        self._metadata = {"render_fps": 60}

    @property
    def render_mode(self) -> str:
        return self._render_mode

    def _convert_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        return {"state": obs[0]}

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)  # type: ignore
        try:
            requests.post(
                "http://127.0.0.1:9000",
                json={
                    "command": "publish",
                    "x": float(obs[0][-3]),
                    "y": float(obs[0][-2]),
                    "yaw": float(obs[0][-1]),
                    "ranges": obs[0][:-3].tolist(),
                },
                timeout=1.0,
            )
        except Exception:
            pass
        return self._convert_obs(obs), reward, done, False, info  # type: ignore

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs = self.env.reset()
        try:
            requests.post("http://127.0.0.1:9000", json={"command": "reset"}, timeout=1.0)
        except Exception:
            pass
        return self._convert_obs(obs), {}  # type: ignore

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        return self.env.render()  # type: ignore

    def close(self):
        try:
            requests.post("http://127.0.0.1:9000", json={"command": "shutdown"}, timeout=1.0)
        except Exception:
            pass
        try:
            self.env.close()
        except Exception:
            pass
