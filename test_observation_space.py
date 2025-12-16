import sys

import numpy as np
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment

setattr(np, "bool", bool)


if sys.platform == "darwin":
    binary_path = "autodrive_macos_build/autodrive.app"
elif sys.platform == "win32":
    binary_path = r"autodrive_windows_build/autodrive/AutoDRIVE Simulator.exe"
else:
    binary_path = "autodrive_linux_build/autodrive.x86_64"

env = UnityToGymWrapper(UnityEnvironment(), allow_multiple_obs=True)

print(env.observation_space)

print(env.step(env.action_space.sample()))

env.close()
