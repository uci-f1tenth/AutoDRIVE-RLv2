## AutoDRIVE
Download and unzip the correct platform binary from the [releases page](https://github.com/uci-f1tenth/AutoDRIVE-RLv2/releases), and put it in this directory, such that the binary is under `autodrive_macos_build/autodrive.app`, `autodrive_windows_build/autodrive/AutoDRIVE Simulator.exe`, or `autodrive_linux_build/autodrive.x86_64`.

## SLAM
to run with slam:
```bash
source /opt/ros/humble/setup.bash
ros2 launch slam_toolbox online_async_launch.py slam_params_file:=autodrive_online_async.yaml
```
to start slam. To start the sim in a devcontainer friendly way, run:
```bash
source /opt/ros/humble/setup.bash
HYDRA_FULL_ERROR=1 UNITY_EDITOR=1 python3 sheeprl/sheeprl.py exp=dreamer_v3_autodrive env=autodrive fabric.accelerator=auto
```
or
```bash
source /opt/ros/humble/setup.bash
HYDRA_FULL_ERROR=1 UNITY_EDITOR=1 python3 sheeprl/sheeprl.py exp=ppo env=autodrive fabric.accelerator=auto
```
for testing.
Then to launch foxglove bridge:
```bash
source /opt/ros/humble/setup.bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```