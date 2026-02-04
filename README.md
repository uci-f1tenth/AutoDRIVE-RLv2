# Local setup (for rapid development)
## Install Unity 2022.3.62f3
Install [unity hub](https://unity.com/download), and install the [unity 2022.3.62f3 editor](https://unity.com/releases/editor/whats-new/2022.3.62f3).
## Clone AutoDRIVE
Clone the AutoDRIVE Simulator:
```bash
git clone --single-branch --branch AutoDRIVE-Simulator https://github.com/uci-f1tenth/AutoDRIVE-v2
```
## Open AutoDRIVE project
[Add the AutoDRIVE from you disk in Unity hub.](https://docs.unity3d.com/hub/manual/AddProject.html). Your Unity Hub should look like this:
![Unity Hub](./images/unity%20hub.png)

Open the AutoDRIVE project in Unity, and open the `F1TENTH - MARL` Scene. Your unity should look like this:
![MARL Scene](./images/MARL%20scene.png)
## Clone AutoDRIVE-RLV2
Clone this project with
```
git clone https://github.com/uci-f1tenth/AutoDRIVE-RLv2
```
## Open this project in vscode
```
cd https://github.com/uci-f1tenth/AutoDRIVE-RLv2
code .
```
Now your Vscode should look like this:
![Vscode](./images/vscode.png)
## Install the Dev containers Extension
Install the [Devcontainers Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
## Install docker engine
I recommend [Orbstack](https://orbstack.dev/download) on macos for great performance. Docker desktop is also a solid tried-and-true choice on [linux](https://docs.docker.com/desktop/setup/install/linux/) and [windows](https://docs.docker.com/desktop/setup/install/windows-install/).
## Open in Dev Container
Open the command palette (Command+Shift+P), and press "Dev Containers: Rebuild and Reopen in Container":
![Vscode](./images/rebuild.png)
## Run the following commands in seperate terminals in vscode
### PPO (Faster) or DreamerV3 (more accurate)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
UNITY_EDITOR=1 HYDRA_FULL_ERROR=1 uv run sheeprl/sheeprl.py exp=ppo env=autodrive fabric.accelerator=auto
```
or
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
UNITY_EDITOR=1 HYDRA_FULL_ERROR=1 uv run sheeprl/sheeprl.py exp=dreamer_v3_autodrive env=autodrive fabric.accelerator=auto
```
### Foxglove
```bash
source /opt/ros/$ROS_DISTRO/setup.bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```
### SLAM
```bash
source /opt/ros/$ROS_DISTRO/setup.bash
ros2 launch slam_toolbox online_async_launch.py slam_params_file:=autodrive_online_async.yaml
```
### ROS_BRIDGE
```bash
source /opt/ros/$ROS_DISTRO/setup.bash
python3 ros_bridge.py
```
## Press play in unity
Hit the play button in the unity editor
## Open Foxglove
Foxglove visualizes our lidar data and ros state. So [make an account](https://foxglove.dev) (the deatils, like your organization), don't matter. Open ws://localhost:8765. It should look like this:
![Running](./images/running.png)
# Remote setup (for deploying, currently only Linux is support)
## AutoDRIVE
Download and unzip the correct platform binary (currently only Linux is supported) from the [releases page](https://github.com/uci-f1tenth/AutoDRIVE-RLv2/releases), and put it in this directory, such that the binary is under `autodrive_macos_build/autodrive.app`, `autodrive_windows_build/autodrive/AutoDRIVE Simulator.exe`, or `autodrive_linux_build/autodrive.x86_64`.
## Th 
TBD