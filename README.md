## AutoDRIVE
Download and unzip the correct platform binary from the [releases page](https://github.com/uci-f1tenth/AutoDRIVE-RLv2/releases), and put it in this directory, such that the binary is under `autodrive_macos_build/autodrive.app`, `autodrive_windows_build/autodrive/AutoDRIVE Simulator.exe`, or `autodrive_linux_build/autodrive.x86_64`.

## SheepRL
In order to run sheeprl, first install `uv` if you haven't already (note this is our package manager / virtual environment creator, so please don't use `python -m venv` or `pip` in tandem. You are free to use those instead if you prefer, but they won't be officially supported):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Now run the RL environment:
```bash
uv run sheeprl/sheeprl.py exp=dreamer_v3_autodrive env=autodrive
```