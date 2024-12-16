# Reinforcement Learning Based Traffic Signal Control
Group Member: Haiyu Wei, Weiyi Guo, Yisong Shen
## Environment Requirement
For this project, we use SUMO Simulator as our traffic environment, and Pytorch as our training framework.
<!-- start install -->
### Install SUMO latest version:

```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```
Set SUMO_HOME variable (default sumo installation path is /usr/share/sumo)
```bash
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
```

### Required Python Packages
Below is the main required packages, the full list of required packages please see the requirements.yaml
```bash
filelock==3.13.1
fsspec==2024.2.0
gym==0.26.2
gym-notices==0.0.8
jinja2==3.1.3
markupsafe==2.1.5
networkx==3.2.1
pettingzoo==1.24.3
sympy==1.13.1
torch==2.5.1+cu121
torchaudio==2.5.1+cu121
torchvision==0.20.1+cu121
```

<!-- end install -->

## How to run
The main training file is `dqn_agent.py`. If your environment is all set, then it will automatically open the SUMO simulator, then begin the training process.
Before you run, please make sure the `csu.netxml`, `csu.rou.xml`, `demand.rou.xml`, `csu.sumocfg` and `dqn_agent.py` is in the same folder.
```bash
python dqn_agent.py
```


