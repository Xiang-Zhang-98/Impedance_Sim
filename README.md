# Impedance_Sim

## Installation
### Step 1: install Mujoco 200 and place the mjkey.txt under ./mujoco
### Step 2: add following lines to bashrc:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/YOUR_USERNAME/.mujoco/mujoco200/bin/

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

export MUJOCO_LICENSE_PATH=/home/YOUR_USERNAME/.mujoco/mjkey.txt
```
### Step 3: install environment

```
conda create --name impedance_sim python=3.7 pip
conda activate impedance_sim
cd rlkit/
pip install -e .
cd ..
cd impedance_envs/
pip install -e .
cd ..
pip install -r requirements.txt
pip install pip install torch==1.13.1+cu117 --index-url https://download.pytorch.org/whl/cu117
```

### Step 4: try to rollout

```
python Roll_out/Roll_out_peg_in_hole.py 