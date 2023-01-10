## README

This project aims at create a simulated MuJoCo C++ Fanuc robot for the usage of the MSC lab.

## Tested Environment

- `Ubuntu 18.04`
- `gcc version 6.5.0`

## Installation

- Install `gcc` on your machine according to [this](https://code.visualstudio.com/docs/languages/cpp)

- Pull the git repo

- Obtain a license on the [MuJoCo website](https://www.roboti.us/license.html), you should receive a `mjkey.txt` file

- Download the MuJoCo version 2.0 binaries for [Ubuntu](https://www.roboti.us/download/mujoco200_linux.zip)

- Unzip the downloaded `mujoco200` directory into `~/.mujoco/mujoco200`, and place your license key (the `mjkey.txt` file) at `~/.mujoco/mjkey.txt`, `~/.mujoco/mujoco200/bin/mjkey.txt`

- Add following lines to your `~/.bashrc`, replace `{USERNAME}` with your own path (`makefile` will heavily rely on these environment path)

  `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH/home/{USERNAME}/.mujoco/mujoco200/bin/`

  `export MUJOCO_PATH=$MUJOCO_PATH/home/{USERNAME}/.mujoco/mujoco200/`

  `export MUJOCO_LICENSE_PATH=/home/{USERNAME}/.mujoco/mujoco200/bin/mjkey.txt`

- To compile the code, type in `make` at the root of this repo, there should not be any error message, built binary files will be stored at `bin` folder, type `./bin/file_name` to run

- Procedures to setup a debug environment may be variant to different IDEs, if you are using VS Code, follow these steps

  - You should find a `.vscode` folder from the repo, this contains tested debug json setting `launch.json`, you could simply replace the path at line 25 to your MuJoCo path (I cannot replace this path with the environment path)
  - To debug the `cpp` file, activate the file and press `F5`, the UI may not work since it depends on the runtime
  - Feel free to ask Xinghao if you stuck at some point using `VS Code` :) 
  
- Checkout `environment.py` to see a demo of trajectory tracking and grasping

## Implemented Functions

If you have any questions, suggestions or find any bugs, please contact the author to fix them. We need heavy test to guarantee the robustness

- Robot `.xml` file (by Wu-Te and Xinghao)
- Robot model initialization and UI initialization (by Xinghao)
- Robot general control loop (by Xinghao)
- Computed-Torque controller to track joint space trajectory (by Xinghao)
- A simple cartesian space impedance controller to track a cartesian space point (by Xiang)
- Apply cartesian space force & torque on the end-effector (by Xiang)
- A bunch of Python APIs. Please check `Example.py` to learn how use them with ctypes package (by Xiang)
- FK & IK in `c++` and `python` (by Wu-Te)
- Joint space trajectory planning in `c++` and `python` (by Wu-Te)
- `environment.py` wrapped all `c++` and `python` functions to a `python class` and demonstrate how to control the robot in `python` (by Xinghao)
- Load objects (by Xinghao)

## TODO

- Velocity control for torque motors (Implemented a naive version, may need more efforts)
- Cameras
- Track object's pose
- Get contact information
- Dynamics in the `xml`file might need fine tunning
- UI design (make it more informative)

## Note

- When codding, please try not to use other dependencies, it might make the installation harder. MuJoCo C++ has many useful APIs for math, please check them before you decide to include other libraries
- Try to add clear and informative comments to your functions, it'll make the code readable
- GOOD LUCK!!!!

