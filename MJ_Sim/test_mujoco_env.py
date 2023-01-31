import mujoco_py
import numpy as np
import os

def test_viewer(path):
    model = mujoco_py.load_model_from_path(path)
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)
    for _ in range(10000):
        sim.data.ctrl[0] = 1
        sim.data.ctrl[1] = 0
        sim.data.ctrl[2] = 0
        sim.data.ctrl[3] = 0
        sim.data.ctrl[4] = 0
        sim.data.ctrl[5] = 0
        # sim.data.ctrl[6] = 0.1
        # sim.data.ctrl[7] = 0.1

        sim.step()
        viewer.render()

if __name__ == '__main__':
    test_viewer('/home/zx/UCBerkeley/Research/Impedance_sim/MJ_Sim/source/LRMate_200iD_vel_crl.xml')
    # test_viewer('pushing.xml')