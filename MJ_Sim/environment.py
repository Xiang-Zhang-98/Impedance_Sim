import ctypes
import os
import sys
import time
import numpy as np
from source.lrmate_kine_base import FK, IK
import source.trajectory_cubic as traj
import matplotlib.pyplot as plt

# To debug in VS Code, add the following line to launch.json under "configurations"
# "env": {"LD_LIBRARY_PATH": "/home/{USER_NAME}/.mujoco/mujoco200/bin/"}


class mj_sim:
    def __init__(self):
        cwd = os.getcwd()
        self.sim = ctypes.cdll.LoadLibrary(cwd + "/bin/mujocosim.so")

        # parameters which will be updated every step
        self.joint_pos = np.zeros(6)
        self.joint_vel = np.zeros(6)
        self.joint_acc = np.zeros(6)
        self.full_jacobian = np.zeros((6, 8))
        self.force_sensor_data = np.zeros(3)

        self.time_render = np.zeros(1)
        self.gripper_close = False

        # build a c type array to hold the return value
        self.joint_pos_holder = (ctypes.c_double * 8)(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )
        self.joint_vel_holder = (ctypes.c_double * 8)(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )
        self.joint_acc_holder = (ctypes.c_double * 8)(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )
        self.time_render_holder = (ctypes.c_double * 1)(0.0)
        self.verbose_holder = (ctypes.c_bool)(False)
        jac = [0.0] * 48
        self.jacobian_holder = (ctypes.c_double * len(jac))(*jac)
        self.force_sensor_data_holder = (ctypes.c_double * 3)(
            0.0, 0.0, 0.0
        )

        # initialize the simulation
        self.n_step = 0
        self.sim.wrapper_init()

        # initialize a PD gain, may need more effort on tunning
        kp = np.array([17, 17, 17, 17, 17, 17])
        kv = np.array([40, 40, 40, 40, 40, 40])
        self.set_pd_gain(kp, kv)

        # initialize path planning
        self.HZ = 125  # this is the frequency of c++ simulator, set in the xml file
        self.traj_pose, self.traj_vel, self.traj_acc = None, None, None

        for _ in range(10):
            self.step()

    def step(self):
        self.sim.wrapper_step()
        self.get_joint_states()
        self.get_sensor_data()
        print(self.force_sensor_data)
        # self.get_jacobian()

    def get_sim_time(self):
        self.sim.wrapper_get_sim_time(self.time_render_holder)
        return self.time_render_holder[0]

    def get_sensor_data(self):
        self.sim.wrapper_get_sensor_reading(self.force_sensor_data_holder)
        self.force_sensor_data = np.array(self.force_sensor_data_holder[:3])

    def get_joint_states(self):
        self.sim.wrapper_get_joint_states(
            self.joint_pos_holder, self.joint_vel_holder, self.joint_acc_holder
        )
        self.joint_pos = np.array(self.joint_pos_holder[:6])
        self.joint_vel = np.array(self.joint_vel_holder[:6])
        self.joint_acc = np.array(self.joint_acc_holder[:6])

    def get_jacobian(self):
        self.sim.wrapper_eef_full_jacobian(self.jacobian_holder)
        self.full_jacobian = np.array(self.jacobian_holder).reshape(6, 8)

    def set_gripper(self, pose=0.0):
        # gripper pose: 0 for fully open; 0.042 for fully close
        pose = 0.042 if pose > 0.042 else pose
        self.sim.wrapper_update_gripper_state((ctypes.c_double)(pose))

    def set_reference_traj(self, ref_joint, ref_vel, ref_acc):
        assert (
            ref_joint.shape == (6,) and ref_vel.shape == (6,) and ref_acc.shape == (6,)
        )
        ref_joint_mj = (ctypes.c_double * 8)(
            ref_joint[0],
            ref_joint[1],
            ref_joint[2],
            ref_joint[3],
            ref_joint[4],
            ref_joint[5],
            0.0,
            0.0,
        )
        ref_vel_mj = (ctypes.c_double * 8)(
            ref_vel[0],
            ref_vel[1],
            ref_vel[2],
            ref_vel[3],
            ref_vel[4],
            ref_vel[5],
            0.0,
            0.0,
        )
        ref_acc_mj = (ctypes.c_double * 8)(
            ref_acc[0],
            ref_acc[1],
            ref_acc[2],
            ref_acc[3],
            ref_acc[4],
            ref_acc[5],
            0.0,
            0.0,
        )
        self.sim.wrapper_update_reference_traj(ref_joint_mj, ref_vel_mj, ref_acc_mj)

    def set_pd_gain(self, kp, kv):
        kp_mj = (ctypes.c_double * 6)(kp[0], kp[1], kp[2], kp[3], kp[4], kp[5])
        kv_mj = (ctypes.c_double * 6)(kv[0], kv[1], kv[2], kv[3], kv[4], kv[5])
        self.sim.wrapper_update_pd_gain(kp_mj, kv_mj)

    def set_controller(self, controller_idx=0):
        controller_idx_mj = (ctypes.c_int)(controller_idx)
        self.sim.wrapper_update_controller_type(controller_idx_mj)

    def ik(self, pose):
        """
        pose: in m, in the world base (bottom of the robot) and in rad
        output: in rad as numpy array, if singular, return False
        """
        joint = IK(pose)
        if joint is False:
            return np.zeros(6)
        else:
            return joint

    def fk(self, joints):
        """
        input: in rad
        output: in m and rad
        """
        joints = np.rad2deg(joints)
        return FK(joints)

    def plan_traj(self, target, via=None):
        dist = abs(target - self.joint_pos / 0.7)
        T = int(np.max(dist) + 1)
        start = np.hstack((self.joint_pos, 0.0))
        target = np.hstack((target, T))
        if via is not None:
            via.reshape(-1, 6)
            n_via = via.shape[0]
            via = np.hstack((via, np.zeros((n_via, 1))))
        else:
            step_size = (target - start) / 3
            via = np.vstack((start + step_size, start + 2 * step_size))
        self.traj_pose, self.traj_vel, self.traj_acc = traj.trajectory_xyz(
            start.reshape(1, -1), target.reshape(1, -1), via, self.HZ
        )
        self.traj_pose = np.vstack((self.traj_pose, target[:6]))
        self.traj_vel = np.vstack((self.traj_vel, np.zeros(6)))
        self.traj_acc = np.vstack((self.traj_acc, np.zeros(6)))

    def go2jpose(self, target):
        self.plan_traj(target)
        # j, v, a = np.zeros(6), np.zeros(6), np.zeros(6)  # for jva plot
        i = 0
        while i < self.traj_pose.shape[0]:
            self.set_reference_traj(
                self.traj_pose[i], self.traj_vel[i], self.traj_acc[i]
            )
            i += 1
            self.step()
            # j = np.vstack((j, self.joint_pos))  # for jva plot
            # v = np.vstack((v, self.joint_vel))  # for jva plot
            # a = np.vstack((a, self.joint_acc))  # for jva plot
        self.set_reference_traj(target, np.zeros(6), np.zeros(6))

        # # for jva plot
        # for _ in range(400):
        #     self.step()
        #     j = np.vstack((j, self.joint_pos))
        #     v = np.vstack((v, self.joint_vel))
        #     a = np.vstack((a, self.joint_acc))
        # self.plot_jva(j, v, a, 0)

    def go2cpose(self, target):
        """
        target: in m, in the world base (bottom of the robot) and in rad
        """
        jtarget = self.ik(target)
        self.go2jpose(jtarget)
    

    def set_joint_velocity(self, target_vel):
        T = 1 / self.HZ
        target_pos = self.joint_pos + T * (self.joint_vel + target_vel) / 2
        target_acc = (target_vel - self.joint_vel) / T
        self.set_reference_traj(target_pos, target_vel, target_acc)

    def plot_jva(self, j, v, a, axis=None):
        fig = plt.figure()
        if axis is None:
            all_axis = [0, 1, 2, 3, 4, 5]
            for axis in all_axis:
                ax = plt.subplot(6, 3, 3 * axis + 1)
                ax.plot(sim.traj_pose[:, axis])
                ax.plot(j[1:, axis])
                ax.legend(["ref", "acc"])
                bx = plt.subplot(6, 3, 3 * axis + 2)
                bx.plot(sim.traj_vel[:, axis])
                bx.plot(v[1:, axis])
                bx.legend(["ref", "acc"])
                cx = plt.subplot(6, 3, 3 * axis + 3)
                cx.plot(sim.traj_acc[:, axis])
                cx.plot(a[1:, axis])
                cx.legend(["ref", "acc"])
        else:
            ax = plt.subplot(1, 3, 1)
            ax.plot(sim.traj_pose[:, axis])
            ax.plot(j[1:, axis])
            ax.legend(["ref", "acc"])
            bx = plt.subplot(1, 3, 2)
            bx.plot(sim.traj_vel[:, axis])
            bx.plot(v[1:, axis])
            bx.legend(["ref", "acc"])
            cx = plt.subplot(1, 3, 3)
            cx.plot(sim.traj_acc[:, axis])
            cx.plot(a[1:, axis])
            cx.legend(["ref", "acc"])
        plt.show()


if __name__ == "__main__":
    sim = mj_sim()
    track_vel = 0
    random_move = 0
    grasp_demo = 1

    ################################################ set velocity of the robot ################################################
    ################################################ May NEED MORE EFFORTS ################################################
    if track_vel:
        j, v, a = np.zeros((1, 6)), np.zeros((1, 6)), np.zeros((1, 6))
        for _ in range(3):
            st = time.time()
            target_vel = np.array([np.random.randn() * 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
            print(target_vel)
            while time.time() - st < 1:
                sim.set_joint_velocity(target_vel)
                sim.step()
                j = np.vstack((j, sim.joint_pos))
                v = np.vstack((v, sim.joint_vel))
                a = np.vstack((a, sim.joint_acc))
        axis = 0
        fig = plt.figure()
        ax = plt.subplot(1, 3, 1)
        ax.plot(j[1:, axis])
        ax.legend(["real pos"])
        bx = plt.subplot(1, 3, 2)
        bx.plot(v[1:, axis])
        bx.legend(["real vel"])
        cx = plt.subplot(1, 3, 3)
        cx.plot(a[1:, axis])
        cx.legend(["real acc"])
        plt.show()

    ################################################ randomly move to a joint ################################################
    if random_move:
        # randomly move in the workspace
        for i in range(5):
            target = np.hstack(
                (
                    np.random.uniform(-2, 2, 1),
                    np.random.uniform(-0.7, 0.7, 2),
                    np.random.uniform(-1, 1, 1),
                    np.random.uniform(-np.pi / 1.5, 0.1, 1),
                    np.random.uniform(-3, 3, 1),
                )
            )
            sim.go2jpose(target)
            for _ in range(100):
                sim.step()

    ################################################ grasp demo ################################################

    if grasp_demo:
        # pick and place a box
        for _ in range(5):
            close, fully_open = 0.0175, 0.0
            sim.go2jpose(np.array([0.0, 0.0, 0.0, 0.0, -np.pi / 2, 0.0]))
            sim.go2cpose(np.array([0.5, -0.3, 0.4, 0.0, np.pi, np.pi]))
            sim.go2cpose(np.array([0.5, -0.3, 0.25, 0.0, np.pi, np.pi]))
            sim.set_gripper(close)
            for _ in range(50):
                sim.step()
            sim.go2cpose(np.array([0.5, -0.3, 0.4, 0.0, np.pi, np.pi]))
            for _ in range(50):
                sim.step()
            sim.go2cpose(np.array([0.5, 0.3, 0.4, 0.0, np.pi, np.pi]))
            sim.go2cpose(np.array([0.5, 0.3, 0.25, 0.0, np.pi, np.pi]))
            sim.set_gripper(fully_open)
            for _ in range(50):
                sim.step()
            sim.go2cpose(np.array([0.5, 0.3, 0.4, 0.0, np.pi, np.pi]))
            sim.go2jpose(np.array([0.0, 0.0, 0.0, 0.0, -np.pi / 2, 0.0]))
            for _ in range(50):
                sim.step()
            
            sim.go2jpose(np.array([0.0, 0.0, 0.0, 0.0, -np.pi / 2, 0.0]))
            sim.go2cpose(np.array([0.5, 0.3, 0.4, 0.0, np.pi, np.pi]))
            sim.go2cpose(np.array([0.5, 0.3, 0.25, 0.0, np.pi, np.pi]))
            sim.set_gripper(close)
            for _ in range(50):
                sim.step()
            sim.go2cpose(np.array([0.5, 0.3, 0.4, 0.0, np.pi, np.pi]))
            for _ in range(50):
                sim.step()
            sim.go2cpose(np.array([0.5, -0.3, 0.4, 0.0, np.pi, np.pi]))
            sim.go2cpose(np.array([0.5, -0.3, 0.25, 0.0, np.pi, np.pi]))
            sim.set_gripper(fully_open)
            for _ in range(50):
                sim.step()
            sim.go2cpose(np.array([0.5, -0.3, 0.4, 0.0, np.pi, np.pi]))
            sim.go2jpose(np.array([0.0, 0.0, 0.0, 0.0, -np.pi / 2, 0.0]))
            for _ in range(50):
                sim.step()
                