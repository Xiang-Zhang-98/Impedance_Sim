import copy
import ctypes
import os
import sys
import time
import numpy as np
from .source.lrmate_kine_base import FK, IK
# import source.trajectory_cubic as traj
import matplotlib.pyplot as plt
import transforms3d.quaternions as trans_quat
import transforms3d.euler as trans_eul
import gym
from gym import spaces

# To debug in VS Code, add the following line to launch.json under "configurations"
# "env": {"LD_LIBRARY_PATH": "/home/{USER_NAME}/.mujoco/mujoco200/bin/"}


class Fanuc_peg_in_hole(gym.Env):
    def __init__(self, render=True):
        super(Fanuc_peg_in_hole, self).__init__()

        cwd = os.getcwd()

        self.sim = ctypes.cdll.LoadLibrary(cwd + "/impedance_envs/impedance_envs/envs/bin/mujocosim.so")

        # parameters which will be updated every step
        self.joint_pos = np.zeros(6)
        self.joint_vel = np.zeros(6)
        self.joint_acc = np.zeros(6)
        self.pose_vel = np.zeros(13)
        self.full_jacobian = np.zeros((6, 8))
        self.force_sensor_data = np.zeros(3)
        self.force_offset = np.zeros(3)

        self.time_render = np.zeros(1)
        self.gripper_close = False
        self.nv = 8 # subject to change according to different tasks

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
        self.Verbose = False
        self.sim.wrapper_set_verbose(self.Verbose)
        self.Render = render
        self.sim.wrapper_set_render(self.Render)
        jac = [0.0] * 6 * self.nv
        self.jacobian_holder = (ctypes.c_double * len(jac))(*jac)
        self.force_sensor_data_holder = (ctypes.c_double * 3)(
            0.0, 0.0, 0.0
        )
        self.pose_vel_holder = (ctypes.c_double * 13)(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )

        # initialize a PD gain, may need more effort on tunning
        # kp = np.array([17, 17, 17, 17, 17, 17])
        # kv = np.array([40, 40, 40, 40, 40, 40])
        kp = 100 * np.array([1, 1, 1, 1, 1, 1])
        kv = 2 * np.sqrt(kp)  # np.array([40, 40, 40, 40, 40, 40])
        self.set_pd_gain(kp, kv)

        # initialize admittance gains
        self.adm_kp = 10 * np.array([1, 1, 1, 1, 1, 1])
        self.adm_m = 1 * np.array([1, 1, 1, 1, 1, 1])
        self.adm_kd = 2 * np.sqrt(np.multiply(self.adm_kp, self.adm_m))
        self.adm_pose_ref = np.zeros(7)
        self.adm_vel_ref = np.zeros(6)

        # initialize path planning
        self.HZ = 125  # this is the frequency of c++ simulator, set in the xml file
        self.traj_pose, self.traj_vel, self.traj_acc = None, None, None

        # peg-in-hole task setting
        self.eef_offset = np.array([0.3, 0.075, 0.0])
        self.work_space_xy_limit = 4
        self.work_space_z_limit = 4
        self.work_space_rollpitch_limit = np.pi * 5 / 180.0
        self.work_space_yaw_limit = np.pi * 10 / 180.0
        self.work_space_origin = np.array([0.49, -0.0169, 0.1-0.054])
        self.work_space_origin_rotm = np.array([[0.0423, -0.2853, 0.9575],
                                                [-0.2853, 0.9150, 0.2853],
                                                [-0.9575, -0.2853, -0.0427]])
            # np.array([[0,0,1], [0,1,0], [-1,0,0]])
        self.goal = np.array([0, 0, 1])
        self.goal_ori = np.array([0, 0, 0])
        self.noise_level = 0.0
        self.ori_noise_level = 0.0
        self.use_noisy_state = True
        self.state_offset = np.zeros(18)
        self.force_noise = False
        self.force_noise_level = 0.2
        self.force_limit = 10 #2
        self.evaluation = self.Render
        self.moving_pos_threshold = 2.5
        self.moving_ori_threshold = 4

        # RL setting
        self.obs_high = [self.work_space_xy_limit,self.work_space_xy_limit,self.work_space_z_limit,
                         self.work_space_rollpitch_limit,self.work_space_rollpitch_limit,self.work_space_yaw_limit,
                         0.1,0.1,0.1,0.1,0.1,0.1,
                         10,10,10,10,10,10]
        self.obs_high = np.array(self.obs_high)
        self.observation_space = spaces.Box(low=-1., high=1., shape=self.get_RL_obs().shape, dtype=np.float32)
        self.action_space = spaces.Box(low=-np.ones(12), high=np.ones(12), dtype=np.float32)
        self.action_vel_high = 0.1 * np.ones(6)
        self.action_vel_low = -0.1 * np.ones(6)
        self.action_kp_high = 20 * np.ones(6)
        self.action_kp_low = 1 * np.ones(6)

        # initialize the simulation
        self.n_step = 0
        self.sim.wrapper_init()
        self.reset()
        self.set_seed()


    def reset(self):
        init_c_pose = np.array([0.5, 0.0, 0.40, 0.0, np.pi, np.pi])
        l = np.array([3,3,0.5])/100
        cube = np.random.uniform(low=-l, high=l)
        init_c_pose[0:3] = init_c_pose[0:3] + cube
        init_j_pose = IK(init_c_pose)
        self.set_joint_states(init_j_pose, 0*init_j_pose, 0*init_j_pose)
        self.force_calibration()
        # Domain-randomization
        # self.state_offset[:2] = np.random.normal(0, np.array([self.noise_level,self.noise_level]))
        # angle = np.random.normal(0, self.ori_noise_level / 180 * np.pi)
        # self.state_offset[5] = angle
        return self.get_RL_obs()


    def set_seed(self, seed=0):
        np.random.seed(seed)
        import random
        random.seed(seed)


    def sim_step(self):
        self.sim.wrapper_step()
        self.get_joint_states()
        self.get_pose_vel()
        self.get_sensor_data()
        # eff_rotm = trans_quat.quat2mat(self.pose_vel[3:7])
        # world_force = eff_rotm @ (self.force_sensor_data - self.force_offset)
        # print(world_force)
        self.get_jacobian()

    def get_RL_obs(self):
        # eef_pos = self.pose_vel[:3] - self.work_space_origin
        # eef_vel = self.pose_vel[7:]
        # eef_world_rotm = trans_quat.quat2mat(self.pose_vel[3:7])
        eef_pos, eef_world_rotm, eef_vel = self.get_eef_pose_vel(self.pose_vel)
        eef_pos = eef_pos - self.work_space_origin
        eef_rotm = np.linalg.inv(self.work_space_origin_rotm)@eef_world_rotm
        eef_eul = trans_eul.mat2euler(eef_rotm)
        world_force = np.zeros(6)
        eef_force = self.force_sensor_data - self.force_offset
        world_force[:3] = eef_world_rotm @ eef_force
        if self.force_noise:
            world_force = world_force + np.random.normal(0, self.force_noise_level, 6)
        world_force = np.clip(world_force, -10, 10)
        state = np.concatenate([100*eef_pos, eef_eul, eef_vel, world_force])
        # state = np.clip(state, -self.obs_high, self.obs_high)
        #return state + self.state_offset
        return state

    def process_action(self, action):
        desired_vel = np.clip(action[:6], -1, 1)
        desired_kp = np.clip(action[6:12], -1, 1)
        desired_vel = (self.action_vel_high + self.action_vel_low)/2 + np.multiply(desired_vel, (self.action_vel_high - self.action_vel_low)/2)
        desired_kp = (self.action_kp_high + self.action_kp_low)/2 + np.multiply(desired_kp, (self.action_kp_high - self.action_kp_low)/2)
        return desired_vel, desired_kp

    def step(self, action):
        # step function for RL
        desired_vel, desired_kp = self.process_action(action)
        self.adm_kp = desired_kp
        self.adm_kd = 2 * np.sqrt(np.multiply(self.adm_kp, self.adm_m))
        init_ob = self.get_RL_obs()
        for i in range(50):
            ob = self.get_RL_obs()
            curr_force = ob[12:]
            if np.abs(np.dot(curr_force, desired_vel) / np.linalg.norm(desired_vel + 1e-6, ord=2)) > self.force_limit:
                break
            delta_ob = ob - init_ob
            if np.linalg.norm(delta_ob[0:3], ord=2) > self.moving_pos_threshold or np.linalg.norm(delta_ob[3:6], ord=2)\
                    > self.moving_ori_threshold / 180 * np.pi:
                break
            if np.abs(ob[0]) > self.work_space_xy_limit:
                desired_vel[0] = -self.action_vel_high[0] * np.sign(ob[0])
            if np.abs(ob[1]) > self.work_space_xy_limit:
                desired_vel[1] = -self.action_vel_high[1] * np.sign(ob[1])
            if np.abs(ob[3]) > self.work_space_rollpitch_limit:
                desired_vel[3] = -self.action_vel_high[3] * np.sign(ob[3])
            if np.abs(ob[4]) > self.work_space_rollpitch_limit:
                desired_vel[4] = -self.action_vel_high[4] * np.sign(ob[4])
            if np.abs(ob[5]) > self.work_space_yaw_limit:
                desired_vel[5] = -self.action_vel_high[5] * np.sign(ob[5])
            if ob[2] > self.work_space_z_limit:
                desired_vel[2] = -self.action_vel_high[2]
            # check done
            if np.linalg.norm(ob[0:3] - self.goal) < 0.3:
                done = False
                desired_vel = np.zeros(6)  # if reach to goal, then stay
                self.adm_pose_ref = self.pose_vel[:7]
                self.adm_vel_ref = desired_vel
                target_joint_vel = self.Cartersian_vel_control(desired_vel)
            else:
                done = False
                self.adm_pose_ref = self.pose_vel[:7]
                self.adm_vel_ref = desired_vel
                target_joint_vel = self.admittance_control()

            # # self.adm_kp = desired_kp
            # # self.adm_kd = np.sqrt(np.multiply(self.adm_kp, self.adm_m))
            # self.adm_pose_ref = self.pose_vel[:7]
            # # self.adm_pose_ref[:3] = self.adm_pose_ref[:3] + 0.02*self.moving_pos_threshold*desired_vel[:3]/np.linalg.norm(desired_vel[:3], ord=2)
            # # adm_eul = trans_eul.quat2euler(self.pose_vel[3:7]) + 2/ 180 * np.pi * self.moving_ori_threshold * desired_vel[3:6]/np.linalg.norm(desired_vel[3:6], ord=2)
            # # self.adm_pose_ref[3:7] = trans_eul.euler2quat(adm_eul[0], adm_eul[1], adm_eul[2], axes='sxyz')
            # self.adm_vel_ref = desired_vel
            # # Adm or vel ctl?
            # target_joint_vel = self.admittance_control()
            # # target_joint_vel = self.Cartersian_vel_control(desired_vel)

            self.set_joint_velocity(target_joint_vel)
            self.sim_step()

        ob = self.get_RL_obs()
        # evalute reward
        dist = np.linalg.norm(ob[0:3] - self.goal)
        # print(dist)
        # time.sleep(0.5)
        if dist < 0.3:
            done = False
            reward = 1000
        else:
            done = False
            reward = np.power(10, 3 - dist)
        # reward = -dist
        if self.evaluation and dist < 0.5:
            done = True
        # print(reward)
        return ob, reward, done, dict(reward_dist=reward)

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

    def get_pose_vel(self):
        self.sim.get_eef_pose_vel(self.pose_vel_holder)
        self.pose_vel = np.array(self.pose_vel_holder[:13])

    def get_jacobian(self):
        self.sim.wrapper_eef_full_jacobian(self.jacobian_holder)
        self.full_jacobian = np.array(self.jacobian_holder).reshape(6, self.nv)

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

    def set_joint_states(self, joint, vel, acc):
        joint_holder = (ctypes.c_double * 8)(joint[0], joint[1], joint[2], joint[3], joint[4], joint[5], 0.0, 0.0)
        vel_holder = (ctypes.c_double * 8)(vel[0], vel[1], vel[2], vel[3], vel[4], vel[5], 0.0, 0.0)
        acc_holder = (ctypes.c_double * 8)(acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], 0.0, 0.0)
        self.sim.wrapper_set_joint_states(joint_holder, vel_holder, acc_holder)

    def set_pd_gain(self, kp, kv):
        kp_mj = (ctypes.c_double * 6)(kp[0], kp[1], kp[2], kp[3], kp[4], kp[5])
        kv_mj = (ctypes.c_double * 6)(kv[0], kv[1], kv[2], kv[3], kv[4], kv[5])
        self.sim.wrapper_update_pd_gain(kp_mj, kv_mj)

    def set_controller(self, controller_idx=0):
        controller_idx_mj = (ctypes.c_int)(controller_idx)
        self.sim.wrapper_update_controller_type(controller_idx_mj)

    def force_calibration(self, H=100):
        force_history = np.zeros([H, 3])
        self.sim_step()
        self.set_reference_traj(
            self.joint_pos, 0 * self.joint_vel, 0 * self.joint_acc
        )
        for _ in range(H):
            self.sim_step()
            force_history[_, :] = self.force_sensor_data
        self.force_offset = np.mean(force_history[int(H / 2):], axis=0)

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

    def set_joint_velocity(self, target_vel):
        T = 1 / self.HZ
        target_pos = self.joint_pos + T * (self.joint_vel + target_vel) / 2
        target_acc = (target_vel - self.joint_vel) / T
        self.set_reference_traj(target_pos, target_vel, target_acc)

    def Cartersian_vel_control(self, vel):
        Full_Jacobian = self.full_jacobian
        Jacobian = Full_Jacobian[:6, :6]
        target_joint_vel = np.linalg.pinv(Jacobian) @ vel
        return target_joint_vel

    def skew_symmetric(self,vec):
        return np.array([[0, -vec[2], vec[1]],
                         [vec[2], 0, -vec[0]],
                         [-vec[1], vec[0], 0]])

    def get_eef_pose_vel(self, link6_pose_vel):
        link6_pos = link6_pose_vel[:3]
        link6_rotm = trans_quat.quat2mat(link6_pose_vel[3:7])
        link6_vel = link6_pose_vel[7:]

        eef_pos = link6_pos + link6_rotm @ self.eef_offset
        eef_rotm = copy.copy(link6_rotm)

        # Let's forget kinematics and assume eef and link 6 have the same vel
        # OH we cannot do that
        eef_vel = copy.copy(link6_vel)
        eef_vel[:3] = link6_vel[:3] + self.skew_symmetric(link6_vel[3:6]) @ link6_rotm @ self.eef_offset
        return eef_pos, eef_rotm, eef_vel

    def get_link6_pose_vel_from_eef(self, eef_pos, eef_rotm, eef_vel):
        link6_rotm = copy.copy(eef_rotm)
        link6_pos = eef_pos - link6_rotm @ self.eef_offset
        link6_vel = copy.copy(eef_vel)
        link6_vel[:3] = eef_vel[:3] - self.skew_symmetric(eef_vel[3:6]) @ link6_rotm @ self.eef_offset

        return link6_pos, link6_rotm, link6_vel


    def admittance_control(self, ctl_ori=True):
        ## Get robot motion from desired dynamics
        eef_pos, eef_rotm, eef_vel = self.get_eef_pose_vel(self.pose_vel)
        eef_pos_d, eef_rotm_d, eef_vel_d = self.get_eef_pose_vel(np.concatenate([self.adm_pose_ref, self.adm_vel_ref]))
        eef_vel_d = self.adm_vel_ref

        # process force
        world_force = np.zeros(6)
        eef_force = self.force_sensor_data - self.force_offset
        world_force[:3] = eef_rotm @ eef_force
        world_force = np.clip(world_force, -10, 10)

        # dynamics
        e = np.zeros(6)
        e[:3] = eef_pos_d - eef_pos
        if ctl_ori:
            # eff_rotm_d = trans_quat.quat2mat(self.adm_pose_ref[3:7])
            eRd = eef_rotm.T @ eef_rotm_d
            dorn = trans_quat.mat2quat(eRd)
            do = dorn[1:]
            e[3:] = do

        e_dot = eef_vel_d - eef_vel
        MA = world_force + np.multiply(self.adm_kp, e) + np.multiply(self.adm_kd, e_dot)
        adm_acc = np.divide(MA, self.adm_m)
        T = 1 / self.HZ
        adm_vel = eef_vel + adm_acc * T # This vel is for eef not link6, which we can control
        if not ctl_ori:
            adm_vel[3:] = 0 * adm_vel[3:]
        link6_pos, link6_rotm, link6_vel = self.get_link6_pose_vel_from_eef(eef_pos, eef_rotm, adm_vel)
        # link6_vel = adm_vel
        # adm_vel = self.pose_vel[7:] + np.array([0.2,0,0,0,0,0])#adm_acc * T

        Full_Jacobian = self.full_jacobian
        Jacobian = Full_Jacobian[:6, :6]
        target_joint_vel = np.linalg.pinv(Jacobian) @ link6_vel

        return target_joint_vel



if __name__ == "__main__":
    sim = Fanuc_peg_in_hole()
    for _ in range(10):
        sim.reset()
        for i in range(10):
            action = np.random.uniform(low=-1, high=1, size=12)
            # action = np.ones(12)
            # action[0:6] = np.zeros(6)
            # action[2] = -1
            sim.step(action)
    
