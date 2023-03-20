import numpy as np
from ubuntu_controller import robot_controller
from scipy.spatial.transform import Rotation as R
import gym
import impedance_envs
import sys
import time
from datetime import datetime
import joblib
import torch
# sys.path.insert(1, './MP-DQN')
# from agents.pdqn_multipass_threshold_3prms_v2 import MultiPassPDQNAgent_Threshold
# from common.platform_domain import PlatformFlattenedActionWrapper
# from common.wrappers import ScaledStateWrapper

def pad_action(act, act_param):
    # params = [np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)]
    # params[act][:] = act_param
    # return (act, params)
    return [act, act_param]

class insertion_primitive(object):
    """
    Peg-in-hole environment
    """
    def __init__(self):
        self.record = True
        self.controller = robot_controller()
        self.Mass = np.array([1,1,1])   # to determine
        self.Inertia = 1*np.array([0.1, 0.1, 0.1])   # to determine
        # self.goal_pose = np.array([0.56046,-0.00418,-0.1256])
        self.true_goal_pose = np.array([0.558,0.0,-0.15768])
        self.noise_level = 0.0
        self.goal_pose = np.array([0.558,0.0,-0.15768])
        self.Kp = np.array([200,200,200,200,200,200])
        self.Kd = np.array([500,500,500,250,250,250])
        self.offset = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        # Load policy
        self.agent = None
        self.seed = None
        self.env = None
        self.load_agent()

        #experiment settings
        self.moving_pos_limit = 2.5
        self.moving_ori_limit = 4/180*np.pi
        self.execute_time = 10
        self.max_steps = 40
        self.contact_time_limit = 2

        # Sim-to-real
        self.sim2real = False

        # action space limit
        self.action_vel_high = 0.1 * np.array([1,1,0,1,1,1])
        self.action_vel_low = -0.1 * np.ones(6)
        self.action_kp_high = 200 * np.array([1,1,1,10,10,10])
        self.action_kp_low = 1 * np.array([1,1,1,10,10,10])

    def workspace_limit(self,TCP_d_pos, TCP_d_euler, TCP_d_vel):
        # unit is cm
        x_limit = 4
        y_limit = 4
        z_limit = 4
        pos_limit = np.array([x_limit,y_limit,z_limit])
        rollpitch_limit = 5
        yaw_limit = 10
        ori_limit = np.array([rollpitch_limit,rollpitch_limit,yaw_limit])/180*np.pi
        vel_limit = 0.01
        TCP_d_pos = np.clip(TCP_d_pos, self.true_goal_pose*100 - pos_limit, self.true_goal_pose*100 + pos_limit)
        TCP_d_euler = np.clip(TCP_d_euler, -ori_limit, ori_limit)
        TCP_d_vel = np.clip(TCP_d_vel, -vel_limit, vel_limit)
        return TCP_d_pos, TCP_d_euler, TCP_d_vel


    def get_primitive(self,state):
        action, _ = self.agent.get_action(state)
        # print(velcmd)
        desired_vel = np.clip(action[:6], -1, 1)
        desired_kp = np.clip(action[6:12], -1, 1)
        desired_vel = (self.action_vel_high + self.action_vel_low)/2 + np.multiply(desired_vel, (self.action_vel_high - self.action_vel_low)/2)
        desired_kp = (self.action_kp_high + self.action_kp_low)/2 + np.multiply(desired_kp, (self.action_kp_high - self.action_kp_low)/2)
        return desired_vel, desired_kp, None

    def experiment(self, init_time):
        step = 0
        not_done = True
        import copy

        # recording
        data = dict(
            observations=[],
            # primitive_id=[],
            vel=[],
            kp=[]
        )
        begin_time = time.time()
        print("Begin one experiment")
        self.goal_pose[0:2] = self.true_goal_pose[0:2] + np.random.normal(0,self.noise_level/100,2)
        while(step < self.max_steps and not_done):
            # get robot state
            state = self.get_state()
            print(state)
            # get velocity cmd
            vel_cmd, kp, action = self.get_primitive(state)

            vel_cmd[-1] = - vel_cmd[-1]
            
            # change kp kd
            self.Kp[:3] = kp[:3]
            self.Kd[:3] = 20 * np.sqrt(np.multiply(self.Kp[:3], self.Mass))
            # recording data
            data["observations"].append(state)
            # data["primitive_id"].append(action)
            data["vel"].append(vel_cmd)
            data["kp"].append(kp)
            current_robot_pos = state[0:6]
            current_robot_pos[0:3] = current_robot_pos[0:3] + self.goal_pose*100
            pos_cmd = np.zeros(6)
            pos_cmd[0:3] = current_robot_pos[0:3] + vel_cmd[0:3]/np.linalg.norm(vel_cmd[0:3]+1e-6,ord = 2) * self.moving_pos_limit
            pos_cmd[3:6] = current_robot_pos[3:6] + vel_cmd[3:6]/np.linalg.norm(vel_cmd[3:6]+1e-6,ord = 2) * self.moving_ori_limit
            print(pos_cmd)
            begin_time = time.time()
            not_contacted = True
            not_done = True
            contact_time = None
            while(time.time() - begin_time < self.execute_time and not_contacted):
                self.send_commend(pos_cmd[0:3],pos_cmd[3:6],np.zeros(6))
                state = self.get_state()
                delta_2_goal = np.abs(state[2] - 1)
                if delta_2_goal<0.5:
                    not_done = False

            print("Finished one primitive, select another one")
            step = step+1
        
        # process data
        data["observations"] = np.array(data["observations"])
        data["vel"] = np.array(data["vel"])
        data["kp"] = np.array(data["kp"])

        now = datetime.now()
        current_time = now.strftime("%H-%M-%S")
        filename = 'exp_data/'+init_time+'/'+current_time+'square_vic.pkl'
        if self.record:
            with open(filename, 'wb') as f:
                joblib.dump(data, f)
        print("Experiment finished, result is:")
        print(not not_done)

    def reset(self):
        # RESET GAINS
        self.Mass = np.array([2,2,2])*1.5
        self.Inertia = 1*np.array([2, 2, 2]) 
        self.Kp = np.array([200,200,200,200,200,200])
        self.Kd = np.array([500,500,500,250,250,250])

        angle = np.pi/180*0
        yaw = np.random.uniform(low=-np.pi / 180 * 5, high=np.pi / 180 * 5)
        l = np.array([3,3,0.5])
        cube = np.random.uniform(low=-l, high=l)
        mb = (cube + np.array([0,0,3.5]))/100 + self.goal_pose
        delta_pose = 10
        TCP_d_pos = mb
        TCP_d_euler = np.array([0,angle,yaw])
        TCP_d_vel = np.zeros(6)
        d_rotm = R.from_euler('xyz', TCP_d_euler).as_matrix()
        TCP_d_euler = R.from_matrix(d_rotm @ self.offset).as_euler('ZYX')
        while(delta_pose>0.001):
            UDP_cmd = np.hstack([TCP_d_pos, TCP_d_euler, TCP_d_vel, self.Kp, self.Kd, self.Mass, self.Inertia])
            self.controller.send(UDP_cmd)

            self.controller.receive()
            delta_pose = np.linalg.norm(TCP_d_pos-self.controller.robot_pose[0:3])
        print("Reset finished")
        # RESET GAINS
        self.Mass = np.array([1,1,1])
        self.Inertia = 1*np.array([2, 2, 2])
        self.Kp = np.array([200,200,200,200,200,200])
        self.Kd = np.array([500,500,500,250,250,250])

    def load_agent(self):
        data = torch.load('/home/fanuc/Xiang/Impedance_Sim/rlkit/data/Fanuc-peg-in-hole-random-peg-pos-w-uncertainty-f=0.3-correct-ctl/Fanuc_peg_in_hole_random_peg_pos_w_uncertainty_f=0.3_correct_ctl_2023_03_16_16_02_09_0000--s-0/params.pkl')
        self.agent = data['evaluation/policy']
        # self.env = data['evaluation/env']
        from rlkit.torch.pytorch_util import set_gpu_mode
        set_gpu_mode(True)
        self.agent.cuda()
        print("Policy loaded")

    def get_state(self):
        self.controller.receive()
        TCP_pos = self.controller.robot_pose[0:3]
        TCP_rotm = self.controller.robot_pose[3:12].reshape([3,3]).T
        TCP_rotm = TCP_rotm @ self.offset.T
        TCP_euler = R.from_matrix(TCP_rotm).as_euler('xyz')
        TCP_vel = self.controller.robot_vel
        TCP_wrench = self.controller.TCP_wrench
        World_force = TCP_rotm @ TCP_wrench[0:3]
        World_torque = TCP_rotm @ TCP_wrench[3:6]*0 # *0 to cancel torque

        # saturate force to make reral like sim
        # World_force = 10* np.clip(World_force, -1,1)
        World_force = -World_force
        state = np.hstack([TCP_pos,TCP_euler,TCP_vel,World_force,World_torque])
        state[0:3] = state[0:3] - self.goal_pose
        # keep in mind, the unit of state in simulation is cm. However, the real robot uses m.
        state[0:3] = state[0:3] * 100
        # state[6:9] = state[6:9] * 100
        return state

    def send_commend(self, TCP_d_pos, TCP_d_euler, TCP_d_vel):
        TCP_d_pos, TCP_d_euler, TCP_d_vel = self.workspace_limit(TCP_d_pos, TCP_d_euler, TCP_d_vel)
        d_rotm = R.from_euler('xyz', TCP_d_euler).as_matrix()
        TCP_d_euler = R.from_matrix(d_rotm @ self.offset).as_euler('ZYX')
        UDP_cmd = np.hstack([TCP_d_pos/100, TCP_d_euler, TCP_d_vel, self.Kp, self.Kd, self.Mass, self.Inertia])
        self.controller.send(UDP_cmd)

if __name__ == "__main__":
    prm = insertion_primitive()
    now = datetime.now()
    current_time = now.strftime("%y-%m-%d")
    from pathlib import Path
    Path("exp_data/"+current_time+"/Fance_peg_in_hole").mkdir(parents=True, exist_ok=True)
    filename = 'exp_data/'+current_time+"/Fance_peg_in_hole"+'/init_time.pkl'
    with open(filename, 'wb') as f:
        joblib.dump(current_time, f)
    for i in range(10):
        prm.reset()
        prm.experiment(current_time+"/Fance_peg_in_hole")
    