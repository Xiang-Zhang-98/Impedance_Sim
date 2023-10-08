import gym
import numpy as np
import argparse
import impedance_envs
import torch
from rlkit.torch.pytorch_util import set_gpu_mode
# def get_robot_skill(state, traj_policy, gain_policy):
#     traj, _ = traj_policy.get_action(state)

#     return action


def rollout_one_episode(env, H, policy):
    state = env.reset()
    max_steps = H
    path = dict(
        states=[],
        actions=[],
        rewards=[],
        terminals=[],
        next_states=[],
        agent_infos=[],
        env_infos=[]
    )
    for i in range(max_steps):
        action,_ = policy.get_action(state)
        path['states'].append(state)
        path['actions'].append(action)
        state, reward, terminal, _ = env.step(action)
        path['next_states'].append(state)
        path['rewards'].append(reward)
        path['terminals'].append(terminal)
        if terminal:
            break
    
    for key in path.keys():
        path[key] = np.array(path[key])
    return path
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str,
                        default='rlkit/data/Fanuc-peg-in-hole-random-peg-pos-w-uncertainty-f=0.3-correct-ctl/Fanuc_peg_in_hole_random_peg_pos_w_uncertainty_f=0.3_correct_ctl_2023_03_16_16_02_09_0000--s-0/params.pkl')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true', default=True)
    parser.add_argument('--reload', action='store_true')
    parser.add_argument('--env', type=str)
    parser.add_argument('--episodes', type=int, default=100)
    args = parser.parse_args()

    env = gym.make('Fanuc_peg_in_hole-v0',render=False)
    policy = torch.load(args.file)['evaluation/policy']
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    for i in range(args.episodes):
        path = rollout_one_episode(
            env,
            args.H,
            policy,
        )