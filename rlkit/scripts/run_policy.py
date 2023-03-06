from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import uuid
from rlkit.core import logger
import impedance_envs
import gym


filename = str(uuid.uuid4())


def simulate_policy(args):
    data = torch.load(args.file)
    policy = data['evaluation/policy']
    # env = data['evaluation/env']
    if args.env == "peg":
        env = gym.make('Fanuc_peg_in_hole-v0',render=True)
    elif args.env == "pivoting":
        env = gym.make('Fanuc_pivoting-v0',render=True)
    elif args.env == "pivoting_easy":
        env = gym.make('Fanuc_pivoting_easy-v0',render=True)
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    while True:
        if args.reload:
            policy = data['evaluation/policy']
            print("Policy loaded")
            if args.gpu:
                set_gpu_mode(True)
                policy.cuda()
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
        )
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--reload', action='store_true')
    parser.add_argument('--env', type=str)
    args = parser.parse_args()

    simulate_policy(args)
