from gym.envs.registration import register 


register(id='Fanuc_peg_in_hole-v0', entry_point='impedance_envs.envs.Fanuc_peg_hole:Fanuc_peg_in_hole')

from .envs.source import  trajectory_cubic
from .envs.source import  lrmate_kine_base
from .envs.source import  trajectory_cubic_bk