from gym.envs.registration import register 


register(id='Fanuc_peg_in_hole-v0', entry_point='impedance_envs.envs.Fanuc_peg_hole:Fanuc_peg_in_hole')
register(id='Fanuc_pivoting-v0', entry_point='impedance_envs.envs.Fanuc_pivoting:Fanuc_pivoting')
register(id='Fanuc_pivoting_easy-v0', entry_point='impedance_envs.envs.Fanuc_pivoting_easy_setup:Fanuc_pivoting')

from .envs.source import  trajectory_cubic
from .envs.source import  lrmate_kine_base
from .envs.source import  trajectory_cubic_bk