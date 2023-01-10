import ctypes
so = ctypes.cdll.LoadLibrary
lib = so('./bin/mujocosim.so')
# initialize the simulation
lib.warpper_init()
step_num = 0
# build a c type array to hold the temp value
state_temp = (ctypes.c_double *12)(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
time_render_temp = (ctypes.c_double *1)(0.0)
Verbose = (ctypes.c_bool)(False)

# lib.set_Verbose(Verbose)
lib.get_sim_time(time_render_temp)
time_render_pre = time_render_temp[0]

while(step_num<6000):
    state = []
    lib.get_sim_time(time_render_temp)
    time_render_now = time_render_temp[0]
    # print(time_render_now)
    if (time_render_now - time_render_pre > 1.0 / 60.0):
        lib.warpper_update_ui()
        time_render_pre = time_render_now
        step_num= step_num+1
        lib.get_eef_pose_vel(state_temp)
        for s in state_temp:
            state.append(s)
        print("Current state (pose & vel of the eef):")
        print(state)
    lib.warpper_step()
# close simulation and window
lib.warpper_close()
