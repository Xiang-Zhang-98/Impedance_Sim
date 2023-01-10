#include <iostream>
#include <stdlib.h>
#include "stdio.h"
#include "string.h"

#include "mujoco.h"
#include "glfw3.h"
#include "mjxmacro.h"
#include "uitools.h"

using namespace std;

//-------------------------------- global -----------------------------------------------
// MuJoCo data structures
mjModel *m = NULL; // MuJoCo model
mjData *d = NULL;  // MuJoCo data
mjvCamera cam;     // abstract camera
mjvOption opt;     // visualization options
mjvScene scn;      // abstract scene
mjrContext con;    // custom GPU context

// controller parameters
mjtNum ref_joint[8];
mjtNum ref_vel[8];
mjtNum ref_acc[8];
mjtNum kp[64];
mjtNum kv[64];

// gripper status
bool gripper_close = false;

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0;
double lasty = 0;

// constants
const int maxgeom = 5000;         // preallocated geom array in mjvScene
const double syncmisalign = 0.1;  // maximum time mis-alignment before re-sync
const double refreshfactor = 0.7; // fraction of refresh available for simulation

// keyboard callback
void keyboard(GLFWwindow *window, int key, int scancode, int act, int mods)
{
    // backspace: reset simulation
    if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE)
    {
        mj_resetData(m, d);
        mj_forward(m, d);
    }
}

// mouse button callback
void mouse_button(GLFWwindow *window, int button, int act, int mods)
{
    // update button state
    button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}

// mouse move callback
void mouse_move(GLFWwindow *window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if (!button_left && !button_middle && !button_right)
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if (button_right)
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if (button_left)
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(m, action, dx / height, dy / height, &scn, &cam);
}

// scroll callback
void scroll(GLFWwindow *window, double xoffset, double yoffset)
{
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn, &cam);
}

void update_ui(GLFWwindow *window)
{
    // get framebuffer viewport
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
    // update scene and render
    mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);
    // swap OpenGL buffers (blocking call due to v-sync)
    glfwSwapBuffers(window);
    // process pending GUI events, call GLFW callbacks
    glfwPollEvents();
}

void computed_torque_control(const mjModel *m, mjData *d)
{
    // d->ctrl controls the actuator in xml file; d->qfrc_applied torques specified in joint space
    // If in xml, actuators are position/velocity, then d->ctrl here means reference position/velocity
    // d->ctrl 0-5 are joints torque's control signal; 6,7 are gripper position's control signal
    // d->qpos, d->qvel, d->qacc are joint positions, velocities, and acculations (0-5 for arm joints; 6,7 for gripper)
    if (m->nu == m->nv)
    {
        // ctrl = M * (ref_acc - kv*e_dot - kp*e) + qfrc_inverse
        mj_rne(m, d, 0, d->qfrc_inverse);
        for (int i = 0; i < m->nv; i++)
            d->qfrc_inverse[i] += m->dof_armature[i] * d->qacc[i] - d->qfrc_passive[i] - d->qfrc_constraint[i];
        mjtNum e[8];               // theta - theta_d
        mjtNum e_dot[8];           // theta_dot - theta_d_dot
        mjtNum kve_dot[8];         // kv * e_dot
        mjtNum kpe[8];             // kp * e
        mjtNum inertial_pd[8];     // ref_acc - kv*e_dot - kp*e
        mjtNum inertial_torque[8]; // M * (ref_acc - kv*e_dot - kp*e)
        mju_sub(e, d->qpos, ref_joint, 8);
        mju_sub(e_dot, d->qvel, ref_vel, 8);
        mju_mulMatVec(kve_dot, kv, e_dot, 8, 8);
        mju_mulMatVec(kpe, kp, e, 8, 8);
        mju_sub(inertial_pd, ref_acc, kve_dot, 8);
        mju_subFrom(inertial_pd, kpe, 8);
        mj_mulM(m, d, inertial_torque, inertial_pd);
        mju_add(d->ctrl, inertial_torque, d->qfrc_inverse, 8);
        // set gripper ctrl
        float gripper_pose = 0.0; // default open the gripper
        if (gripper_close)
            gripper_pose = 0.042;
        // 0.042 gripper fully close; 0.0 fully open
        d->ctrl[6] = gripper_pose;
        d->ctrl[7] = gripper_pose;
    }
}

int main(void)
{
    // print version, check compatibility
    cout << "MuJoCo Pro library version " << 0.01 * mj_version() << endl;
    if (mjVERSION_HEADER != mj_version())
        mju_error("Headers and library have different versions");
    char *license_path = getenv("MUJOCO_LICENSE_PATH");
    mj_activate(license_path); // Check MuJoCo activation license

    char error[1000];
    m = mj_loadXML("source/LRMate_200iD.xml", 0, error, 1000); // MuJoCo model
    d = mj_makeData(m);                                        // MuJoCo data

    if (!glfwInit())
        mju_error("Could not initialize GLFW"); // init GLFW

    // create window, make OpenGL context current, request v-sync
    GLFWwindow *window = glfwCreateWindow(1800, 1200, "Fanuc Robot Simulation", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // create scene and context
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    // init timer for control and ui
    mjtNum timer_render = d->time;

    // set up a controller, we'll not use this as we want to have more freedoms of the control loop
    // mjcb_control = computed_torque_control;

    // run main loop, target real-time simulation and 60 fps rendering
    while (!glfwWindowShouldClose(window))
    {
        // advance interactive simulation for 1/60 sec
        //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
        //  this loop will finish on time for the next frame to be rendered at 60 fps.
        //  Otherwise add a cpu timer and exit this loop when it is time to render.
        if (d->time - timer_render > 1.0 / 60.0)
        {
            update_ui(window);
            timer_render = d->time;
        }

        mj_step1(m, d); // frequency about 20Hz
        computed_torque_control(m, d);
        mj_step2(m, d);
    }

    //free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free MuJoCo model and data, deactivate
    mj_deleteData(d);
    mj_deleteModel(m);
    mj_deactivate();

// terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
    glfwTerminate();
#endif
    cout << "Have a nice day :)" << endl;
    return 1;
}