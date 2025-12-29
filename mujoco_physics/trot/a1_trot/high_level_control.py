import globals
import numpy as np
from parameters import parms
from set_command_step import set_command_step

def high_level_control():
    if (globals.prev_step < globals.step):
        globals.prev_step = globals.step

        vx = globals.xdot_ref
        vx_ = 1 #0.1; #desired
        globals.xdot_ref =set_command_step(vx_,vx,parms.vx_min,parms.vx_max,parms.dvx)

        # vy = globals.ydot_ref
        # vy_ = 0.4; #desired
        # globals.ydot_ref =set_command_step(vy_,vy,parms.vy_min,parms.vy_max,parms.dvy)

        omega = globals.psidot_ref
        omega_ = -0.7; #desired
        globals.psidot_ref =set_command_step(omega_,omega,parms.omega_min,parms.omega_max,parms.domega)
