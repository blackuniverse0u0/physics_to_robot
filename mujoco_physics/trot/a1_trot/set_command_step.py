
def set_command_step(cmd_des,cmd_curr,cmd_min,cmd_max,cmd_rate):

    temp_rate =  (cmd_des - cmd_curr) ;
    abs_rate = abs(temp_rate);
    rate = min(abs_rate,cmd_rate);

    if (cmd_curr > cmd_des):
        cmd = cmd_curr - rate
    elif (cmd_curr < cmd_des):
        cmd = cmd_curr + rate;
    else:
        cmd = cmd_curr;

    if (cmd > cmd_max):
        cmd = cmd_max;

    if (cmd < cmd_min):
        cmd = cmd_min;

    return cmd
