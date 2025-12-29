import mujoco as mj 
import time
from modern_robotics import skew,vec_to_so3,vec_to_se3,get_space_jacobian
import numpy as np 

xml_path = 'robot_pos.xml'
# xml_path = 'robot_torque.xml'
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)

# Homing Pose : M 
data.qpos[:] = 0 
mj.mj_kinematics(model,data) # TODO: 0을 넣고 값을 초기화 시킴.

# robot pose and orientation 
base_name = 'base_link'
base_id = mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,base_name)

# Homing pose에서의 base 정보(기준점)
# TODO: 로봇이 움직인다면 계속 업데이트 해야함 
robot_pos_world = data.xpos[base_id].copy()
# array([0. , 0. , 0.5])

robot_ori_world = data.xmat[base_id].copy()
# array([1., 0., 0., 0., 1., 0., 0., 0., 1.])
# (Pdb) robot_ori_world.shape
# (9,)
# 따라서 행렬로 만들어줘야함 
robot_ori_world = data.xmat[base_id].reshape(3,3).copy()

def get_base_to_screw_params(joint_names):
    
    twists = [] # TODO: twist가 맞는지? 
    # 정의
    # twist  = v = [w,v] R^6    
    # S = [omega, v]
    # omega: 조인트 회전축의 단위 벡터 (unit axis of rotation)
    # v = -omega x q : 조인트 축 위의 한 점 q에 의해 결정되는 원점에서의 선속도
    # 위에서 표현하는건 base frame기준
    """
    Calculates the Screw Axes (S) for each joint relative to the Base Frame.
    
    Terminology:
    - Screw Axis (S) = [omega, v] in R^6
    - omega (w): Unit vector representing the direction of the joint axis.
    - v: Linear velocity at the Base Frame origin when the joint rotates.
        Calculated as v = -omega x q (or q x omega).
    """
    
    for i,name in enumerate(joint_names):
        j_id = mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,name)
        p_joint_world = data.xanchor[j_id]
        # 관절의 회전 중심점 위치 pos world 기준 
        # (3,)

        axis_joint_world = data.xaxis[j_id]
        # 관절의 회전 축 벡터 
        # (3,)
        # (Pdb) axis_joint_world
        # array([1., 0., 0.])
        
        # Base Frame기준으로 변환 
        pos_base = robot_ori_world.T@(p_joint_world - robot_pos_world)
        rot_base = robot_ori_world.T @ axis_joint_world 

        if i == len(joint_names)-1:
            M = np.eye(4)
            M[:3,:3] = vec_to_so3(skew(rot_base),0)
            M[:3,3] = pos_base

        v =-skew(rot_base)@pos_base
        twist = np.concatenate((rot_base,v)) #TODO: 이게 twist를 뜻하는게 맞나?
        twists.append(twist)
        
    return twists,M 

# --- 3. Parameter Loading ---
target_leg_joints = ["FL_j1", "FL_j2", "FL_j3"]
twists,M = get_base_to_screw_params(target_leg_joints)

t0,t1,t2 = 0,0,0 

T2 = vec_to_se3(twists[2],t2)
T1 = vec_to_se3(twists[1],t1)
T0 = vec_to_se3(twists[0],t0)

qpos = T0@T1@T2@M

# theta를 움직이게한후 
# mujoco q_pos + calculated q_pos비교 


# MuJoCo 뷰어 실행
with mj.viewer.launch_passive(model, data) as viewer:
    
    # [수정 1, 2] if 대신 while 사용, 함수 호출 () 추가
    while viewer.is_running(): 
        
        step_start = time.time() # 루프 시작 시간 기록

        # 현재 상태 읽기 (필요 시)
        # 주의: data.qpos는 참조(reference)이므로 값 저장이 필요하면 .copy()를 써야 함
        q = data.qpos 
        qvel = data.qvel 
        
        # 물리 연산 1스텝 진행
        mj.mj_step(model, data)

        # 뷰어 화면 업데이트
        viewer.sync()

        # [선택 사항] 시뮬레이션 속도를 실제 시간과 비슷하게 맞추기
        # 이 코드가 없으면 컴퓨터 성능에 따라 너무 빨리 재생될 수 있습니다.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)