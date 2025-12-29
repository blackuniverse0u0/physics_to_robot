import mujoco as mj
import mujoco.viewer
import numpy as np
import time
from modern_robotics import vec_to_se3,space_jacobian

# --- 1. FK 및 Screw Axis 계산 함수 ---
def get_base_to_screw_params(model, data, base_name, joint_names, ee_body_name):
    # Homing Pose (q=0) 초기화
    data.qpos[:] = 0
    mj.mj_kinematics(model, data) 
    
    base_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, base_name) 
    R_wb = data.xmat[base_id].reshape(3, 3).copy()
    p_wb = data.xpos[base_id].copy()
    
    twists = []
    
    # Screw Axis (S) 계산
    for name in joint_names:
        j_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        p_j_w = data.xanchor[j_id]  # 3,
        axis_j_w = data.xaxis[j_id] # 3,
        
        # Base Frame 기준으로 변환
        omega = R_wb.T @ axis_j_w
        q = R_wb.T @ (p_j_w - p_wb)
        v = np.cross(-omega, q)
        S = np.concatenate((omega, v))
        twists.append(S)
        
    # M (Home Configuration) 계산
    ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, ee_body_name)
    p_ee_w = data.xpos[ee_id]
    R_ee_w = data.xmat[ee_id].reshape(3, 3)
    
    M = np.eye(4)
    M[:3, :3] = R_wb.T @ R_ee_w 
    M[:3, 3] = R_wb.T @ (p_ee_w - p_wb) 
    
    return twists, M

# --- 2. 초기 설정 ---
xml_path = 'robot_pos.xml'
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

base_name = 'base_link'
legs = ["FL","FR","RL","RR"]
target_joints = ["FL_j1", "FL_j2", "FL_j3"]
end_effector = "FL_calf" # 주의: FL_calf 바디의 원점(무릎 관절 위치)을 의미함

# 운동학 파라미터 추출 (한 번만 실행)
twists, M = get_base_to_screw_params(model, data, base_name, target_joints, end_effector)

# --- 3. 뷰어 실행 및 애니메이션 ---
print("Simulating... Press ESC to exit.")

with mj.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    
    while viewer.is_running():
        # 시간 흐름 계산
        t = time.time() - start_time
        
        # [애니메이션] 관절 각도 생성 (Sin 파형으로 움직임 생성)
        # q1: 롤링, q2: 피칭(허벅지), q3: 피칭(정강이)
        theta1 = 0.5 * np.sin(2 * t)         # 좌우 흔들림
        theta2 = 0.5 * np.sin(2 * t + 1.0)   # 앞뒤 걷기 동작
        theta3 = -0.5 * (np.sin(2 * t) + 1)  # 무릎 굽힘 (항상 음수 쪽으로)
        
        theta1_dot = np.cos(2 * t)         
        theta2_dot = np.cos(2 * t + 1.0)   
        theta3_dot = -(np.cos(2 * t) + 1) 

        current_thetas = [theta1, theta2, theta3]
        current_thetas_dot = np.array([theta1_dot,theta2_dot,theta3_dot])
        Js = space_jacobian(twists,current_thetas)

        end_effector_vel = Js@current_thetas_dot
        
        # breakpoint()
        # ---------------------------------------------------------
        # A. MuJoCo 시뮬레이션 업데이트
        # ---------------------------------------------------------
        # FL
        data.qpos[0] = theta1 # FL_j1
        data.qpos[1] = theta2 # FL_j2
        data.qpos[2] = theta3 # FL_j3
        # # FR
        # data.qpos[3] = theta1 # FL_j1
        # data.qpos[4] = theta2 # FL_j2
        # data.qpos[5] = theta3 # FL_j3
        # # RL
        # data.qpos[6] = theta1 # FL_j1
        # data.qpos[7] = theta2 # FL_j2
        # data.qpos[8] = -theta3 # FL_j3
        # # RR
        # data.qpos[9] = theta1 # FL_j1
        # data.qpos[10] = theta2 # FL_j2
        # data.qpos[11] = -theta3 # FL_j3
        
        # 기구학 정보 갱신 (위치가 변했으므로 필수)
        mj.mj_kinematics(model, data) 
        
        # ---------------------------------------------------------
        # B. PoE 수식 계산 (Python 연산)
        # ---------------------------------------------------------
        T_poe = np.eye(4)
        # T = e^(S1*t1) * e^(S2*t2) * e^(S3*t3)
        for i in range(len(target_joints)):
            # 주의: vec_to_se3 함수가 T_poe 오른쪽에 곱해져야 함 (Body Frame이 아닌 Space Frame 방식이라면 순서 중요)
            # Lynch 교재의 Forward Kinematics (Space form): T = e^(S0*th0) ... * M
            # 위 반복문은 순서대로 곱해지므로 (T_poe = T_poe @ Next) -> T0 @ T1 @ T2 형태가 됨. 맞음.
            T_poe = T_poe @ vec_to_se3(twists[i], current_thetas[i])
            
        T_poe = T_poe @ M # 마지막에 Home Pose 곱하기
        pos_poe = T_poe[:3, 3]

        # ---------------------------------------------------------
        # C. 검증 및 오차 출력
        # ---------------------------------------------------------
        # MuJoCo 상의 실제 End-Effector 위치 (Base Frame 기준)
        base_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, base_name)
        ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, end_effector)
        
        p_base_w = data.xpos[base_id]
        R_base_w = data.xmat[base_id].reshape(3, 3)
        p_ee_w = data.xpos[ee_id]
        
        # World -> Base 변환
        pos_mujoco = R_base_w.T @ (p_ee_w - p_base_w)
        
        error = np.linalg.norm(pos_poe - pos_mujoco)
        
        # 터미널에 실시간 정보 출력 (f-string 사용)
        print(f"\rTime: {t:.2f}s | Error: {error:.6f} | PoE: {pos_poe} | MuJoCo: {pos_mujoco}", end="")
        
        # 뷰어 업데이트
        viewer.sync()
        
        # 속도 조절 (너무 빠르지 않게)
        time.sleep(model.opt.timestep)