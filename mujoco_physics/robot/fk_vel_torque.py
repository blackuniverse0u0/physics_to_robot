import mujoco as mj
import mujoco.viewer
import numpy as np
import time

# =========================================================
# 1. RobotMath: 로봇 수학 라이브러리 (Screw Theory)
# =========================================================
class RobotMath:
    @staticmethod
    def skew(v):
        """벡터를 왜곡 대칭 행렬(Skew-symmetric matrix)로 변환"""
        v = np.array(v).flatten()
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    @staticmethod
    def vec_to_se3(S, theta):
        """Screw Axis S와 각도 theta를 받아 변환 행렬 T 계산 (Rodrigues)"""
        w = S[:3]
        v = S[3:]
        
        # 회전이 없는 경우 (Prismatic)
        if np.linalg.norm(w) < 1e-6:
            T = np.eye(4)
            T[:3, 3] = v * theta
            return T
            
        # 회전이 있는 경우 (Revolute)
        skew_w = RobotMath.skew(w)
        I = np.eye(3)
        R = I + np.sin(theta) * skew_w + (1 - np.cos(theta)) * (skew_w @ skew_w)
        G = I * theta + (1 - np.cos(theta)) * skew_w + (theta - np.sin(theta)) * (skew_w @ skew_w)
        p = G @ v
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        return T

    @staticmethod
    def adjoint(T):
        """Adjoint Matrix (6x6) 변환: 속도/힘을 다른 좌표계로 보낼 때 사용"""
        R = T[:3, :3]
        p = T[:3, 3]
        adj = np.zeros((6, 6))
        adj[:3, :3] = R
        adj[3:, 3:] = R
        adj[3:, :3] = RobotMath.skew(p) @ R
        return adj

    @staticmethod
    def get_space_jacobian(S_list, thetas):
        """Space Jacobian J_s 계산"""
        S_mat = np.array(S_list)
        if S_mat.shape[0] != 6 and S_mat.shape[1] == 6:
            S_mat = S_mat.T
            
        n_joints = len(thetas)
        Js = np.zeros((6, n_joints))
        T = np.eye(4)
        Js[:, 0] = S_mat[:, 0]
        
        for i in range(1, n_joints):
            T = T @ RobotMath.vec_to_se3(S_mat[:, i-1], thetas[i-1])
            Js[:, i] = RobotMath.adjoint(T) @ S_mat[:, i]
        return Js

    @staticmethod
    def get_geometric_jacobian(J_space, p_tip_base):
        """Space Jacobian -> Geometric Jacobian 변환"""
        J_geo = np.zeros_like(J_space)
        J_w = J_space[:3, :]
        J_v = J_space[3:, :]
        
        # 선속도 부분 보정: v_tip = v_space - p x w
        J_geo[:3, :] = J_w
        J_geo[3:, :] = J_v - RobotMath.skew(p_tip_base) @ J_w
        return J_geo

# =========================================================
# 2. MuJoCo 파라미터 추출 함수
# =========================================================
def get_base_to_screw_params(model, data, base_name, joint_names, ee_body_name):
    # 초기 자세(0)에서 파라미터 추출
    data.qpos[:] = 0
    mj.mj_kinematics(model, data)
    
    base_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, base_name)
    R_wb = data.xmat[base_id].reshape(3, 3).copy()
    p_wb = data.xpos[base_id].copy()
    
    twists = []
    for name in joint_names:
        j_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        p_j_w = data.xanchor[j_id]
        axis_j_w = data.xaxis[j_id]
        
        # World 좌표계를 Base 좌표계로 변환
        omega = R_wb.T @ axis_j_w
        q = R_wb.T @ (p_j_w - p_wb)
        v = np.cross(-omega, q)
        S = np.concatenate((omega, v))
        twists.append(S)
        
    ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, ee_body_name)
    p_ee_w = data.xpos[ee_id]
    R_ee_w = data.xmat[ee_id].reshape(3, 3)
    
    # M: 초기 상태의 End-Effector 변환 행렬 (Base 기준)
    M = np.eye(4)
    M[:3, :3] = R_wb.T @ R_ee_w
    M[:3, 3] = R_wb.T @ (p_ee_w - p_wb)
    
    # return np.array(twists).T, M

    # [수정됨] Body 대신 Site 정보를 가져옵니다.
    site_name = 'FL_tip_site'
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
    p_ee_w = data.site_xpos[site_id]          # xpos -> site_xpos
    R_ee_w = data.site_xmat[site_id].reshape(3, 3) # xmat -> site_xmat
    
    M = np.eye(4)
    M[:3, :3] = R_wb.T @ R_ee_w
    M[:3, 3] = R_wb.T @ (p_ee_w - p_wb)
    
    return np.array(twists).T, M    

# =========================================================
# 3. 메인 시뮬레이션 및 검증
# =========================================================
xml_path = 'robot_pos_force_sensor.xml'
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# 설정값
base_name = 'base_link'
target_joints = ["FL_j1", "FL_j2", "FL_j3"] # 앞왼쪽 다리만 제어/검증
end_effector = "FL_calf"                    # 발 끝 링크 이름
site_name = "FL_tip_site"                   # 센서가 달린 Site 이름

# ID 및 센서 이름 매핑
dof_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n) for n in target_joints]
actuator_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, f"act_{n}") for n in target_joints]
tau_sensor_names = [f"sens_tau_{n}" for n in target_joints]
force_sensor_name = "sens_force_FL"

# Screw Parameter 추출
S_list, M = get_base_to_screw_params(model, data, base_name, target_joints, end_effector)

print("================================================================")
print(" MuJoCo Jacobian Verification (Force Sensor vs Torque Sensor)")
print("================================================================")

with mj.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    
    while viewer.is_running():
        t = time.time() - start_time
        
        # --- A. Trajectory (바닥을 꾹꾹 누르는 동작) ---
        # theta2, theta3를 사인파로 움직여 발이 땅에 닿았다 떨어지게 함
        theta2 = 0.8 + 0.3 * np.sin(2 * t) 
        theta3 = -1.6 
        q_target = np.array([0.0, theta2, theta3])
        
        # 제어 입력 (Position Control -> Torque 생성)
        data.ctrl[actuator_ids] = q_target
        mj.mj_step(model, data)
        
        # --- B. 센서 데이터 읽기 (Observation) ---
        q_meas = data.qpos[dof_ids] # 현재 관절 각도
        
        # 1. 토크 센서 값 (실제 부하)
        tau_meas = np.array([data.sensor(n).data[0] for n in tau_sensor_names])
        
        # 2. 힘 센서 값 (발 끝 Site 기준)
        f_sens_site = data.sensor(force_sensor_name).data # [fx, fy, fz]
        
        # --- C. 좌표계 변환 (Site Frame -> Base Frame) ---
        # Jacobian은 Base Frame 기준이므로, 센서값도 Base Frame으로 돌려야 함
        site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
        base_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, base_name)
        
        R_site_w = data.site_xmat[site_id].reshape(3, 3) # Site orientation in World
        R_base_w = data.xmat[base_id].reshape(3, 3)      # Base orientation in World
        
        # R_site_base = R_wb^T * R_ws
        R_site_base = R_base_w.T @ R_site_w
        f_sens_base = R_site_base @ f_sens_site # Base Frame에서의 외력
        
        # --- D. Jacobian 업데이트 ---
        T_poe = np.eye(4)
        for i in range(len(target_joints)):
            T_poe = T_poe @ RobotMath.vec_to_se3(S_list[:, i], q_meas[i])
        T_poe = T_poe @ M
        p_tip_base = T_poe[:3, 3]

        J_space = RobotMath.get_space_jacobian(S_list, q_meas)
        J_geo = RobotMath.get_geometric_jacobian(J_space, p_tip_base)
        
        # Linear Jacobian (3x3) - 힘(Linear Force)과 관계된 부분
        J_lin = J_geo[3:, :] 
        
        # --- E. 검증: Tau_calculated = J^T * F_sensor ---
        # 센서 힘(반작용력)이 주어졌을 때, 이를 버티기 위해 필요한 이론적 토크
        tau_from_force = J_lin.T @ f_sens_base
        
        # 오차(Diff) = 실제 토크 - 이론적 토크
        # 중력이 켜져 있다면 이 값은 '다리 무게를 버티는 토크'가 됩니다.
        diff = tau_meas - tau_from_force
        
        # --- 출력 ---
        if t > 0.1:
            print(f"\r[t={t:.2f}] "
                  f"F_z(Tip): {f_sens_base[2]:5.2f}N | "
                  f"Tau_meas: {np.array2string(tau_meas, precision=2, separator=',')} | "
                  f"Tau_calc(J.T*F): {np.array2string(tau_from_force, precision=2, separator=',')} | "
                  f"Diff(Gravity): {np.array2string(diff, precision=2, separator=',')}", end="")
            
        viewer.sync()
        time.sleep(model.opt.timestep)