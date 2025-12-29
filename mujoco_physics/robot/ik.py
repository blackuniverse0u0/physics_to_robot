import mujoco as mj
import mujoco.viewer
import numpy as np
import time

# 사용자 라이브러리 (my_robotics.py)
import modern_robotics as mr

# ==============================================================================
# 1. 헬퍼 함수 (Skew, FK, 좌표 변환)
# ==============================================================================
def calculate_fk(S_list, M, q):
    """FK: Joint 각도를 받아 End-Effector의 Pose(T) 계산"""
    T = np.eye(4)
    for i in range(len(q)):
        T = T @ mr.vec_to_se3(S_list[i], q[i])
    return T @ M

def transform_world_to_base(model, data, base_name, p_world_pt):
    """검증용: World 좌표계의 점을 Base 좌표계로 변환"""
    base_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, base_name)
    p_base_w = data.xpos[base_id]           
    R_base_w = data.xmat[base_id].reshape(3, 3) 
    # R_wb^T * (p_world - p_base)
    return R_base_w.T @ (p_world_pt - p_base_w)

def get_leg_params_with_offset(model, data, base_name, leg_prefix, ee_suffix="_calf"):
    """
    초기 파라미터 추출
    [중요] MuJoCo의 'calf' body는 무릎 위치입니다. 
        따라서 Z축으로 -0.2m (XML 형상 기준) 내려간 곳을 실제 발끝(EE)으로 설정합니다.
    """
    joint_names = [f"{leg_prefix}_j{i}" for i in range(1, 4)]
    ee_body_name = f"{leg_prefix}{ee_suffix}"

    # 1. Kinematics 초기화 (Home Pose)
    data.qpos[:] = 0
    mj.mj_kinematics(model, data) 
    
    base_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, base_name) 
    R_wb = data.xmat[base_id].reshape(3, 3).copy()
    p_wb = data.xpos[base_id].copy()
    
    twists = []
    
    # 2. Screw Axis 추출
    for name in joint_names:
        j_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        p_j_w = data.xanchor[j_id] 
        axis_j_w = data.xaxis[j_id] 
        
        # Base Frame으로 변환
        omega = R_wb.T @ axis_j_w
        q_vec = R_wb.T @ (p_j_w - p_wb)
        v = np.cross(-omega, q_vec)
        S = np.concatenate((omega, v))
        twists.append(S)
        
    # 3. M Matrix (Home Pose of FOOT) 계산
    ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, ee_body_name)
    p_knee_w = data.xpos[ee_id] # 이것은 무릎 위치
    R_knee_w = data.xmat[ee_id].reshape(3, 3)
    
    # [XML 기준] Calf 길이 0.2m 반영 (발끝 위치 계산)
    # Global 좌표계에서의 발끝 위치 = 무릎 위치 + R_knee * offset_local
    foot_offset_local = np.array([0, 0, -0.2]) 
    p_foot_w = p_knee_w + R_knee_w @ foot_offset_local
    
    M = np.eye(4)
    M[:3, :3] = R_wb.T @ R_knee_w 
    M[:3, 3] = R_wb.T @ (p_foot_w - p_wb) 
    
    # 4. 링크 길이 추출 (Analytical IK용)
    # L1: Hip~Thigh (Side offset), L2: Thigh, L3: Calf
    # p_j1, p_j2는 조인트 앵커 위치 사용
    j1_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_names[0])
    j2_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_names[1])
    j3_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_names[2])
    
    p_j1 = R_wb.T @ (data.xanchor[j1_id] - p_wb)
    p_j2 = R_wb.T @ (data.xanchor[j2_id] - p_wb)
    p_j3 = R_wb.T @ (data.xanchor[j3_id] - p_wb)
    p_foot = M[:3, 3]
    
    l1_side = abs(p_j2[1] - p_j1[1]) 
    l2 = np.linalg.norm(p_j3 - p_j2)
    l3 = np.linalg.norm(p_foot - p_j3) # 이제 p_foot을 쓰므로 0.2m가 나옴
    
    return twists, M, (l1_side, l2, l3)

# ==============================================================================
# 2. IK Solver (수정됨: 발끝 제어 & 무릎 방향)
# ==============================================================================
def solve_numerical_ik(S_list, M, current_q, target_pos, max_iter=5, tol=1e-2):
    """
    [Numerical] Space Jacobian을 발끝 속도(Linear Velocity of Tip)로 변환하여 사용
    """
    q = np.array(current_q, dtype=float).copy()
    success = False
    
    for _ in range(max_iter):
        T = calculate_fk(S_list, M, q)
        current_pos = T[:3, 3]
        err = target_pos - current_pos
        
        if np.linalg.norm(err) < tol:
            success = True
            break
            
        Js = mr.space_jacobian(S_list, q)
        J_w = Js[:3, :]
        J_v_origin = Js[3:, :] # 원점 기준 선형 속도
        
        # [핵심] Tip Velocity Jacobian으로 변환
        # v_tip = v_origin + w x p_tip
        p_skew = mr.skew(current_pos)
        J_pos = J_v_origin - p_skew @ J_w 
        
        # Damped Least Squares
        lamb = 1e-4
        J_inv = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + lamb * np.eye(3))
        
        q_delta = J_inv @ err
        q += q_delta
        
    return q, success

def solve_analytical_ik(target_pos, lengths, leg_sign, knee_dir=1):
    """
    [Analytical]
    leg_sign: 1 (Left), -1 (Right) -> Y축 방향 결정
    knee_dir: 1 (Knee forward), -1 (Knee backward/inward) -> q3 부호 결정
    """
    x, y, z = target_pos
    l1, l2, l3 = lengths
    
    # 작업 공간 검사
    dist_origin = np.linalg.norm(target_pos)
    if dist_origin > (l1 + l2 + l3) * 1.05: return np.zeros(3), False

    try:
        # 1. Hip Roll (q1)
        h_sq = y**2 + z**2 - l1**2
        if h_sq < 0: return np.zeros(3), False
        h = np.sqrt(h_sq)
        
        phi = np.arctan2(y, -z)
        psi = np.arctan2(l1, h)
        q1 = (phi - psi) if leg_sign > 0 else (phi + psi)

        # 2. Hip Pitch (q2) & Knee Pitch (q3)
        # 회전된 프레임에서의 x, z (h는 회전된 평면상의 높이)
        r_sq = x**2 + h**2
        r = np.sqrt(r_sq)
        
        # 코사인 제2법칙 (무릎)
        cos_q3 = (l2**2 + l3**2 - r_sq) / (2 * l2 * l3)
        if abs(cos_q3) > 1.0: return np.zeros(3), False
        
        # [XML 반영] 무릎 방향 처리
        # XML 설정상 FR/FL은 -3.14~0 (음수), RR/RL은 0~3.14 (양수)일 가능성 큼
        if knee_dir == -1: # Front legs (Knee bends backward in negative q)
             q3 = -np.arccos(cos_q3)
        else: # Rear legs (Knee bends forward/inward in positive q)
             q3 = np.arccos(cos_q3)

        # 힙 피치
        phi_leg = np.arctan2(-x, h)
        cos_alpha = (l2**2 + r_sq - l3**2) / (2 * l2 * r)
        if abs(cos_alpha) > 1.0: return np.zeros(3), False
        alpha = np.arccos(cos_alpha)
        
        q2 = phi_leg + alpha
        
        return np.array([q1, q2, q3]), True
        
    except Exception:
        return np.zeros(3), False

# ==============================================================================
# 3. 메인 시뮬레이션
# ==============================================================================

xml_path = 'robot_pos.xml'  # [수정됨] 제공된 XML 파일명
try:
    model = mj.MjModel.from_xml_path(xml_path)
except:
    print(f"Error: '{xml_path}' not found. Please save the provided XML code.")
    exit()

data = mj.MjData(model)
base_name = 'base_link'

# 시뮬레이션 속도 조절
sim_speed = 0.3 # 0.3배속 (매우 천천히)

# [설정] XML의 joint limit을 고려하여 무릎 방향(knee_dir) 설정
# knee_dir: -1 (Front legs, range -3.14~0), 1 (Rear legs, range 0~3.14)
leg_config = {
    "FL": {"idx": [0, 1, 2], "sign": 1,  "ik": "numerical",  "plane": "xy"},
    "FR": {"idx": [3, 4, 5], "sign": -1, "ik": "numerical",  "plane": "xy"},
    "RL": {"idx": [6, 7, 8], "sign": 1,  "ik": "analytical", "plane": "xz", "knee_dir": 1},
    "RR": {"idx": [9, 10, 11],"sign": -1,"ik": "analytical", "plane": "xz", "knee_dir": 1},
}
leg_config = {
    "FL": {"idx": [0, 1, 2], "sign": 1,  "ik": "numerical",  "plane": "xy"},
    "FR": {"idx": [3, 4, 5], "sign": -1, "ik": "numerical",  "plane": "xy"},
    "RL": {"idx": [6, 7, 8], "sign": 1,  "ik": "numerical", "plane": "xz", "knee_dir": 1},
    "RR": {"idx": [9, 10, 11],"sign": -1,"ik": "numerical", "plane": "xz", "knee_dir": 1},
}

robot_params = {}
for leg, cfg in leg_config.items():
    # 이제 무릎(calf)이 아니라 실제 발끝 위치를 기준으로 파라미터를 가져옵니다.
    S, M, lens = get_leg_params_with_offset(model, data, base_name, leg, ee_suffix="_calf")
    ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f"{leg}_calf")
    
    robot_params[leg] = {
        "S": S, "M": M, "lengths": lens, 
        "q_curr": np.zeros(3), 
        "ee_body_id": ee_id
    }

print("Simulating... Press ESC to exit.")
print("Goal: Front Legs -> Circle (XY), Rear Legs -> Circle (XZ)")

with mj.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    
    while viewer.is_running():
        # 시간 업데이트 (슬로우 모션)
        t = (time.time() - start_time) * sim_speed
        
        radius = 0.05
        freq = 2.0 
        
        print("\033[H\033[J") # 화면 클리어
        print(f"Time (Sim): {t:.2f}s | Speed: x{sim_speed}")
        print("-" * 75)
        print(f"{'Leg':<4} | {'Method':<10} | {'Valid':<5} | {'Error (m)':<10} | {'Status'}")
        print("-" * 75)

        for leg, cfg in leg_config.items():
            params = robot_params[leg]
            M, S = params["M"], params["S"]
            
            # 1. 목표 궤적 생성 (발끝 기준)
            p0 = M[:3, 3] # 초기 발끝 위치
            target_pos = p0.copy()
            
            if cfg["plane"] == "xy":
                # 바닥 닦기 동작
                target_pos[0] += radius * np.cos(freq * t)
                target_pos[1] += radius * np.sin(freq * t)
            elif cfg["plane"] == "xz":
                # 걷기 동작 (수직 원)
                target_pos[0] += radius * np.cos(freq * t)
                target_pos[2] += radius * np.sin(freq * t)
            
            # 2. IK 풀이
            if cfg["ik"] == "numerical":
                q_sol, is_valid = solve_numerical_ik(S, M, params["q_curr"], target_pos)
            else:
                q_sol, is_valid = solve_analytical_ik(
                    target_pos, params["lengths"], cfg["sign"], cfg.get("knee_dir", 1)
                )

            # 3. MuJoCo 적용
            if is_valid:
                idxs = cfg["idx"]
                data.qpos[idxs[0]], data.qpos[idxs[1]], data.qpos[idxs[2]] = q_sol
                params["q_curr"] = q_sol 
            
            # 4. 검증 (Validation)
            # 실제 발끝 위치 = Knee Body Pos + Rotation * Offset(0,0,-0.2)
            knee_pos_w = data.xpos[params["ee_body_id"]]
            knee_rot_w = data.xmat[params["ee_body_id"]].reshape(3, 3)
            real_foot_w = knee_pos_w + knee_rot_w @ np.array([0, 0, -0.2])
            
            real_foot_base = transform_world_to_base(model, data, base_name, real_foot_w)
            
            final_error = np.linalg.norm(target_pos - real_foot_base)
            
            # 출력
            status_str = "OK" if is_valid and final_error < 0.02 else "BAD"
            print(f"{leg:<4} | {cfg['ik']:<10} | {str(is_valid):<5} | {final_error:.6f}   | {status_str}")

        mj.mj_kinematics(model, data) 
        viewer.sync()
        # time.sleep(model.opt.timestep)