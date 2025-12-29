import mujoco as mj
import mujoco.viewer
import numpy as np
import time

# [중요] GUI 충돌 방지를 위해 Agg 백엔드 사용 (창 안 띄우고 저장만 함)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# 사용자 라이브러리 (또는 modern_robotics)
import modern_robotics as mr

# ==============================================================================
# 1. 헬퍼 함수
# ==============================================================================
def calculate_fk(S_list, M, q):
    T = np.eye(4)
    for i in range(len(q)):
        T = T @ mr.vec_to_se3(S_list[i], q[i])
    return T @ M

def transform_world_to_base(model, data, base_name, p_world_pt):
    base_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, base_name)
    p_base_w = data.xpos[base_id]           
    R_base_w = data.xmat[base_id].reshape(3, 3) 
    return R_base_w.T @ (p_world_pt - p_base_w)

def get_leg_params_with_offset(model, data, base_name, leg_prefix, ee_suffix="_calf"):
    joint_names = [f"{leg_prefix}_j{i}" for i in range(1, 4)]
    ee_body_name = f"{leg_prefix}{ee_suffix}"

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
        
        omega = R_wb.T @ axis_j_w
        q_vec = R_wb.T @ (p_j_w - p_wb)
        v = np.cross(-omega, q_vec)
        S = np.concatenate((omega, v))
        twists.append(S)
        
    ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, ee_body_name)
    p_knee_w = data.xpos[ee_id]
    R_knee_w = data.xmat[ee_id].reshape(3, 3)
    
    # Offset: Knee to Foot (-0.2m in Z)
    foot_offset_local = np.array([0, 0, -0.2]) 
    p_foot_w = p_knee_w + R_knee_w @ foot_offset_local
    
    M = np.eye(4)
    M[:3, :3] = R_wb.T @ R_knee_w 
    M[:3, 3] = R_wb.T @ (p_foot_w - p_wb) 
    
    # Link lengths
    j1_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_names[0])
    j2_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_names[1])
    j3_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_names[2])
    
    p_j1 = R_wb.T @ (data.xanchor[j1_id] - p_wb)
    p_j2 = R_wb.T @ (data.xanchor[j2_id] - p_wb)
    p_j3 = R_wb.T @ (data.xanchor[j3_id] - p_wb)
    p_foot = M[:3, 3]
    
    l1_side = abs(p_j2[1] - p_j1[1]) 
    l2 = np.linalg.norm(p_j3 - p_j2)
    l3 = np.linalg.norm(p_foot - p_j3)
    
    return twists, M, (l1_side, l2, l3)

# ==============================================================================
# 2. IK Solver
# ==============================================================================
def solve_numerical_ik(S_list, M, current_q, target_pos, max_iter=5, tol=1e-2):
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
        J_v_origin = Js[3:, :]
        
        # Tip Velocity Jacobian
        p_skew = mr.skew(current_pos) # 로컬 함수 사용
        J_pos = J_v_origin - p_skew @ J_w 
        
        # Damped Least Squares (DLS)
        # Damped Pseudo-Inverse
        # 특이점(Singularity) 근처에서 로봇이 폭주하는 것을 막기 위해 사용
        # $$Minimize || J \Delta q - \Delta x ||^2 + \lambda^2 || \Delta q ||^2$$
        # $$\Delta q = J^T (J J^T + \lambda^2 I)^{-1} \Delta x$$
        lamb = 1e-4
        J_inv = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + lamb * np.eye(3))
        
        q_delta = J_inv @ err
        q += q_delta
        
    return q, success

def solve_analytical_ik(target_pos, lengths, leg_sign, knee_dir=1):
    x, y, z = target_pos
    l1, l2, l3 = lengths
    
    dist_origin = np.linalg.norm(target_pos)
    if dist_origin > (l1 + l2 + l3) * 1.05: return np.zeros(3), False

    try:
        h_sq = y**2 + z**2 - l1**2
        if h_sq < 0: return np.zeros(3), False
        h = np.sqrt(h_sq)
        
        phi = np.arctan2(y, -z)
        psi = np.arctan2(l1, h)
        q1 = (phi - psi) if leg_sign > 0 else (phi + psi)

        r_sq = x**2 + h**2
        r = np.sqrt(r_sq)
        
        cos_q3 = (l2**2 + l3**2 - r_sq) / (2 * l2 * l3)
        if abs(cos_q3) > 1.0: return np.zeros(3), False
        
        if knee_dir == -1: 
             q3 = -np.arccos(cos_q3)
        else: 
             q3 = np.arccos(cos_q3)

        phi_leg = np.arctan2(-x, h)
        cos_alpha = (l2**2 + r_sq - l3**2) / (2 * l2 * r)
        if abs(cos_alpha) > 1.0: return np.zeros(3), False
        alpha = np.arccos(cos_alpha)
        
        q2 = phi_leg + alpha
        
        return np.array([q1, q2, q3]), True
        
    except Exception:
        return np.zeros(3), False

# ==============================================================================
# 3. 메인 실행 (데이터 수집 -> 플로팅)
# ==============================================================================

if __name__ == "__main__":
    xml_path = 'robot_pos.xml' # or 'robot_pos.xml'
    try:
        model = mj.MjModel.from_xml_path(xml_path)
    except:
        print(f"Error: '{xml_path}' not found.")
        exit()

    data = mj.MjData(model)
    base_name = 'base_link'

    # 시뮬레이션 설정
    sim_speed = 1.0 
    simulation_duration = 5.0

    leg_config = {
        "FL": {"idx": [0, 1, 2], "sign": 1,  "ik": "numerical",  "plane": "xy"},
        "FR": {"idx": [3, 4, 5], "sign": -1, "ik": "numerical",  "plane": "xy"},
        "RL": {"idx": [6, 7, 8], "sign": 1,  "ik": "numerical", "plane": "xz", "knee_dir": 1},
        "RR": {"idx": [9, 10, 11],"sign": -1,"ik": "numerical", "plane": "xz", "knee_dir": 1},
    }
    # leg_config = {
    #     "FL": {"idx": [0, 1, 2], "sign": 1,  "ik": "analytical",  "plane": "xy"},
    #     "FR": {"idx": [3, 4, 5], "sign": -1, "ik": "analytical",  "plane": "xy"},
    #     "RL": {"idx": [6, 7, 8], "sign": 1,  "ik": "analytical", "plane": "xz", "knee_dir": 1},
    #     "RR": {"idx": [9, 10, 11],"sign": -1,"ik": "analytical", "plane": "xz", "knee_dir": 1},
    # }    
    

    robot_params = {}
    for leg, cfg in leg_config.items():
        S, M, lens = get_leg_params_with_offset(model, data, base_name, leg, ee_suffix="_calf")
        ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f"{leg}_calf")
        robot_params[leg] = {
            "S": S, "M": M, "lengths": lens, 
            "q_curr": np.zeros(3), 
            "ee_body_id": ee_id
        }

    log_data = {leg: {"target": [], "actual": []} for leg in leg_config}

    print(f"Simulating for {simulation_duration} seconds...")

    # [중요] viewer 컨텍스트 안에서 데이터 수집 완료 후 빠져나옴
    with mj.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            current_time = time.time() - start_time
            t = current_time * sim_speed
            
            # 시간이 되면 루프 탈출 -> viewer 종료됨
            if t > simulation_duration:
                break

            radius = 0.05
            freq = 2.0 
            
            # 진행상황
            if int(t * 10) % 5 == 0:
                print(f"\rTime: {t:.2f} / {simulation_duration:.2f} s", end="")

            for leg, cfg in leg_config.items():
                params = robot_params[leg]
                M, S = params["M"], params["S"]
                
                # 1. Target
                p0 = M[:3, 3] 
                target_pos = p0.copy()
                target_pos[2] = target_pos[2]+ 0.1 # 발끝보다 위에 있어야 풀수있음.
                
                if cfg["plane"] == "xy":
                    target_pos[0] += radius * np.cos(freq * t)
                    target_pos[1] += radius * np.sin(freq * t)
                elif cfg["plane"] == "xz":
                    target_pos[0] += radius * np.cos(freq * t)
                    target_pos[2] += radius * np.sin(freq * t)
                
                # 2. IK
                if cfg["ik"] == "numerical":
                    q_sol, is_valid = solve_numerical_ik(S, M, params["q_curr"], target_pos)
                else:
                    q_sol, is_valid = solve_analytical_ik(
                        target_pos, params["lengths"], cfg["sign"], cfg.get("knee_dir", 1)
                    )

                # 3. Apply
                if is_valid:
                    idxs = cfg["idx"]
                    data.qpos[idxs[0]], data.qpos[idxs[1]], data.qpos[idxs[2]] = q_sol
                    params["q_curr"] = q_sol 
                
                # 4. Log Actual
                knee_pos_w = data.xpos[params["ee_body_id"]]
                knee_rot_w = data.xmat[params["ee_body_id"]].reshape(3, 3)
                real_foot_w = knee_pos_w + knee_rot_w @ np.array([0, 0, -0.2])
                real_foot_base = transform_world_to_base(model, data, base_name, real_foot_w)
                
                log_data[leg]["target"].append(target_pos)
                log_data[leg]["actual"].append(real_foot_base)

            mj.mj_kinematics(model, data) 
            viewer.sync()
            time.sleep(model.opt.timestep)

    # ==========================================================================
    # 4. 그래프 저장 (Viewer 종료 후 실행)
    # ==========================================================================
    print("\nSimulation finished. Generating plot...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, leg in enumerate(leg_config.keys()):
        ax = axes[i]
        targets = np.array(log_data[leg]["target"])
        actuals = np.array(log_data[leg]["actual"])
        
        if len(targets) == 0: continue

        plane = leg_config[leg]["plane"]
        
        if plane == "xy":
            
            ax.plot(targets[:, 0], targets[:, 1], 'r--', label='Target')
            ax.plot(actuals[:, 0], actuals[:, 1], 'b-', alpha=0.7, label='Actual')
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title(f"{leg} Leg (XY Plane - Numerical)")
            ax.axis('equal')
        else:
            
            ax.plot(targets[:, 0], targets[:, 2], 'r--', label='Target')
            ax.plot(actuals[:, 0], actuals[:, 2], 'b-', alpha=0.7, label='Actual')
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Z (m)")
            ax.set_title(f"{leg} Leg (XZ Plane - Analytical)")
            ax.axis('equal')
            
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    filename = "trajectory_result.png"
    plt.savefig(filename)
    print(f"Saved plot to '{filename}'")