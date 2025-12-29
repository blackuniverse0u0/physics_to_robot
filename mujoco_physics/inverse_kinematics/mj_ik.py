import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 사용자 정의 모듈 (같은 폴더에 있어야 함)
from inverse_kinematics import inverse_kinematics

# 1. 설정 및 초기화
xml_path = 'universal_robots_ur5e/scene.xml'
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 초기 관절 각도 설정
q = np.array([-1.23, -1.5, 0.5, -1.5708, -1.5708, 0])
data.qpos = q
mujoco.mj_forward(model, data) # 초기 위치 반영

# 시뮬레이션 파라미터
simend = 3      # 종료 시간
t_init = 1      # 대기 시간
dt = 0.02       # 스텝 간격
r = 0.1         # 원 반지름
f = 1           # 원 주파수

# 목표 위치 초기값
x_ref, y_ref, z_ref = 0.5, 0.2, 0.5
x_ref, y_ref, z_ref = 0.6, 0.45, 0.6
phi_ref, theta_ref, psi_ref = 3.14, 0, 0
x0, y0 = x_ref - r, y_ref # 원의 중심점 계산용

# 데이터 저장용 리스트
logs = {'ref_x': [], 'ref_y': [], 'act_x': [], 'act_y': []}

print("시뮬레이션 시작...")

# 2. 뷰어 실행 (launch_passive: 코드로 제어권을 가짐)
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    # # 뷰어 설정 (카메라 등 필요하면 여기서 설정)
    # viewer.cam.azimuth = -130
    # viewer.cam.elevation = -15
    # viewer.cam.distance = 2
    # viewer.cam.lookat = np.array([0.0, 0.0, 0.5])
    
    while viewer.is_running() and data.time < simend:
        step_start = time.time()

        # 시간 업데이트
        data.time += dt

        # --- A. 궤적 생성 (원 그리기) ---
        if data.time >= t_init:
            current_t = data.time - t_init
            x_ref = x0 + r * np.cos(2 * np.pi * f * current_t)
            y_ref = y0 + r * np.sin(2 * np.pi * f * current_t)
        
        # --- B. 역기구학 (Inverse Kinematics) 풀기 ---
        # 목표 자세 벡터 [x, y, z, roll, pitch, yaw]
        target_pose = np.array([x_ref, y_ref, z_ref, phi_ref, theta_ref, psi_ref])
        
        # fsolve로 IK 해(q) 찾기
        q = fsolve(inverse_kinematics, q, args=(target_pose))
        
        # --- C. 로봇 상태 업데이트 ---
        data.qpos = q  # 계산된 각도를 로봇에 강제 주입
        mujoco.mj_forward(model, data) # 위치 갱신 (물리 연산 X)

        # --- D. 데이터 저장 ---
        if data.time > t_init:
            logs['ref_x'].append(x_ref)
            logs['ref_y'].append(y_ref)
            logs['act_x'].append(data.site('attachment_site').xpos[0])
            logs['act_y'].append(data.site('attachment_site').xpos[1])

        # --- E. 화면 갱신 ---
        viewer.sync()

        # 실제 시간과 속도 맞추기 (선택 사항)
        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


print("시뮬레이션 종료. 그래프 생성 중...")

# Matplotlib 설정 (mjpython 사용 시 GUI 충돌 방지용 백엔드 설정)
import matplotlib
matplotlib.use('Agg') # 화면에 창을 띄우지 않고 파일로만 저장하는 모드
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.plot(logs['ref_x'], logs['ref_y'], 'k--', linewidth=2, label='Reference')
plt.plot(logs['act_x'], logs['act_y'], 'r-', linewidth=1.5, label='Actual')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('End-effector Trajectory')
plt.axis('equal')
plt.legend()
plt.grid(True)

# plt.show()  <-- mjpython에서는 여전히 위험할 수 있음
plt.savefig('result_graph.png') 
print("그래프가 'result_graph.png'로 저장되었습니다.")