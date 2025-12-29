import mujoco 
import numpy as np
import time 

# XML 파일 로드 (만드신 파일명으로 변경)
model = mujoco.MjModel.from_xml_path('robot_torque.xml')
data = mujoco.MjData(model)

# 1. 시뮬레이션 한 스텝 진행 (초기화)
mujoco.mj_step(model, data)

# 2. Dynamics 관련 행렬 추출
# nv = 자유도 개수 (Base 6 + Joints 12 = 18)
nv = model.nv 

# Mass Matrix M(q) : (nv x nv)
M = np.zeros((nv, nv))
mujoco.mj_fullM(model, M, data.qM) # data.qM은 희소행렬 형태이므로 Dense로 변환

# Bias Forces (Coriolis + Gravity): C(q, q_dot) + G(q)
bias = data.qfrc_bias # (nv,)

print(f"Total DoF (nv): {nv}")
print(f"Mass Matrix Shape: {M.shape}")
print(f"Bias Force Shape: {bias.shape}")

# 예시: Base Link(몸체)의 질량이 5.0kg 근처인지 M행렬의 첫 요소로 확인
# (Floating base의 경우 M[0,0], M[1,1], M[2,2]는 전체 시스템 질량과 관련됨)
print(f"System effective mass at root (approx): {M[0,0]:.2f} kg") 
# -> 15.0kg (Base 5 + Legs 10)가 나와야 정상

# MuJoCo 뷰어 실행
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    # [수정 1, 2] if 대신 while 사용, 함수 호출 () 추가
    while viewer.is_running(): 
        
        step_start = time.time() # 루프 시작 시간 기록

        # 현재 상태 읽기 (필요 시)
        # 주의: data.qpos는 참조(reference)이므로 값 저장이 필요하면 .copy()를 써야 함
        q = data.qpos 
        qvel = data.qvel 
        
        # 물리 연산 1스텝 진행
        mujoco.mj_step(model, data)

        # 뷰어 화면 업데이트
        viewer.sync()

        # [선택 사항] 시뮬레이션 속도를 실제 시간과 비슷하게 맞추기
        # 이 코드가 없으면 컴퓨터 성능에 따라 너무 빨리 재생될 수 있습니다.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)