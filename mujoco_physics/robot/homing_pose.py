import mujoco as mj
import numpy as np
from utils import skew,vec_to_se3 

# --- 2. MuJoCo Setup & Parameter Extraction ---
xml_path = 'robot.xml'
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# 초기화: Homing Pose
data.qpos[:] = 0
mj.mj_kinematics(model, data)



import mujoco as mj 
import time

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