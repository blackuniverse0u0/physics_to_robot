import mujoco
import mujoco.viewer
import numpy as np
import time

# XML 파일 경로 (사용자 코드에 맞춰 변경)
MODEL_XML_PATH = "furo_flat.xml"

def deg2rad(deg):
    return deg * (np.pi / 180.0)

# ==========================================
# 자세 정의 (Right-Handed)
# ==========================================

# [Up Pose: Walking Ready]
q_up = np.array([
    0, deg2rad(45),  deg2rad(-90),   # FL
    0, deg2rad(45),  deg2rad(-90),   # FR
    0, deg2rad(-45), deg2rad(90),    # RL
    0, deg2rad(-45), deg2rad(90)     # RR
])

# [Down Pose: Sitting]
q_down = np.array([
    0, deg2rad(85),  deg2rad(-158),  # FL
    0, deg2rad(85),  deg2rad(-158),  # FR
    0, deg2rad(-85), deg2rad(158),   # RL
    0, deg2rad(-85), deg2rad(158)    # RR
])


def main():
    m = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    d = mujoco.MjData(m)

    # ==========================================
    # 시간 및 속도 설정 (핵심 변경 사항)
    # ==========================================
    duration = 60.0  # 전체 실행 시간
    wait_time = 5.0  # 초기 대기 시간
    
    # [요청사항 반영]
    # 2초 동안 Up, 2초 동안 Down -> 총 주기 4초
    # Frequency = 1 / 4.0 = 0.25 Hz
    freq = 0.25      

    # 초기 위치 설정 (시작부터 Down 자세)
    d.qpos[7:] = q_down
    
    # Viewer 실행
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            # 현재 실제 경과 시간 측정
            now = time.time() - start_time
            if now > duration:
                break

            # 5초 대기 후 움직임 시작
            if now < wait_time:
                target_qpos = q_down
            else:
                t_move = now - wait_time
                
                # Cosine 보간
                # 주기 4초 (0.25Hz)일 때: 
                # t_move가 0~2초 구간에서는 alpha가 0->1 (일어남)
                # t_move가 2~4초 구간에서는 alpha가 1->0 (앉음)
                alpha = 0.5 * (1 - np.cos(2 * np.pi * freq * t_move))
                
                target_qpos = q_down + (q_up - q_down) * alpha

            # 제어 입력 인가
            d.ctrl[:] = target_qpos

            # 시뮬레이션 스텝 시작 시간 기록
            step_start = time.time()
            
            # 물리 연산 수행
            mujoco.mj_step(m, d)
            
            # 화면 업데이트
            viewer.sync()

            # [실제 시간 동기화 로직]
            # 시뮬레이션의 timestep(보통 0.001초 또는 0.002초)과
            # 실제 연산 시간의 차이만큼 대기하여 실제 시간 속도에 맞춤
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
