import mujoco
import mujoco.viewer
import numpy as np
import time
import multiprocessing  # 멀티프로세싱 모듈
from visualizer import launch_plotter # 함수 임포트

# XML 파일 경로
MODEL_XML_PATH = "furo_flat.xml"

def deg2rad(deg):
    return deg * (np.pi / 180.0)

# 자세 데이터
q_up = np.array([
    0, deg2rad(45),  deg2rad(-90),   # FL
    0, deg2rad(45),  deg2rad(-90),   # FR
    0, deg2rad(-45), deg2rad(90),    # RL
    0, deg2rad(-45), deg2rad(90)     # RR
])

q_down = np.array([
    0, deg2rad(85),  deg2rad(-158),  # FL
    0, deg2rad(85),  deg2rad(-158),  # FR
    0, deg2rad(-85), deg2rad(158),   # RL
    0, deg2rad(-85), deg2rad(158)    # RR
])

def main():
    m = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    d = mujoco.MjData(m)

    # ---------------------------------------------------------
    # 1. 멀티프로세싱 설정
    # ---------------------------------------------------------
    # 데이터 전송을 위한 큐 생성
    data_queue = multiprocessing.Queue()
    
    # 그래프 프로세스 생성 및 시작
    plot_process = multiprocessing.Process(target=launch_plotter, args=(data_queue,))
    plot_process.start()

    # 시뮬레이션 설정
    duration = 60.0
    wait_time = 5.0
    freq = 0.25      

    d.qpos[7:] = q_down
    
    try:
        # mjpython을 위해 launch_passive 사용
        with mujoco.viewer.launch_passive(m, d) as viewer:
            start_time = time.time()
            
            while viewer.is_running():
                now_real = time.time()
                sim_time = now_real - start_time

                if sim_time > duration:
                    break

                if sim_time < wait_time:
                    target_qpos = q_down
                else:
                    t_move = sim_time - wait_time
                    alpha = 0.5 * (1 - np.cos(2 * np.pi * freq * t_move))
                    target_qpos = q_down + (q_up - q_down) * alpha

                d.ctrl[:] = target_qpos

                step_start = time.time()
                mujoco.mj_step(m, d)
                viewer.sync()
                
                # -----------------------------------------------------
                # 2. 데이터 전송 (큐에 넣기)
                # -----------------------------------------------------
                # FL Leg 데이터 (0, 1, 2번 인덱스)
                fl_pos = d.qpos[7:10].copy() # copy() 중요
                fl_vel = d.qvel[6:9].copy()
                fl_trq = d.actuator_force[0:3].copy()
                
                # 큐가 꽉 차면 예전 데이터는 무시 (지연 방지)
                if not data_queue.full():
                    data_queue.put((sim_time, fl_pos, fl_vel, fl_trq))

                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                    
    except KeyboardInterrupt:
        pass
    finally:
        # 종료 시 그래프 프로세스도 정리
        data_queue.put(None)
        plot_process.join()
        print("Simulation Ended.")

if __name__ == "__main__":
    # macOS 멀티프로세싱 안전장치
    multiprocessing.set_start_method('spawn', force=True)
    main()
