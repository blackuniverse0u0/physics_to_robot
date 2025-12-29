import mujoco
import mujoco.viewer
import numpy as np
import time

# XML 모델 정의 (기존과 동일)
xml_string = """
<mujoco model="double_pendulum">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual>
    <headlight ambient="0.5 0.5 0.5"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0.2"/>
    <material name="arm" rgba="0.7 0.3 0.1 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1" material="grid"/>
    <camera name="fixed" pos="2 -2 1.5" xyaxes="0.7071 0.7071 0 -0.408 0.408 0.816"/>
    <body name="base" pos="0 0 0.5">
      <geom type="cylinder" size="0.05 0.1" rgba="0.3 0.3 0.3 1"/>
      <body name="link1" pos="0 0 0.12">
        <joint name="joint1" type="hinge" axis="0 1 0" range="-180 180" damping="0.5"/>
        <geom name="link1_geom" type="capsule" fromto="0 0 0 0 0 0.4" size="0.04" material="arm" mass="1.0"/>
        <body name="link2" pos="0 0 0.4">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-180 180" damping="0.5"/>
          <geom name="link2_geom" type="capsule" fromto="0 0 0 0 0 0.3" size="0.035" material="arm" mass="0.8"/>
          <site name="end_effector" pos="0 0 0.3" size="0.05" rgba="1 0 0 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor1" joint="joint1" gear="50" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="motor2" joint="joint2" gear="30" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
  <sensor>
    <jointpos name="joint1_pos" joint="joint1"/>
    <jointpos name="joint2_pos" joint="joint2"/>
    <jointvel name="joint1_vel" joint="joint1"/>
    <jointvel name="joint2_vel" joint="joint2"/>
    <actuatorfrc name="joint1_torque" actuator="motor1"/>
    <actuatorfrc name="joint2_torque" actuator="motor2"/>
  </sensor>
</mujoco>
"""
def create_figure():
    fig = mujoco.MjvFigure()
    fig.title = "Joint Data (Pos/Vel/Torque)"
    fig.xlabel = "Time (s)"
    fig.gridsize = [5, 5]
    fig.range[0] = [0, 10]  # X-axis
    fig.range[1] = [-10, 10] # Y-axis
    
    # 6개의 선 이름 설정
    legends = ["J1 Pos", "J2 Pos", "J1 Vel", "J2 Vel", "J1 Trq", "J2 Trq"]
    for i, name in enumerate(legends):
        # f-string이나 일반 문자열을 byte로 변환하지 않고 직접 대입하거나
        # 아래와 같이 초기화합니다.
        fig.linename[i] = name[:15] # 16자 제한이 있으므로 안전하게 슬라이싱
    
    # 색상 설정 (RGB)
    colors = [[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1], [0.5, 0.5, 1]]
    for i in range(len(colors)):
        fig.linergb[i] = colors[i]
        
    return fig

def main():
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    fig = create_figure()
    
    # 뷰포트 설정 (오른쪽 하단에 그래프 배치)
    viewport = mujoco.MjrRect(10, 10, 500, 300)

    # with mujoco.viewer.launch_passive(model, data) as viewer:
    with mujoco.viewer.launch(model, data) as viewer:
        # 카메라 설정
        viewer.cam.fixedcamid = 0
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        
        while viewer.is_running():
            step_start = time.time()

            # 제어기: 사인파 입력
            data.ctrl[0] = 0.5 * np.sin(2 * data.time)
            data.ctrl[1] = 0.3 * np.cos(3 * data.time)

            mujoco.mj_step(model, data)

            # 그래프 데이터 업데이트 (주기적으로)
            if data.time % 0.02 < model.opt.timestep:
                # 데이터 수집 (센서 순서: pos2, vel2, frc2)
                current_values = data.sensordata.copy()
                
                for i in range(6):
                    # MuJoCo Figure에 포인트 추가 (mju_addToFigure 사용이 안정적임)
                    # 여기선 직접 인덱싱을 사용하되 실시간 이동 효과 구현
                    n = fig.linepnt[i]
                    if n < 1000:
                        fig.linedata[i][2*n] = data.time
                        fig.linedata[i][2*n+1] = current_values[i]
                        fig.linepnt[i] += 1
                    else:
                        # 데이터 시프트 (고급 사용법 대신 간단한 덮어쓰기 로직)
                        # 실제로는 ring buffer를 쓰는 것이 좋으나 MjvFigure 제약상 이동 필요
                        fig.linedata[i][0:1998] = fig.linedata[i][2:2000]
                        fig.linedata[i][1998] = data.time
                        fig.linedata[i][1999] = current_values[i]

                # X축 범위 자동 조정
                if data.time > fig.range[0][1]:
                    fig.range[0][0] = data.time - 10
                    fig.range[0][1] = data.time

                # 그래프 및 텍스트 갱신
                viewer.update_hists() # 히스토리 업데이트
                
            # 뷰어 동기화 (그래프 포함)
            with viewer.lock():
                # 별도의 윈도우가 아닌 뷰어 내부 UI에 그래프를 그립니다.
                # viewer.user_scn.figs[0]을 직접 제어하기보다 custom_plots를 활용하거나 
                # 아래와 같이 직접 fig를 렌더링 리스트에 추가합니다.
                viewer.sync()
                # 렌더링 컨텍스트가 확보된 상태에서 그리기 (비공식 API 활용 주의)
                # 단순 확인을 위해 Text Overlay만 먼저 확실히 나오게 합니다.
                
            # 실시간 동기화
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
