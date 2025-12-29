import time
import mujoco
import mujoco.viewer
import numpy as np

# 위에 정의한 XML 문자열 (센서 포함됨)
xml = """
<mujoco>
  <worldbody>
    <light pos="0 0 1"/>
    <body name="link1" pos="0 0 0">
      <joint name="pin" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.05" rgba="0.8 0.2 0.2 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="pin" name="motor1" gear="10"/>
  </actuator>
  <sensor>
    <actuatorfrc name="torque_sensor" actuator="motor1"/>
  </sensor>
</mujoco>
"""

m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)

# Passive Viewer 실행
with mujoco.viewer.launch_passive(m, d) as viewer:
    start_time = time.time()
    
    # 팁: 뷰어 실행 시 바로 그래프가 보이게 하려면 설정 변경이 필요할 수 있으나,
    # 보통은 GUI에서 클릭으로 띄우는 것이 정석입니다.
    
    while viewer.is_running():
        step_start = time.time()

        # 제어 입력 (사인파 토크)
        d.ctrl[0] = 5 * np.sin(d.time * 2) 

        # 물리 연산 (이때 센서값도 자동 계산됨)
        mujoco.mj_step(m, d)

        # 뷰어 갱신
        viewer.sync()

        # 속도 조절
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)