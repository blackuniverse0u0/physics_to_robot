import mujoco
import mujoco.viewer
import numpy as np

xml_path = "scene_flat.xml"
xml_path = "furo_flat.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# --- [중요] 시작할 때 Keyframe 'home' 불러오기 ---
# XML에 정의된 첫 번째(ID 0) 키프레임의 qpos를 현재 데이터로 복사합니다.
mujoco.mj_resetDataKeyframe(model, data, 0)

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("시뮬레이션 시작! (Ctrl+C로 종료)")
    while viewer.is_running():
        # 만약 "제자리 서기"를 테스트하고 싶다면 액션을 주지 않거나(0),
        # PD 제어기의 목표 위치를 현재 자세로 유지해야 합니다.
        
        # 예: 현재 자세 유지 (Gravity Compensation이 없으므로 서서히 주저앉을 수 있음)
        # data.ctrl[:] = data.qpos[7:] # 현재 관절 각도를 목표값으로 설정 (간이 유지)
        
        mujoco.mj_step(model, data)
        viewer.sync()
