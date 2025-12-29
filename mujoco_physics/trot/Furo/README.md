# 환경 파일(env/) 사용 가이드

## 개요
`env/` 디렉토리에는 로봇 테스트를 위한 다양한 환경 파일이 포함되어 있습니다.

## 환경 파일 종류

### 1. `environment.xml` (풀 환경)
다음 요소를 포함하는 복잡한 환경:
- **계단** (6단)
- **경사로** (20도, 30도)
- **미끄러운 지형** (마찰 계수 조절)
- **고정 장애물** (12개)
- **움직이는 장애물** (구체 1개, 상자 10개)
- **도킹 스테이션** (ArUco 마커 포함)

### 2. `door.xml` (간소화 환경)
다음 요소를 포함하는 간단한 환경:
- **계단** (6단)
- **경사로** (20도, 30도)
- **미끄러운 지형**
- **고정 장애물** (12개)
- **움직이는 장애물** (구체 1개, 상자 10개)

## 사용 방법

### 방법 1: scene.xml 수정
기존 `scene.xml`을 다음과 같이 수정:

```xml
<mujoco model="qugv scene">
  <include file="furo/qugv.xml"/>
  <include file="env/door.xml"/>  <!-- 또는 env/environment.xml -->
</mujoco>
```

### 방법 2: 새 scene 파일 생성
`scene_with_env.xml` 파일을 새로 생성:

```xml
<mujoco model="qugv with environment">
  <!-- 로봇 모델 포함 -->
  <include file="furo/qugv.xml"/>

  <!-- 환경 파일 포함 -->
  <include file="env/environment.xml"/>
</mujoco>
```

### 방법 3: Python 코드에서 직접 지정
`homing_pose.py` 파일에서 XML 경로만 변경:

```python
import mujoco
import mujoco.viewer

# 방법 3-1: 새 scene 파일 사용
xml_path = "scene_with_env.xml"

# 또는 방법 3-2: 환경 파일을 포함하도록 수정한 scene.xml 사용
xml_path = "scene.xml"

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Keyframe 'home' 불러오기
mujoco.mj_resetDataKeyframe(model, data, 0)

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("시뮬레이션 시작!")
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
```

## 주의사항

1. **로봇 모델 필수**: `env/` 파일들은 환경만 정의하므로, 로봇 모델(`furo/qugv.xml`)을 반드시 함께 include해야 합니다.

2. **파일 경로**: XML include 경로는 메인 XML 파일의 위치를 기준으로 합니다.

3. **충돌 설정**: `door.xml`에는 contact 페어가 정의되어 있지만, `environment.xml`에는 더 상세한 충돌 설정이 있습니다.

## 테스트 환경 선택 가이드

- **간단한 테스트**: `door.xml` 사용
- **복잡한 시나리오**: `environment.xml` 사용 (도킹 스테이션 포함)
- **커스텀 환경**: 위 파일들을 복사하여 수정
