# QUGV XML 업그레이드 가이드

qugv.xml을 rbq.xml처럼 자세하게 업데이트하는 방법에 대한 가이드입니다.

## 1. XML 선언 추가

**현재 (qugv.xml):**
```xml
<mujoco model="krm_qugv">
```

**업그레이드 (rbq.xml 스타일):**
```xml
<?xml version="1.0" encoding="utf-8"?>
<mujoco model="krm_qugv">
```

**설명:** XML 파일의 표준 선언을 추가하여 버전과 인코딩을 명시합니다.

---

## 2. Compiler 옵션 개선

**현재:**
```xml
<compiler angle="radian" autolimits="true"/>
```

**업그레이드:**
```xml
<compiler
    angle="radian"
    coordinate="local"
    autolimits="true"
/>
```

**설명:** `coordinate="local"` 옵션을 추가하여 좌표계를 명시적으로 지정합니다.

---

## 3. Option 태그 상세화

**현재:**
```xml
<option timestep="0.002" iterations="50" tolerance="1e-10" solver="Newton" gravity="0 0 -9.81" cone="elliptic" impratio="100"/>
```

**업그레이드:**
```xml
<option
    timestep="0.002"
    gravity="0 0 -9.81"
    integrator="RK4"
    cone="elliptic"
    impratio="100"
    tolerance="1e-10">
    <flag warmstart="enable" energy="enable" contact="enable" frictionloss="enable"/>
</option>
```

**설명:**
- `integrator="RK4"` 추가: 적분 방법 명시 (Runge-Kutta 4차)
- `<flag>` 태그 추가: 물리 시뮬레이션 옵션 세부 설정
  - `warmstart`: 초기값 재사용으로 성능 향상
  - `energy`: 에너지 보존 계산 활성화
  - `contact`: 접촉 감지 활성화
  - `frictionloss`: 마찰 손실 계산 활성화

---

## 4. Default 클래스 구조 재설계

**현재:**
```xml
<default>
  <default class="krm">
    <joint damping="0.5" frictionloss="0.1" armature="0.01"/>
    <geom type="mesh" contype="0" conaffinity="0" group="2" rgba="1 1 1 1"/>

    <default class="visual">...</default>
    <default class="collision">...</default>
    <default class="foot">...</default>
  </default>
</default>
```

**업그레이드:**
```xml
<default>
  <default class="krm">
    <geom friction="0.8 0.02 0.01" margin="0.001" condim="3"/>

    <default class="abduction">
      <default class="left_abduction">
        <joint axis="1 0 0" range="-0.785 0.785" damping="0.5" armature="0.01" frictionloss="0.1"/>
        <motor ctrlrange="-50 50" ctrllimited="true"/>
      </default>
      <default class="right_abduction">
        <joint axis="-1 0 0" range="-0.785 0.785" damping="0.5" armature="0.01" frictionloss="0.1"/>
        <motor ctrlrange="-50 50" ctrllimited="true"/>
      </default>
    </default>

    <default class="hip">
      <joint axis="0 1 0" range="-2.356 2.356" damping="0.5" armature="0.01" frictionloss="0.1"/>
      <motor ctrlrange="-50 50" ctrllimited="true"/>
    </default>

    <default class="knee">
      <joint axis="0 1 0" range="-2.844 0" damping="0.5" armature="0.01" frictionloss="0.1"/>
      <motor ctrlrange="-50 50" ctrllimited="true"/>
    </default>

    <default class="visual">
      <geom type="mesh" contype="0" conaffinity="0" group="2"/>
    </default>

    <default class="collision">
      <geom contype="1" conaffinity="1" group="3"/>
      <default class="foot">
        <geom type="sphere" size="0.025" priority="1" condim="6" friction="0.8 0.8 0.8"/>
      </default>
    </default>
  </default>
</default>
```

**설명:**
- 조인트 타입별로 기본값 분리 (abduction, hip, knee)
- 왼쪽/오른쪽 다리 구분 (left_abduction, right_abduction)
- 각 클래스에 motor 설정 추가
- foot의 condim을 3에서 6으로 변경하여 더 정확한 접촉 시뮬레이션

---

## 5. Asset 섹션 - 재질 및 텍스처 추가

**현재:**
```xml
<asset>
  <mesh name="TORSO" file="mesh/TORSO.STL" />
  ...
</asset>
```

**업그레이드:**
```xml
<asset>
  <!-- 재질 정의 -->
  <material name="metal" rgba=".9 .95 .95 1"/>
  <material name="black" rgba="0 0 0 1"/>
  <material name="white" rgba="1 1 1 1"/>
  <material name="gray" rgba="0.671705 0.692426 0.774270 1"/>

  <!-- 텍스처 및 텍스처 재질 정의 -->
  <material name="torso_material" texture="torso_texture"/>
  <texture name="torso_texture" type="2d" file="mesh/torso_skin.png" />
  <material name="hip_material" texture="hip_texture"/>
  <texture name="hip_texture" type="2d" file="mesh/hip_skin.png" />

  <!-- 메시 정의 -->
  <mesh name="TORSO" file="mesh/TORSO.STL" scale="1 1 1"/>
  <mesh name="TORSO_collision" file="mesh/TORSO_collision.STL" scale="1 1 1"/>
  ...
</asset>
```

**설명:**
- 시각적 품질 향상을 위한 재질 및 텍스처 추가
- 각 부품별로 visual용 메시와 collision용 메시를 분리
- `scale` 속성 명시

---

## 6. Inertial 속성 상세화

**현재:**
```xml
<inertial pos="-0.009 0.001 0" quat="0.707107 0 0 0.707107" mass="1.683" diaginertia="0.004 0.003 0.003" />
```

**업그레이드:**
```xml
<inertial
    pos="-0.009 0.001 0"
    mass="1.683"
    fullinertia="0.004 0.003 0.003 0.0001 0.0001 0.0001"/>
```

**설명:**
- `diaginertia` 대신 `fullinertia` 사용
- fullinertia 형식: Ixx Iyy Izz Ixy Ixz Iyz (6개 값으로 완전한 관성 텐서 표현)
- 더 정확한 물리 시뮬레이션을 위해 비대각 요소 포함

---

## 7. Collision Geometry 개선

**현재:**
```xml
<body name="FL_THIGH" pos="0 0.11125 0">
  <inertial .../>
  <joint .../>
  <geom mesh="FL_THIGH" class="visual"/>
  <geom mesh="FL_THIGH" class="collision"/>
  ...
</body>
```

**업그레이드:**
```xml
<body name="FL_THIGH" pos="0 0.11125 0">
  <inertial .../>
  <joint .../>
  <geom type="mesh" mesh="FL_THIGH" material="thigh_material" class="visual"/>
  <geom type="cylinder" size="0.06 0.035" euler="1.57 0 0" class="collision"/>
  <geom type="box" size="0.025 0.02 0.125" pos="-0.035 0 -0.125" class="collision"/>
  ...
</body>
```

**설명:**
- Visual geometry는 메시 사용 (외관)
- Collision geometry는 단순 형상(cylinder, box) 사용 (성능 향상)
- 복잡한 메시 대신 기본 형상으로 충돌 검사 속도 향상

---

## 8. 카메라 추가

**현재:** 카메라 없음

**업그레이드:**
```xml
<body name="base" pos="0 0 0.5" childclass="krm">
  <freejoint/>
  <site name="imu" pos="0 0 0"/>

  <!-- 트래킹 카메라 -->
  <camera name="track_cam" mode="track" target="base" pos="0 -2 1" euler="20 0 0"/>

  <!-- 고정 카메라들 -->
  <body name="front_cam_body" pos="0.4 0 0">
    <camera name="front_cam" mode="fixed" pos="0 0 0" quat="1 0 0 0"
            focal="1.93e-3 1.93e-3" resolution="640 360"
            sensorsize="3896e-6 2140e-6"/>
  </body>
  ...
</body>
```

**설명:**
- 로봇을 따라가는 트래킹 카메라 추가
- 다양한 각도에서 관찰할 수 있는 고정 카메라들 추가
- 시뮬레이션 시각화 및 비전 센서 활용 가능

---

## 9. Actuator 타입 변경 고려

**현재 (Position Control):**
```xml
<actuator>
  <position name="FL_HR" joint="FL_HR_JOINT" kp="800" ctrlrange="-0.785 0.785" forcerange="-50 50"/>
  ...
</actuator>
```

**업그레이드 옵션 1 (Torque Control):**
```xml
<actuator>
  <motor class="left_abduction" name="FL_HR" joint="FL_HR_JOINT"/>
  ...
</actuator>
```

**업그레이드 옵션 2 (유지):**
```xml
<actuator>
  <position name="FL_HR" joint="FL_HR_JOINT" kp="800" ctrlrange="-0.785 0.785" forcerange="-50 50"/>
  ...
</actuator>
```

**설명:**
- rbq.xml은 `motor` 타입 사용 (토크 제어)
- qugv.xml은 `position` 타입 사용 (위치 제어)
- 선택 기준:
  - **Motor (Torque Control)**: 저수준 제어, 강화학습에 적합, 더 현실적
  - **Position Control**: 고수준 제어, 경로 추적에 적합, 안정적

---

## 10. Sensor 섹션 대폭 확장

**현재:**
```xml
<sensor>
  <framequat name="orientation" objtype="site" objname="imu"/>
  <gyro name="gyro" site="imu"/>
  <accelerometer name="accel" site="imu"/>
  <framepos name="pos_FL" objtype="site" objname="FL_foot_site"/>
  ...
</sensor>
```

**업그레이드:**
```xml
<sensor>
  <!-- 조인트 위치 센서 (12개 조인트) -->
  <jointpos name="FL_HR_JOINT_pos" joint="FL_HR_JOINT"/>
  <jointpos name="FL_HP_JOINT_pos" joint="FL_HP_JOINT"/>
  <jointpos name="FL_KN_JOINT_pos" joint="FL_KN_JOINT"/>
  <!-- ... 나머지 9개 조인트 -->

  <!-- 조인트 속도 센서 (12개 조인트) -->
  <jointvel name="FL_HR_JOINT_vel" joint="FL_HR_JOINT"/>
  <jointvel name="FL_HP_JOINT_vel" joint="FL_HP_JOINT"/>
  <jointvel name="FL_KN_JOINT_vel" joint="FL_KN_JOINT"/>
  <!-- ... 나머지 9개 조인트 -->

  <!-- 조인트 토크 센서 (12개 조인트) -->
  <jointactuatorfrc name="FL_HR_JOINT_torque" joint="FL_HR_JOINT" noise="0.0"/>
  <jointactuatorfrc name="FL_HP_JOINT_torque" joint="FL_HP_JOINT" noise="0.0"/>
  <jointactuatorfrc name="FL_KN_JOINT_torque" joint="FL_KN_JOINT" noise="0.0"/>
  <!-- ... 나머지 9개 조인트 -->

  <!-- IMU 센서 -->
  <framequat name="imu_quat" objtype="site" objname="imu"/>
  <gyro name="imu_gyro" site="imu"/>
  <accelerometer name="imu_acc" site="imu"/>

  <!-- 프레임 위치/속도 센서 -->
  <framepos name="frame_pos" objtype="site" objname="imu"/>
  <framelinvel name="frame_vel" objtype="site" objname="imu"/>
</sensor>
```

**설명:**
- 모든 조인트의 위치, 속도, 토크 센서 추가 (총 36개 센서)
- IMU 센서 이름 변경 (일관성)
- 베이스 링크의 위치 및 속도 센서 추가
- 강화학습 및 제어에 필요한 모든 상태 정보 획득 가능

---

## 11. Contact 섹션 유지 또는 개선

**현재:**
```xml
<contact>
  <exclude body1="base" body2="FL_THIGH"/>
  <exclude body1="base" body2="FR_THIGH"/>
  <exclude body1="base" body2="RL_THIGH"/>
  <exclude body1="base" body2="RR_THIGH"/>
</contact>
```

**유지 또는 확장:**
```xml
<contact>
  <!-- 베이스와 허벅지 간 충돌 제외 -->
  <exclude body1="base" body2="FL_THIGH"/>
  <exclude body1="base" body2="FR_THIGH"/>
  <exclude body1="base" body2="RL_THIGH"/>
  <exclude body1="base" body2="RR_THIGH"/>

  <!-- 필요시 추가 충돌 제외 규칙 -->
  <exclude body1="FL_HIP" body2="FL_THIGH"/>
  <exclude body1="FR_HIP" body2="FR_THIGH"/>
  <exclude body1="RL_HIP" body2="RL_THIGH"/>
  <exclude body1="RR_HIP" body2="RR_THIGH"/>
</contact>
```

**설명:**
- 인접한 링크 간 불필요한 충돌 감지 제외
- 시뮬레이션 안정성 및 성능 향상

---

## 12. 추가 권장사항

### 12.1 주석 추가
```xml
<!-- FL (Front Left) 다리 -->
<body name="FL_HIP" pos="0.31218 0.09 0">
  ...
</body>
```

### 12.2 코드 포맷팅
- 들여쓰기 일관성 유지 (4 spaces)
- 속성이 많을 경우 여러 줄로 분리
- 관련 요소끼리 그룹화

### 12.3 명명 규칙 통일
- 현재: `FL_HR_JOINT`, `FL_HP_JOINT` 등
- rbq.xml 스타일: `joint9_FLR`, `joint10_FLP` 등
- 프로젝트 전체에서 일관된 명명 규칙 선택

---

## 업그레이드 우선순위

### 필수 (즉시 적용 권장):
1. ✅ Sensor 섹션 확장 (상태 피드백 향상)
2. ✅ Collision geometry 개선 (성능 향상)
3. ✅ Default 클래스 재구조화 (유지보수성 향상)
4. ✅ Inertial 속성 상세화 (물리 정확도 향상)

### 권장 (필요에 따라):
5. 📋 재질 및 텍스처 추가 (시각화 품질)
6. 📋 카메라 추가 (모니터링 편의성)
7. 📋 Option 태그 flag 추가 (세밀한 제어)

### 선택적:
8. 🔧 Actuator 타입 변경 (제어 방식에 따라)
9. 🔧 XML 선언 추가 (표준 준수)
10. 🔧 Contact 섹션 확장 (문제 발생 시)

---

## 적용 예시

전체 적용된 qugv.xml 예시 (일부):

```xml
<?xml version="1.0" encoding="utf-8"?>
<mujoco model="krm_qugv">
    <compiler angle="radian" coordinate="local" autolimits="true"/>

    <option
        timestep="0.002"
        gravity="0 0 -9.81"
        integrator="RK4"
        cone="elliptic"
        impratio="100"
        tolerance="1e-10">
        <flag warmstart="enable" energy="enable" contact="enable" frictionloss="enable"/>
    </option>

    <default>
        <default class="krm">
            <geom friction="0.8 0.02 0.01" margin="0.001" condim="3"/>

            <default class="hip">
                <joint damping="0.5" frictionloss="0.1" armature="0.01"/>
                <motor ctrlrange="-50 50" ctrllimited="true"/>
            </default>

            <!-- ... 다른 클래스 정의 ... -->
        </default>
    </default>

    <!-- ... 나머지 내용 ... -->
</mujoco>
```

---

## 참고사항

- 업그레이드 후에는 반드시 시뮬레이션을 실행하여 동작을 확인하세요
- 물리 파라미터 변경 시 로봇의 동작이 달라질 수 있습니다
- 점진적으로 적용하면서 각 변경사항의 영향을 확인하는 것이 좋습니다
- 원본 파일은 백업해두세요

