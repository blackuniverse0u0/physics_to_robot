# Unitree A1 Trot Gait Controller - 파일별 상세 설명

## 목차
1. [시스템 개요](#시스템-개요)
2. [실행에 필요한 파일 목록](#실행에-필요한-파일-목록)
3. [파일별 상세 설명](#파일별-상세-설명)
4. [제어 흐름도](#제어-흐름도)
5. [수학적 원리](#수학적-원리)

---

## 시스템 개요

이 프로젝트는 Unitree A1 4족 보행 로봇의 **Trot(트로트) 걸음걸이 제어기**를 MuJoCo 물리 시뮬레이터에서 구현한 것입니다.

**주요 특징:**
- 계층적 제어 구조 (Hierarchical Control Architecture)
- 유한 상태 기계(FSM) 기반 걸음걸이 조율
- 데카르트 공간 궤적 생성 + 해석적 역기구학
- PD 제어 + 가상 힘 제어 (Virtual Force Control)
- 1000Hz 제어 주기 (MuJoCo timestep = 0.001s)

**Trot 걸음걸이:**
대각선 방향의 다리 쌍이 함께 움직이는 걸음걸이
- 쌍 A: FR(앞 오른쪽) + RL(뒤 왼쪽)
- 쌍 B: FL(앞 왼쪽) + RR(뒤 오른쪽)

---

## 실행에 필요한 파일 목록

### 1. 메인 실행 파일
- `mj_a1_trot.py` - 시뮬레이션 메인 루프 및 시각화

### 2. MuJoCo 모델 파일
- `../unitree_robotics_a1/scene.xml` - 로봇 + 환경 씬
- `../unitree_robotics_a1/a1.xml` - A1 로봇 MJCF 모델

### 3. 설정/전역 변수 파일
- `globals.py` - 모든 모듈이 공유하는 전역 상태 변수
- `parameters.py` - 로봇 및 제어기 파라미터

### 4. 제어 계층 파일
- `high_level_control.py` - 상위 레벨 속도 명령 생성
- `state_machine.py` - FSM 기반 걸음걸이 상태 관리
- `cartesian_traj.py` - 데카르트 공간 발 궤적 생성
- `joint_traj.py` - 관절 공간 궤적 변환
- `joint_control.py` - 하위 레벨 관절 토크 제어

### 5. 운동학/동역학 파일
- `forward_kinematics_leg.py` - 단일 다리 순기구학
- `forward_kinematics_robot.py` - 전체 로봇 순기구학
- `inverse_kinematics_analytic.py` - 해석적 역기구학
- `jac_end_effector_leg.py` - 야코비안 계산
- `stance_force.py` - 지지 다리 가상 힘 계산

### 6. 유틸리티 파일
- `quintic_poly.py` - 5차 다항식 보간
- `set_command_step.py` - 명령 속도 제한
- `utility.py` - 회전 변환 유틸리티
- `robot_data.py` - 로봇 링크 데이터 정의

---

## 파일별 상세 설명

### 1. mj_a1_trot.py (메인 시뮬레이션 파일)

**역할:**
- MuJoCo 시뮬레이션 초기화 및 메인 루프 실행
- GLFW를 통한 3D 시각화 창 관리
- 제어 루프 호출 및 시뮬레이션 진행

**주요 코드 흐름:**
```python
# 1. 초기화
model = mj.MjModel.from_xml_path(xml_path)  # XML에서 모델 로드
data = mj.MjData(model)                      # 시뮬레이션 데이터 생성
globals.init()                               # 전역 변수 초기화

# 2. 초기 자세 설정
pos = [0, 0, 0.3]  # 몸통 위치
quat = [1,0,0,0]   # 몸통 자세
qleg = [hip, pitch, knee]  # 각 다리 관절 각도
data.qpos = np.concatenate((pos, quat, qleg, qleg, qleg, qleg))

# 3. 메인 루프 (시뮬레이션이 끝날 때까지)
while not glfw.window_should_close(window):
    # 제어 주기 동안 여러 번 실행 (1000Hz 제어, 15Hz 렌더링)
    while (data.time - time_prev < 1.0/15.0):
        state_machine()      # FSM 업데이트
        cartesian_traj()     # 발 궤적 생성
        joint_traj()         # 관절 궤적 변환

        if flag_trajectory_generation == 1:  # 운동학 모드
            data.qpos = ...  # 직접 위치 설정
        else:  # 동역학 모드
            joint_control()       # 토크 계산
            high_level_control()  # 상위 명령 업데이트
            data.ctrl = globals.trq  # 토크 적용
            mj.mj_step(model, data)  # 물리 시뮬레이션 1스텝

    # 4. 렌더링
    mj.mjv_updateScene(...)
    mj.mjr_render(...)
```

**원리:**
- MuJoCo는 물리 엔진으로 `mj_step()`을 호출할 때마다 동역학 방정식을 풀어 다음 상태를 계산
- 제어기는 현재 상태(`data.qpos`, `data.qvel`)를 읽어 토크(`data.ctrl`)를 계산
- 시각화는 물리 시뮬레이션보다 낮은 주파수(15Hz)로 업데이트하여 실시간 성능 확보

---

### 2. globals.py (전역 상태 관리)

**역할:**
- 모든 모듈이 공유하는 전역 변수 정의 및 초기화
- 파이썬의 모듈 단위 전역 변수를 사용하여 모듈 간 데이터 공유

**주요 변수:**

**FSM 상태:**
```python
fsm[4]      # 각 다리의 현재 상태 (1=stand, 2=stance, 3=swing)
t_fsm[4]    # 각 다리가 현재 상태에 진입한 시각
```

**궤적 참조값:**
```python
lx_ref[4], ly_ref[4], lz_ref[4]         # 발 위치 참조값 (다리 프레임)
lxdot_ref[4], lydot_ref[4], lzdot_ref[4]  # 발 속도 참조값
q_ref[12]   # 관절 각도 참조값 (3개 관절 × 4개 다리)
u_ref[12]   # 관절 속도 참조값
```

**센서 데이터:**
```python
q_act[12]           # 실제 관절 각도 (MuJoCo에서 읽음)
u_act[12]           # 실제 관절 속도
pos_quat_trunk[7]   # 몸통 위치 + 쿼터니언 [x,y,z,q0,qx,qy,qz]
vel_angvel_trunk[6] # 몸통 선속도 + 각속도 [vx,vy,vz,ωx,ωy,ωz]
```

**제어 명령:**
```python
xdot_ref, ydot_ref, psidot_ref  # 목표 속도 (전진, 측면, 요)
trq[12]                          # 계산된 관절 토크
```

**원리:**
- Python의 모듈 시스템을 활용: `import globals` 후 `globals.변수명`으로 접근
- 모든 모듈이 같은 메모리를 공유하므로 한 모듈에서 수정하면 다른 모듈에서도 반영됨
- `globals.init()` 함수로 모든 변수를 초기값으로 리셋

---

### 3. parameters.py (파라미터 정의)

**역할:**
- 로봇 물리적 특성 및 제어기 파라미터 정의
- `parms` 싱글톤 객체로 전역 접근

**주요 파라미터:**

**FSM 상태 ID:**
```python
fsm_stand = 1   # 정지 상태
fsm_stance = 2  # 지지 상태 (발이 땅에 닿음)
fsm_swing = 3   # 스윙 상태 (발이 공중에 있음)
```

**타이밍:**
```python
t_stand = 0.1s   # 초기 정지 시간
t_step = 0.15s   # 스윙/지지 단계 지속 시간
```

**기하학적 파라미터:**
```python
lz0 = -0.2486m  # 다리 프레임에서 발의 공칭 높이
hcl = 0.075m    # 스윙 시 발 들어올리는 높이
```

**로봇 물리:**
```python
mass = 12.453 kg  # 로봇 총 질량
gravity = 9.81 m/s²
```

**속도 제한:**
```python
vx_min/max = [-2.0, 2.0] m/s  # 전진 속도 범위
dvx = 0.1 m/s²  # 전진 속도 변화율
vy_min/max = [-1.0, 1.0] m/s  # 측면 속도 범위
omega_min/max = [-2.0, 2.0] rad/s  # 요 각속도 범위
```

**원리:**
- 클래스 기반 파라미터 관리로 네임스페이스 분리
- 물리적 파라미터는 실제 A1 로봇 사양에 기반
- 타이밍 파라미터는 안정적인 걸음걸이를 위해 실험적으로 조정됨

---

### 4. state_machine.py (상태 기계)

**역할:**
- 각 다리의 상태(stand/stance/swing)를 시간에 따라 전환
- Trot 걸음걸이 패턴 생성: 대각선 쌍이 동기화되어 움직임

**원리:**

**상태 전이 규칙:**

1. **Stand → Swing (초기 시작)**
   - 조건: `time >= t_fsm + t_stand` (정지 시간이 지남)
   - Leg 0, 3 (대각선 쌍): Swing 상태로 전환
   - Leg 1, 2 (반대 쌍): Stance 상태로 전환

2. **Stance → Swing**
   - 조건: `time >= t_fsm + t_step` (지지 시간이 지남)
   - 발을 들어올리기 위한 궤적 설정:
     - `lz_i = lz0, lz_f = lz0 + hcl` (수직 이동)
     - `lx_i/f, ly_i/f`: 속도 명령에 기반한 수평 이동
   - 요(yaw) 회전을 위한 보정:
     ```python
     if leg_no in [0, 1]:  # 앞다리
         ly_i -= 0.5*c*psidot_ref*t_step
         ly_f += 0.5*c*psidot_ref*t_step
     else:  # 뒷다리
         ly_i += 0.5*c*psidot_ref*t_step
         ly_f -= 0.5*c*psidot_ref*t_step
     ```

3. **Swing → Stance**
   - 조건: `time >= t_fsm + t_step` (스윙 시간이 지남)
   - 발이 착지하여 지지 시작
   - 뒤로 쓸어내리는 궤적 설정:
     - `lx_i = +0.5*xdot_ref*t_step → lx_f = -0.5*xdot_ref*t_step`
   - 스텝 카운터 증가 (앞다리가 착지할 때)

**수학적 배경:**
- **Raibert Heuristic**: 발 착지 위치를 속도와 관련시킴
  - 앞으로: `lx = ±0.5 * v * t_step` (속도의 절반만큼 앞/뒤)
  - 요 회전: 앞/뒷다리를 반대 방향으로 이동시켜 모멘트 생성

**코드 예시:**
```python
for leg_no in range(4):
    if (time >= t_fsm[leg_no] + t_stand and fsm[leg_no] == fsm_stand):
        if leg_no in [0, 3]:  # FR + RL 쌍
            fsm[leg_no] = fsm_swing
            t_fsm[leg_no] = time
            # 궤적 파라미터 설정...
        if leg_no in [1, 2]:  # FL + RR 쌍
            fsm[leg_no] = fsm_stance
            # ...
```

---

### 5. cartesian_traj.py (데카르트 궤적 생성)

**역할:**
- 각 다리의 발 끝 위치/속도를 다리 프레임 좌표계에서 생성
- Quintic polynomial(5차 다항식)을 사용한 부드러운 궤적

**원리:**

**Stand/Stance 상태:**
- 수직 방향만 제어 (발이 땅에 닿아 있음)
```python
lz_ref[leg], lzdot_ref[leg], _ = quintic_poly(
    t - t_fsm[leg], t_i[leg], t_f[leg], lz_i[leg], lz_f[leg]
)
lx_ref = ly_ref = 0  # 수평 이동 없음
```

**Swing 상태:**
- 수평 방향: 선형 보간
```python
lx_ref[leg], lxdot_ref[leg], _ = quintic_poly(
    t - t_fsm[leg], t_i[leg], t_f[leg], lx_i[leg], lx_f[leg]
)
ly_ref[leg], lydot_ref[leg], _ = quintic_poly(...)
```

- 수직 방향: 포물선 궤적 (2단계)
```python
if t - t_fsm[leg] <= 0.5 * t_step:  # 전반부: 올라가기
    lz_ref, lzdot_ref, _ = quintic_poly(
        t - t_fsm, t_i, t_f/2, lz_i, lz_f
    )
else:  # 후반부: 내려가기
    lz_ref, lzdot_ref, _ = quintic_poly(
        t - t_fsm, t_f/2, t_f, lz_f, lz_i
    )
```

**수학적 배경:**
- 5차 다항식은 위치, 속도, 가속도의 경계 조건을 모두 만족
- 스윙 궤적을 2단계로 나누어 부드러운 포물선 생성
- 시작/끝 속도와 가속도를 0으로 설정하여 충격 최소화

---

### 6. joint_traj.py (관절 궤적 변환)

**역할:**
- 데카르트 공간의 발 궤적을 관절 공간으로 변환
- 역기구학(IK) + 야코비안 역행렬 사용

**코드 흐름:**
```python
for leg_no in range(4):
    # 1. 위치: 역기구학으로 관절 각도 계산
    X_ref = [lx_ref[leg], ly_ref[leg], lz_ref[leg]]
    q_leg = inverse_kinematics_analytic(X_ref)
    globals.q_ref[3*leg:3*leg+3] = q_leg

    # 2. 속도: 야코비안 역행렬로 관절 속도 계산
    J_foot = jac_end_effector_leg(q_leg, leg_no)
    J_inv = np.linalg.inv(J_foot)
    Xdot_ref = [lxdot_ref[leg], lydot_ref[leg], lzdot_ref[leg]]
    u_leg = J_inv @ Xdot_ref
    globals.u_ref[3*leg:3*leg+3] = u_leg
```

**원리:**
- **역기구학**: 주어진 발 위치 (lx, ly, lz)에 대해 관절 각도 (q1, q2, q3) 계산
- **야코비안 관계**: `Xdot = J(q) * qdot` → `qdot = J^(-1) * Xdot`
- 3×3 야코비안 행렬은 정방행렬이므로 직접 역행렬 계산 가능

---

### 7. joint_control.py (관절 제어)

**역할:**
- 각 다리의 상태에 따라 관절 토크 계산
- Stand/Swing: PD 제어
- Stance: PD + 가상 힘 제어

**제어 법칙:**

**1. Stand/Swing 상태 - 단순 PD 제어:**
```python
trq = gain * (-kp*(q_act - q_ref) - kd*(u_act - u_ref))
# gain = 10, kp = 10, kd = 1
```

**2. Stance 상태 - PD + 중력 보상:**
```python
# 가상 힘 계산 (stance_force 함수 호출)
F0, F1, F2, F3 = stance_force(leg_no)

# 야코비안 전치를 통한 토크 변환
J = jac_end_effector_leg(q_leg, leg_no)
trq_grav = -J.T @ F

# 총 토크
trq = trq_grav + gain*(-kp*(q_act - q_ref) - kd*(u_act - u_ref))
```

**원리:**
- **PD 제어**: 비례-미분 제어로 위치와 속도 오차를 모두 보상
- **가상 힘 제어**: 지지 다리에 가상의 지면 반력을 계산하여 몸통 자세/속도 제어
- **야코비안 전치**: `τ = J^T * F` (가상 일의 원리)
  - 발 끝에 작용하는 힘을 관절 토크로 변환

**지지 다리 조율:**
- 대각선 쌍이 동시에 지지 상태일 때만 `stance_force()` 호출
- 두 발의 힘을 조율하여 몸통에 원하는 힘/모멘트 생성

---

### 8. high_level_control.py (상위 제어)

**역할:**
- 목표 속도를 설정하고 부드럽게 증가시킴
- 스텝마다 한 번씩 업데이트 (급격한 변화 방지)

**코드:**
```python
def high_level_control():
    if globals.prev_step < globals.step:  # 새로운 스텝 감지
        globals.prev_step = globals.step

        # 전진 속도
        vx = globals.xdot_ref  # 현재 속도
        vx_ = 1.0  # 목표 속도
        globals.xdot_ref = set_command_step(
            vx_, vx, parms.vx_min, parms.vx_max, parms.dvx
        )

        # 요 각속도
        omega = globals.psidot_ref
        omega_ = 1.0  # 목표 각속도
        globals.psidot_ref = set_command_step(
            omega_, omega, parms.omega_min, parms.omega_max, parms.domega
        )
```

**원리:**
- **Rate Limiting**: 급격한 속도 변화를 제한하여 안정성 확보
- **스텝 동기화**: 발이 착지하는 순간에만 명령 업데이트
- 현재 구현은 하드코딩된 목표값, 실제로는 조이스틱/키보드 입력을 받아야 함

---

### 9. stance_force.py (가상 힘 제어)

**역할:**
- 지지 다리가 몸통을 안정화시키기 위해 가해야 할 지면 반력 계산
- 선형 시스템 `A*F = b` 풀이

**원리:**

**1. 문제 정의:**
두 지지 다리(예: leg 0, leg 3)가 몸통에 가하는 힘을 `F0`, `F3`라 하면:
```
합력:   F0 + F3 = F_desired
합모멘트: r0×F0 + r3×F3 = M_desired
```

행렬 형태로:
```python
A = [I3,  I3 ]  # 3×6 행렬
    [R0,  R3 ]  # 3×6 행렬 (R은 skew-symmetric)

F = [F0]  # 6×1 벡터
    [F3]

b = [fx, fy, fz]     # 힘 목표
    [Mx, My, Mz]     # 모멘트 목표
```

**2. 목표 설정 (b 벡터):**
```python
# 위치 제어
fx = 100 * (xdot_ref - xdot)  # 전진 속도 추종
fy = 100 * (ydot_ref - ydot)  # 측면 속도 추종
fz = 50*(-10*(z - z_ref) - 1*zdot) + mass*gravity  # 높이 유지

# 자세 제어
Mx = 50*(-10*roll - 0.5*ωx)   # 롤 각도 0으로
My = 50*(-10*pitch - 0.5*ωy)  # 피치 각도 0으로
Mz = 10*(psidot_ref - psidot)  # 요 각속도 추종

b = [fx, fy, fz, Mx, My, Mz]
```

**3. 힘 계산:**
```python
A_inv = np.linalg.pinv(A)  # Pseudo-inverse (6×6)
F = A_inv @ b               # 최소 노름 해
```

**수학적 배경:**
- **Wrench (렌치)**: 힘과 모멘트를 합친 6차원 벡터
- **Force Distribution**: 과결정 시스템(더 많은 발)에서 pseudo-inverse로 최적 분배
- **PD 제어**: 각 축에 PD 게인을 적용하여 위치/자세 안정화
- **중력 보상**: `fz`에 `mg` 추가하여 자유 낙하 방지

**좌표계 변환:**
```python
# 요 각도만 제거한 body frame 사용
Rz = rotation(euler[2], 2)  # Yaw 회전 행렬
R_body = Rz.T @ R           # Yaw 제거
vel_body = Rz.T @ vel       # 속도를 body frame으로 변환
```
→ 롤/피치만 제어하고 요는 적분하지 않음

---

### 10. forward_kinematics_leg.py (다리 순기구학)

**역할:**
- 관절 각도가 주어졌을 때 발 끝 위치 계산
- 동차 변환 행렬(Homogeneous Transformation) 사용

**코드 구조:**
```python
def forward_kinematics_leg(q, leg_no):
    # 파라미터
    L = 0.2  # 넓적다리/정강이 길이
    w = ±0.08505  # 고관절 측면 오프셋 (좌/우에 따라 부호 반대)

    # 각 관절 변환
    H01 = [[1,  0,   0,  0  ],    # Hip abduction
           [0,  c1, -s1, 0  ],
           [0,  s1,  c1, 0  ],
           [0,  0,   0,  1  ]]

    H12 = [[c2,  0,  s2, 0  ],    # Hip pitch
           [0,   1,  0,  w  ],
           [-s2, 0,  c2, 0  ],
           [0,   0,  0,  1  ]]

    H23 = [[c3,  0,  s3, 0  ],    # Knee pitch
           [0,   1,  0,  0  ],
           [-s3, 0,  c3, -L ],
           [0,   0,  0,  1  ]]

    # 합성 변환
    H03 = H01 @ H12 @ H23

    # 발 끝 위치
    end_eff_pos_local = [0, 0, -L, 1]
    end_eff_pos = H03 @ end_eff_pos_local

    return end_eff_pos[0:3], H01, H02, H03
```

**원리:**
- **DH Convention (변형)**: 각 조인트를 4×4 동차 변환으로 표현
- **관절 구조**:
  1. Hip abduction (q1): X축 회전
  2. Hip pitch (q2): Y축 회전, Y 방향으로 w만큼 이동
  3. Knee pitch (q3): Y축 회전, Z 방향으로 -L만큼 이동
- **좌표계**: 다리 프레임 원점은 hip joint, Z축이 아래 방향

---

### 11. forward_kinematics_robot.py (로봇 전체 순기구학)

**역할:**
- 몸통 자세 + 모든 다리 관절 → 모든 링크 위치 계산
- `robot_data.py`의 링크 정보 사용

**알고리즘:**
```python
def forward_kinematics_robot(q):
    q_trunk = q[:7]   # [x, y, z, q0, qx, qy, qz]
    q_legs = q[7:]    # 12개 관절 각도

    # Step 1: 각 링크의 로컬 변환 계산
    for i in range(2, 14):  # body[2]~body[13]: 모든 다리 링크
        joint_axis = robot.body[i].joint_axis
        angle = q_legs[j]
        R_q = rotation(angle, axis_id)
        robot.body[i].R_local = quat2rotation(quat) @ R_q
        robot.body[i].H_local = [R_local | o_local]
                                 [  0     |    1    ]

    # Step 2: 글로벌 변환 계산 (트리 구조 순회)
    robot.body[1].H_global = [R_trunk | pos_trunk]
                              [   0    |     1     ]

    for leg in range(4):
        H_global = robot.body[1].H_global  # 몸통에서 시작
        for joint in leg_joints:
            H_global = H_global @ H_local
            robot.body[joint].H_global = H_global

    # Step 3: 각 링크 위치 추출
    for leg in range(4):
        shoulder_pos = H_global_shoulder @ [0,0,0,1]
        elbow_pos = H_global_elbow @ [0,0,0,1]
        end_eff_pos = H_global_calf @ [0,0,-L,1]
```

**원리:**
- **트리 구조**: 몸통(root) → 4개 다리(branch) → 각 다리 3개 링크
- **글로벌 변환**: `H_global_child = H_global_parent @ H_local_child`
- **질량 중심**: 몸통 COM은 `trunk_com_pos = H_trunk @ ipos`

---

### 12. inverse_kinematics_analytic.py (역기구학)

**역할:**
- 발 끝 목표 위치 (lx, ly, lz) → 관절 각도 (q1, q2, q3) 계산
- 해석적 해(Closed-form solution)

**알고리즘:**
```python
def inverse_kinematics_analytic(X_ref):
    lx, ly, lz = X_ref
    L = 0.2  # 링크 길이

    # 발 끝까지 거리
    l = sqrt(lx² + ly² + lz²)

    # 해법 (삼각법 사용)
    q_abduction = arcsin(ly / l)
    q_knee = -π + arccos((2L² - l²) / (2L²))
    q_hip = -0.5*q_knee + arcsin(-lx / l)

    return [q_abduction, q_hip, q_knee]
```

**기하학적 원리:**
1. **Abduction 각도**: 측면 오프셋 `ly`를 사용
2. **Knee 각도**: 코사인 법칙 `l² = L² + L² - 2L²cos(π-q_knee)`
3. **Hip 각도**: 수직/수평 성분을 이용한 삼각법

**제약:**
- 2개 링크 길이가 같음 (L = 0.2m)
- 특이점(singularity): `l > 2L` (도달 불가), `l = 0` (무한 해)

---

### 13. jac_end_effector_leg.py (야코비안)

**역할:**
- 관절 속도 → 발 끝 속도 매핑: `Xdot = J(q) * qdot`
- 3×3 야코비안 행렬 계산

**공식:**
```python
def jac_end_effector_leg(q, leg_no):
    sol = forward_kinematics_leg(q, leg_no)

    # 각 조인트 축 (베이스 프레임에서)
    n1 = R00 @ [1,0,0]  # Hip abduction axis
    n2 = R01 @ [0,1,0]  # Hip pitch axis
    n3 = R02 @ [0,1,0]  # Knee pitch axis

    # 각 조인트에서 발 끝까지 벡터
    r1 = end_eff_pos - o01
    r2 = end_eff_pos - o02
    r3 = end_eff_pos - o03

    # 야코비안 컬럼: J_i = n_i × r_i
    Jv_E = [n1×r1, n2×r2, n3×r3]

    return Jv_E
```

**수학적 배경:**
- **회전 조인트**: 선속도 기여 = `ω × r = (n × r) * qdot`
- **Skew-symmetric matrix**: `vec2skew(n) @ r = n × r`
- 각 컬럼은 각 관절이 발 끝 속도에 기여하는 정도

---

### 14. quintic_poly.py (5차 다항식)

**역할:**
- 시작/끝 위치, 속도, 가속도를 모두 지정한 부드러운 궤적 생성

**수학:**
```python
def quintic_poly(t, t0, tf, q0, qf):
    # 5차 다항식: q(t) = a0 + a1*t + ... + a5*t^5

    # 경계 조건 (6개 방정식)
    # q(t0) = q0, q(tf) = qf
    # qdot(t0) = 0, qdot(tf) = 0
    # qddot(t0) = 0, qddot(tf) = 0

    A = [[1, t0, t0², t0³, t0⁴, t0⁵],
         [1, tf, tf², tf³, tf⁴, tf⁵],
         [0,  1, 2t0, 3t0², 4t0³, 5t0⁴],
         [0,  1, 2tf, 3tf², 4tf³, 5tf⁴],
         [0,  0,  2,  6t0, 12t0², 20t0³],
         [0,  0,  2,  6tf, 12tf², 20tf³]]

    b = [q0, qf, 0, 0, 0, 0]

    a = inv(A) @ b  # 계수 해결

    # 현재 시간 t에서 평가
    q = a[0] + a[1]*t + a[2]*t² + a[3]*t³ + a[4]*t⁴ + a[5]*t⁵
    qdot = a[1] + 2*a[2]*t + 3*a[3]*t² + 4*a[4]*t³ + 5*a[5]*t⁴
    qddot = 2*a[2] + 6*a[3]*t + 12*a[4]*t² + 20*a[5]*t³

    return q, qdot, qddot
```

**왜 5차?**
- 6개 경계 조건을 만족하려면 최소 5차 필요
- 시작/끝에서 가속도=0 → 저크(jerk) 최소화 → 부드러운 움직임

---

### 15. set_command_step.py (속도 제한)

**역할:**
- 목표 속도로 부드럽게 증가/감소 (rate limiting)

**알고리즘:**
```python
def set_command_step(cmd_des, cmd_curr, cmd_min, cmd_max, cmd_rate):
    # 필요한 변화량
    delta = cmd_des - cmd_curr

    # 최대 변화율 적용
    rate = min(abs(delta), cmd_rate)

    # 방향에 따라 증가/감소
    if cmd_curr > cmd_des:
        cmd = cmd_curr - rate
    elif cmd_curr < cmd_des:
        cmd = cmd_curr + rate
    else:
        cmd = cmd_curr

    # 범위 제한
    cmd = clip(cmd, cmd_min, cmd_max)

    return cmd
```

**원리:**
- 급격한 변화를 방지하여 로봇이 넘어지지 않도록 함
- 매 스텝마다 최대 `cmd_rate`만큼만 변화

---

### 16. utility.py (회전 변환 유틸리티)

**역할:**
- 다양한 회전 표현 간 변환 함수 모음

**주요 함수:**

**기본 회전 행렬:**
```python
rotation(angle, axis)  # X/Y/Z축 기본 회전
```

**쿼터니언 ↔ 회전 행렬:**
```python
quat2rotation(q)       # q = [q0, qx, qy, qz] → R (3×3)
rotation2quat(R)       # R → q
mat2quat(R)            # 안정적 버전 (고유값 분해)
quat2mat(q)            # 안정적 버전
```

**쿼터니언 ↔ 오일러 각:**
```python
quat2bryant(q)         # q → ZYX 오일러 각
bryant2quat(euler)     # euler → q
mat2bryant(R)          # R → euler
bryant2mat(euler)      # euler → R
```

**쿼터니언 연산:**
```python
quat_product(q, p)     # 쿼터니언 곱셈
quat_conjugate(q)      # 켤레 (역회전)
quat_normalize(q)      # 정규화
```

**각속도 변환:**
```python
quat2angvelBody(q, qdot)    # ωb = 2*q_conj*qdot (body frame)
quat2angvelWorld(q, qdot)   # ω = 2*qdot*q_conj (world frame)
```

**기타:**
```python
vec2skew(v)            # 벡터 → 반대칭 행렬 (외적용)
quat2axisangle(q)      # q → (axis, angle)
```

**수학적 배경:**
- **쿼터니언**: 4D 단위 벡터로 3D 회전 표현, 짐벌락 없음
- **Bryant 각**: ZYX 순서 오일러 각 (yaw-pitch-roll)
- **Skew-symmetric matrix**: `[v]_× @ u = v × u` (외적을 행렬 곱셈으로)

---

### 17. robot_data.py (로봇 데이터)

**역할:**
- A1 로봇의 모든 링크(body) 물리적 특성 정의
- MuJoCo XML에서 추출한 데이터를 Python 객체로 재구성

**데이터 구조:**
```python
class Robot:
    class Body:
        parent       # 부모 링크 이름
        name         # 링크 이름
        pos          # 부모 기준 위치
        quat         # 부모 기준 자세 (쿼터니언)
        ipos         # COM 위치 (링크 프레임)
        iquat        # COM 자세
        mass         # 질량
        inertia      # 관성 텐서 [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
        joint_axis   # 조인트 축
        joint_range  # 조인트 각도 범위
```

**데이터 예시:**
```python
robot.add_body(
    1, parent='ground', name='trunk',
    pos=[0, 0, 0.43],
    quat=[1, 0, 0, 0],
    ipos=[0, 0.0041, -0.0005],  # COM offset
    iquat=[1, 0, 0, 0],
    mass=4.713,
    inertia=[0.0158533, 0.0377999, 0.0456542, -3.66e-05, -6.11e-05, -2.75e-05],
    joint_axis=[0, 0, 0, 1, 0, 0, 0],  # Free joint (7-DOF)
    joint_range=[-6.28319, 6.28319]
)

robot.add_body(
    2, parent='trunk', name='FR_hip_joint',
    pos=[0.183, -0.047, 0],  # 앞 오른쪽 고관절
    quat=[1, 0, 0, 0],
    mass=0.696,
    joint_axis=[1, 0, 0],     # X축 회전 (abduction)
    joint_range=[-0.802851, 0.802851]
)
```

**링크 번호:**
- Body 1: Trunk (몸통)
- Body 2-4: FR 다리 (hip, thigh, calf)
- Body 5-7: FL 다리
- Body 8-10: RR 다리
- Body 11-13: RL 다리

**원리:**
- MuJoCo XML의 `<body>`, `<joint>`, `<inertial>` 태그 정보를 파싱
- 쿼터니언 자동 정규화 (`quat_normalize`)
- `forward_kinematics_robot.py`에서 이 데이터 사용

---

## 제어 흐름도

```
┌─────────────────────────────────────────────────────────────┐
│                    mj_a1_trot.py (메인 루프)                    │
│                                                              │
│  while not done:                                             │
│    ┌─────────────────────────────────────────────────────┐  │
│    │ Control Loop (1000 Hz)                              │  │
│    │                                                      │  │
│    │  1. state_machine()  ────────────────────────┐      │  │
│    │     - 각 다리 FSM 상태 업데이트               │      │  │
│    │     - 궤적 시작/끝 위치 설정                  │      │  │
│    │                                              ▼      │  │
│    │  2. cartesian_traj()  ◄────────────── globals      │  │
│    │     - 5차 다항식으로 발 궤적 생성             │      │  │
│    │     - lx_ref, ly_ref, lz_ref 계산            │      │  │
│    │                                              │      │  │
│    │  3. joint_traj()                             │      │  │
│    │     - 역기구학: X_ref → q_ref               │      │  │
│    │     - 야코비안: Xdot_ref → u_ref            │      │  │
│    │                                              │      │  │
│    │  4. joint_control()                          │      │  │
│    │     - Stand/Swing: PD 제어                  │      │  │
│    │     - Stance: PD + 가상 힘 제어 ◄───┐       │      │  │
│    │                                      │       │      │  │
│    │  5. high_level_control()             │       │      │  │
│    │     - 목표 속도 업데이트 (스텝마다)  │       │      │  │
│    │                                      │       │      │  │
│    │                         stance_force()       │      │  │
│    │                            - A*F = b 풀이   │      │  │
│    │                            - 지면 반력 계산  │      │  │
│    │                                              │      │  │
│    │  6. data.ctrl = globals.trq                 │      │  │
│    │  7. mj.mj_step(model, data)  ─────► MuJoCo  │      │  │
│    │                                      물리 시뮬│      │  │
│    └──────────────────────────────────────────────┘      │
│                                                           │
│    Rendering (15 Hz)                                      │
│    - mjv_updateScene()                                    │
│    - mjr_render()                                         │
└───────────────────────────────────────────────────────────┘

보조 모듈:
┌──────────────────────────────────────────────────────────┐
│ forward_kinematics_leg/robot: q → X                      │
│ inverse_kinematics_analytic: X → q                       │
│ jac_end_effector_leg: J(q) 계산                          │
│ quintic_poly: 부드러운 궤적 생성                          │
│ set_command_step: 속도 제한                               │
│ utility: 회전 변환 (quat, euler, matrix)                 │
│ robot_data: 로봇 링크 데이터                              │
└──────────────────────────────────────────────────────────┘
```

---

## 수학적 원리

### 1. 동차 변환 행렬 (Homogeneous Transformation)

**정의:**
```
H = [R | p]  ∈ SE(3)  (4×4 행렬)
    [0 | 1]

R: 3×3 회전 행렬
p: 3×1 위치 벡터
```

**합성:**
```
H_AC = H_AB @ H_BC
```

**순기구학:**
```
X_end = H_base_to_end @ X_local
```

### 2. 역기구학 (Inverse Kinematics)

**문제:**
주어진 발 끝 위치 `X = [lx, ly, lz]`에 대해 관절 각도 `q = [q1, q2, q3]` 찾기

**해법 (기하학적):**
- 3-링크 평면 팔의 해석적 해 사용
- 삼각법과 코사인 법칙 활용

**제약:**
- 특이점: `l = 0` (몸통과 발이 같은 위치), `l > 2L` (도달 불가)
- 다중 해: 일반적으로 2개 (elbow up/down), 여기서는 1개만 선택

### 3. 야코비안 (Jacobian)

**정의:**
```
Xdot = J(q) * qdot
```

**역야코비안:**
```
qdot = J^(-1) * Xdot
```

**계산 (회전 조인트):**
```
J_i = n_i × (X_end - O_i)

n_i: i번째 조인트 축 벡터
O_i: i번째 조인트 위치
```

### 4. 가상 힘 제어 (Virtual Force Control)

**Wrench (렌치):**
```
W = [F]  ∈ R^6
    [M]

F: 힘 (3×1)
M: 모멘트 (3×1)
```

**지면 반력의 합:**
```
W_total = Σ [    F_i    ]
          [ r_i × F_i ]

A*F = b 형태로 변환
A: 6×(3n) 행렬 (n개 지지 다리)
F: 3n×1 벡터 (각 다리의 힘)
b: 6×1 목표 wrench
```

**해:**
```
F = A^+ * b  (pseudo-inverse)
```

### 5. PD 제어 (Proportional-Derivative Control)

**제어 법칙:**
```
τ = -Kp*(q - q_ref) - Kd*(qdot - qdot_ref)

Kp: 비례 게인 (위치 오차)
Kd: 미분 게인 (속도 오차)
```

**안정성 조건 (단순 시스템):**
```
Kd > 0, Kp > 0
임계 감쇠: Kd = 2*sqrt(Kp*m)
```

### 6. Quintic Polynomial (5차 다항식)

**형태:**
```
q(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
```

**경계 조건 (6개):**
```
q(t0) = q0,   q(tf) = qf
qdot(t0) = 0, qdot(tf) = 0
qddot(t0) = 0, qddot(tf) = 0
```

**장점:**
- 시작/끝에서 속도와 가속도가 0 → 충격 없음
- 연속적인 jerk → 부드러운 움직임

### 7. 쿼터니언 (Quaternion)

**정의:**
```
q = [q0, qx, qy, qz] ∈ R^4
||q|| = 1 (단위 쿼터니언)

q0 = cos(θ/2)
[qx, qy, qz] = sin(θ/2) * axis
```

**회전 행렬 변환:**
```
R = [1-2(qy²+qz²)   2(qxqy-q0qz)   2(qxqz+q0qy)]
    [2(qxqy+q0qz)   1-2(qx²+qz²)   2(qyqz-q0qx)]
    [2(qxqz-q0qy)   2(qyqz+q0qx)   1-2(qx²+qy²)]
```

**각속도 변환:**
```
ωb = 2 * q_conj ⊗ qdot  (body frame)
ω = 2 * qdot ⊗ q_conj   (world frame)

⊗: 쿼터니언 곱셈
```

### 8. Raibert Heuristic (발 착지 위치)

**원리:**
속도 추종을 위한 발 착지 위치 결정

**공식:**
```
x_foot = x_hip + 0.5*v*t_stance + Kp*(v - v_ref)

0.5*v*t_stance: 중립점 (몸통 아래)
Kp*(v - v_ref): 속도 오차 보상
```

**구현:**
```python
lx_i = -0.5 * xdot_ref * t_step  # 뒤에서 시작
lx_f = +0.5 * xdot_ref * t_step  # 앞에서 끝
```

---

## 실행 예시

```bash
cd /Users/joonhyunshin/Physics/physics_to_robot/mujoco_physics/trot/a1_trot
python mj_a1_trot.py
```

**시뮬레이션 시퀀스:**

1. **t=0~0.1s**: Stand 상태
   - 모든 다리가 정지
   - 높이 0.3m에서 균형

2. **t=0.1s**: 첫 스텝 시작
   - Leg 0, 3 → Swing
   - Leg 1, 2 → Stance

3. **t=0.25s**: 첫 스텝 완료
   - Leg 0, 3 → Stance
   - Leg 1, 2 → Swing

4. **t=0.4s 이후**: 주기적 trot
   - 0.15s마다 상태 전환
   - 전진 속도 1.0 m/s로 증가
   - 요 각속도 1.0 rad/s로 회전

**기대 결과:**
- 로봇이 앞으로 이동하면서 제자리에서 회전
- 몸통 높이 약 0.249m 유지
- 롤/피치 각도 0 근처 유지

---

## 요약

이 트로트 제어기는 다음 계층으로 구성됩니다:

1. **High-level**: 목표 속도 설정
2. **FSM**: 다리 상태 조율
3. **Trajectory**: 발 궤적 생성
4. **IK**: 관절 각도 계산
5. **Control**: PD + 가상 힘 제어
6. **Physics**: MuJoCo 시뮬레이션

각 계층은 이전 계층의 출력을 입력으로 받아 처리하며, 모든 데이터는 `globals.py`를 통해 공유됩니다.
