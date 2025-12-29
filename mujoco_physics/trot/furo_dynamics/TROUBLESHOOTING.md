# Furo Trot Controller - 문제 해결 가이드

## 해결된 문제들

### 1. Array Shape Mismatch (✅ 해결됨)

**문제:**
```
ValueError: could not broadcast input array from shape (19,) into shape (96,)
```

**원인:**
- Furo scene.xml에 환경(docking station 등)이 포함되어 총 96 DOF
- 로봇만 19 DOF (base freejoint 7 + leg joints 12)

**해결:**
```python
# mj_furo_trot.py 수정
# Keyframe 사용
mj.mj_resetDataKeyframe(model, data, 0)

# 로봇 부분만 읽기/쓰기
globals.q_act = data.qpos[7:19].copy()  # 12개 관절만
globals.u_act = data.qvel[6:18].copy()
```

### 2. Inverse Kinematics NaN (✅ 해결됨)

**문제:**
```
RuntimeWarning: invalid value encountered in arccos
WARNING: Nan, Inf or huge value in CTRL at ACTUATOR
```

**원인:**
- 역기구학이 도달 불가능한 위치 계산 시도
- arcsin/arccos에 [-1, 1] 범위 밖 값 입력

**해결:**
```python
# inverse_kinematics_analytic.py 수정
# 거리 제한
l_max = L1 + L2 - 0.01  # 0.69m
l_min = 0.05
l = np.clip(l, l_min, l_max)

# 삼각함수 입력값 제한
ly_ratio = np.clip(ly / l, -1.0, 1.0)
cos_knee = np.clip(cos_knee, -1.0, 1.0)
lx_ratio = np.clip(-lx / l, -1.0, 1.0)
```

## 현재 상태

### ✅ 정상 작동하는 기능
- MuJoCo 모델 로딩 (96 DOF)
- Keyframe 초기화
- FSM 상태 전환 (Stand → Swing/Stance)
- 궤적 생성 (cartesian_traj, joint_traj)
- 관절 제어 (PD + 가상 힘 제어)
- 시뮬레이션 실행 (에러 없음)

### ⚠️ 개선 필요한 부분

#### 1. 로봇 높이 유지
**증상:** 로봇이 떨어짐 (0.95m → 0.09m)

**가능한 원인:**
- 제어 게인이 Furo의 무게에 비해 부족
- 가상 힘 제어의 목표 높이가 잘못됨
- PD 게인이 너무 작음

**해결 방법:**
```python
# parameters.py 조정
self.lz0 = -0.533  # 현재 값 확인 필요

# stance_force.py 게인 증가
fx0 = 100 * (globals.xdot_ref - xdot)  # 80 → 100
fz0 = 50 * (-10*(z - z_ref) - 1*zdot) + parms.mass * parms.gravity  # 40 → 50

# joint_control.py PD 게인 조정
gain = 15  # 10 → 15
kp = 15    # 10 → 15
kd = 2     # 1 → 2
```

#### 2. 걸음걸이 안정성
**증상:** 로봇이 옆으로 기울어짐

**해결 방법:**
```python
# parameters.py
self.t_step = 0.25  # 0.2 → 0.25 (더 느린 걸음)
self.hcl = 0.08     # 0.10 → 0.08 (발 덜 들기)

# high_level_control.py
vx_ = 0.3  # 0.5 → 0.3 (더 느린 속도)
omega_ = 0.3  # 0.5 → 0.3
```

#### 3. 발 궤적 최적화
**증상:** 발이 지면 아래로 내려가려고 함

**해결 방법:**
```python
# state_machine.py - Raibert heuristic 조정
globals.lx_i[leg_no] = -0.4 * globals.xdot_ref * t_step  # 0.5 → 0.4
globals.lx_f[leg_no] = 0.4 * globals.xdot_ref * t_step
```

## 디버깅 도구

### 1. 초기화 테스트
```bash
python test_init.py
```
- 모든 모듈 로딩 확인
- 순기구학/역기구학 검증
- 첫 제어 스텝 확인

### 2. 시뮬레이션 테스트 (GUI 없이)
```bash
python test_simulation.py
```
- 1000 스텝 실행
- 위치, 자세, FSM 상태 출력
- 안정성 확인

### 3. 전체 시뮬레이션 (GUI)
```bash
python mj_furo_trot.py
```
- 시각적 확인
- 실시간 모니터링

## 파라미터 튜닝 가이드

### 제어 게인 조정 순서
1. **PD 게인** (joint_control.py)
   - kp: 위치 추종 강도 (너무 크면 진동)
   - kd: 댐핑 (너무 작으면 불안정)

2. **가상 힘 게인** (stance_force.py)
   - 높이 제어: `fz0` 계수
   - 자세 제어: `Mx0, My0` 계수
   - 속도 추종: `fx0, fy0` 계수

3. **걸음걸이 타이밍** (parameters.py)
   - t_step: 스텝 지속 시간
   - hcl: 발 들어올림 높이

### 디버그 플래그
```python
# mj_furo_trot.py
print_camera_config = 1  # 카메라 위치 출력
print_model = 1          # 모델 정보 저장

# test_simulation.py
render_every = 10  # 더 자주 출력
```

## 성능 지표

### 목표 성능
- ✅ 시뮬레이션 안정성: 에러 없이 실행
- ⚠️ 높이 유지: 0.55m ± 0.05m
- ⚠️ 전진 속도: 0.3~0.5 m/s
- ⚠️ 회전 안정성: ±5° 이내

### 현재 성능 (2초 시뮬레이션)
- ✅ 안정성: 에러 없음
- ❌ 높이: 0.95m → 0.09m (떨어짐)
- ⚠️ 속도: 0.19 m/s (목표보다 느림)
- ✅ FSM: 정상 전환

## 다음 단계

1. ✅ Array shape 문제 해결
2. ✅ IK NaN 문제 해결
3. ⚠️ 제어 게인 튜닝 (진행 중)
4. ⚠️ 높이 유지 개선
5. ⬜ 걸음걸이 안정화
6. ⬜ 속도 제어 최적화
7. ⬜ 실제 로봇 배포 준비
