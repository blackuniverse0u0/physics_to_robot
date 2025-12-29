# Furo Robot Trot Gait Controller

Unitree A1 트로트 제어기를 기반으로 Furo 로봇에 맞게 수정한 걸음걸이 제어기입니다.

## 파일 구조

```
furo_dynamics/
├── mj_furo_trot.py              # 메인 시뮬레이션 파일
├── globals.py                    # 전역 상태 변수
├── parameters.py                 # Furo 로봇 파라미터
├── state_machine.py              # FSM 기반 걸음걸이 조율
├── cartesian_traj.py             # 데카르트 공간 발 궤적 생성
├── joint_traj.py                 # 관절 공간 궤적 변환
├── joint_control.py              # 관절 토크 제어
├── high_level_control.py         # 상위 레벨 속도 명령
├── forward_kinematics_leg.py     # 다리 순기구학
├── forward_kinematics_robot.py   # 전체 로봇 순기구학
├── inverse_kinematics_analytic.py # 해석적 역기구학
├── jac_end_effector_leg.py       # 야코비안 계산
├── stance_force.py               # 가상 힘 제어
├── quintic_poly.py               # 5차 다항식 궤적
├── set_command_step.py           # 속도 제한
└── utility.py                    # 회전 변환 유틸리티
```

## Furo vs A1 주요 차이점

### 로봇 기하학
- **A1**:
  - 다리 길이: 0.4m (thigh=0.2m, calf=0.2m)
  - 고관절 측면 오프셋: 0.08505m
  - 몸통 질량: 12.453kg

- **Furo**:
  - 다리 길이: 0.7m (thigh=0.35m, calf=0.35m)
  - 고관절 측면 오프셋: 0.11125m
  - 고관절 전후 오프셋: ±0.3985m
  - 몸통 질량: 15.702kg

### 제어 파라미터
- **스텝 시간**: 0.15s → 0.2s (Furo가 더 무거움)
- **발 들어올림 높이**: 0.075m → 0.10m
- **목표 속도**: 전진 1.0 m/s → 0.5 m/s, 회전 1.0 rad/s → 0.5 rad/s
- **제어 게인**: Furo의 무게에 맞게 조정

### 관절 구조
- **A1**: Hip Abduction (X축), Hip Pitch (Y축), Knee (Y축)
- **Furo**: Hip Roll (X축), Hip Pitch (Y축), Knee (Y축) - 동일 구조

## 실행 방법

```bash
cd /Users/joonhyunshin/Physics/physics_to_robot/mujoco_physics/trot/furo_dynamics
python mj_furo_trot.py
```

## 초기 설정

- **초기 높이**: 0.55m (A1은 0.3m)
- **초기 관절 각도**:
  - Hip roll: 0.0 rad
  - Hip pitch: 0.785 rad
  - Knee: -2.84 rad

## 제어 흐름

1. **High-level Control**: 목표 속도 설정 (0.5 m/s 전진, 0.5 rad/s 회전)
2. **State Machine**: Trot 걸음걸이 FSM (FR+RL / FL+RR 대각선 쌍)
3. **Cartesian Trajectory**: 5차 다항식으로 발 궤적 생성
4. **Joint Trajectory**: 역기구학 + 야코비안으로 관절 궤적 변환
5. **Joint Control**:
   - Stand/Swing: PD 제어
   - Stance: PD + 가상 힘 제어 (몸통 안정화)

## 주요 수정 사항

### parameters.py
- Furo의 물리적 특성 반영 (질량, 다리 길이)
- 더 긴 스텝 시간과 낮은 목표 속도

### forward_kinematics_leg.py
- Furo의 링크 길이 (L1=0.35m, L2=0.35m)
- 고관절 측면 오프셋 (w=±0.11125m)

### inverse_kinematics_analytic.py
- 새로운 링크 길이에 맞게 역기구학 공식 수정

### forward_kinematics_robot.py
- Furo의 고관절 위치 (±0.3985m 전후, ±0.07m 좌우)

### stance_force.py
- Furo의 무게에 맞게 제어 게인 조정
- 더 무거운 로봇을 위한 힘 분배

### mj_furo_trot.py
- Furo XML 파일 경로 (`../Furo/scene.xml`)
- 초기 자세: 높이 0.55m, keyframe 기반 관절 각도
- 카메라 거리 조정 (더 큰 로봇)

## 예상 동작

1. **t=0~0.1s**: Stand 상태 (정지)
2. **t=0.1s~**: Trot 걸음걸이 시작
   - FR + RL 쌍 → Swing
   - FL + RR 쌍 → Stance
3. **t=0.3s 이후**: 주기적 trot (0.2s 주기)
4. **최종**: 전진하면서 제자리 회전

## 문제 해결

### 로봇이 넘어지는 경우
- `parameters.py`에서 `t_step` 증가 (더 느린 걸음)
- `high_level_control.py`에서 목표 속도 감소
- `stance_force.py`에서 제어 게인 조정

### 발이 지면을 뚫는 경우
- `parameters.py`에서 `lz0` 값 확인 (초기 발 높이)
- `inverse_kinematics_analytic.py`의 역기구학 해 확인

### 불안정한 걸음걸이
- `joint_control.py`에서 PD 게인 조정
- `parameters.py`에서 `hcl` (발 들어올림 높이) 조정

## 참고 자료

- 원본 A1 트로트 제어기: `../a1_trot/`
- Furo 로봇 모델: `../Furo/scene.xml`
- 전체 문서: `../trot.md`
