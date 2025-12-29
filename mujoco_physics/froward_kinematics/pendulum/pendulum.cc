#include "tasks/pendulum/pendulum.h"
#include "mjpc/utilities.h"

namespace mjpc {
void Pendulum::Residual(const mjModel* model, const mjData* data, double* residual) const {
  // 1. 목표 지점 (GUI 파라미터에서 가져옴)
  double target_x = parameters[0];
  double target_z = parameters[1];

  // 2. PoE Forward Kinematics (FK)
  double q1 = data->qpos[0];
  double q2 = data->qpos[1];
  double L1 = 0.5, L2 = 0.5;
  
  double current_x = L1 * sin(q1) + L2 * sin(q1 + q2);
  double current_z = 1.0 - (L1 * cos(q1) + L2 * cos(q1 + q2));

  // 3. 잔차(Residual) 설정: 목표와 현재 위치의 차이를 최소화하도록 MPC가 설계됨
  residual[0] = current_x - target_x;
  residual[1] = current_z - target_z;
  
  // 4. 제어 비용 (에너지 최소화)
  residual[2] = data->ctrl[0] * 0.01;
  residual[3] = data->ctrl[1] * 0.01;
}
}