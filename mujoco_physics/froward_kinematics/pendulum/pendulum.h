#include "mjpc/task.h"

namespace mjpc {
class Pendulum : public Task {
 public:
  std::string Name() const override { return "PoE Pendulum"; }
  std::string XmlPath() const override { return "tasks/pendulum/pendulum.xml"; }

  // 목표 위치(IK 타겟)를 계산하는 핵심 함수
  void Residual(const mjModel* model, const mjData* data, double* residual) const override;
};
}