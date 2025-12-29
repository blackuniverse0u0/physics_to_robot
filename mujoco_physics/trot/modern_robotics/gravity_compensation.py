import time
import mujoco
import mujoco.viewer
import numpy as np

# -----------------------------------------------------------
# 1. Math Helper Class (PoE 구현을 위한 선형대수 함수들)
# -----------------------------------------------------------
class PoEUtils:
    @staticmethod
    def Skew(v):
        """3D 벡터를 3x3 Skew-symmetric 행렬로 변환 (so3)"""
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])

    @staticmethod
    def VecTose3(S):
        """6D Screw vector를 4x4 se3 행렬로 변환"""
        w = S[:3]
        v = S[3:]
        se3 = np.eye(4)
        se3[:3, :3] = PoEUtils.Skew(w)
        se3[:3, 3] = v
        return se3

    @staticmethod
    def MatrixExp3(so3theta):
        """Rodrigues 공식을 이용한 회전 행렬 지수 함수"""
        w_vec = np.array([so3theta[2, 1], so3theta[0, 2], so3theta[1, 0]])
        theta = np.linalg.norm(w_vec)
        if theta < 1e-6:
            return np.eye(3)
        omega_hat = so3theta / theta
        return np.eye(3) + np.sin(theta) * omega_hat + (1 - np.cos(theta)) * np.dot(omega_hat, omega_hat)

    @staticmethod
    def MatrixExp6(se3theta):
        """스크류 모션에 대한 4x4 변환 행렬 지수 함수"""
        omega_skew = se3theta[:3, :3]
        v = se3theta[:3, 3]
        w_vec = np.array([omega_skew[2, 1], omega_skew[0, 2], omega_skew[1, 0]])
        theta = np.linalg.norm(w_vec)
        
        if theta < 1e-6:
            return np.eye(4) + se3theta
        
        R = PoEUtils.MatrixExp3(omega_skew * theta)
        G_theta = (np.eye(3) * theta + (1 - np.cos(theta)) * omega_skew + 
                   (theta - np.sin(theta)) * np.dot(omega_skew, omega_skew)) / theta
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.dot(G_theta, v / theta) # v는 이미 theta가 곱해진 상태가 아님을 주의 (구현상 편의를 위해 수정)
        
        # 다시 정확한 수식: MatrixExp6(S*theta)
        # S*theta에서 w*theta, v*theta를 추출해야 함.
        # 위 로직을 단순화하여 MatrixExp6(Make_se3(S)*theta) 형태로 사용
        return T 
    
    @staticmethod
    def Adjoint(T):
        """변환 행렬 T의 Adjoint Map (6x6)"""
        R = T[:3, :3]
        p = T[:3, 3]
        p_skew = PoEUtils.Skew(p)
        AdT = np.zeros((6, 6))
        AdT[:3, :3] = R
        AdT[3:, :3] = np.dot(p_skew, R)
        AdT[3:, 3:] = R
        return AdT

# -----------------------------------------------------------
# 2. Robot Kinematics Class (User Request: PoE, FK, Jacobian)
# -----------------------------------------------------------
class RobotKinematics:
    def __init__(self):
        # XML 모델에 맞춘 Screw Axis (S) 정의
        # Base Frame: L_joint1 위치
        # L_joint1: (0,0,0), Axis x(1,0,0)
        # L_joint2: (0, 0.1, 0), Axis y(0,1,0)
        # L_joint3: (0, 0.1, -0.3), Axis y(0,1,0) (Link2 길이 고려)
        
        # S = [omega; v], v = -omega x q
        
        # Joint 1
        w1 = np.array([1, 0, 0])
        q1 = np.array([0, 0, 0])
        v1 = -np.cross(w1, q1)
        S1 = np.concatenate([w1, v1])

        # Joint 2
        w2 = np.array([0, 1, 0])
        q2 = np.array([0, 0.1, 0])
        v2 = -np.cross(w2, q2)
        S2 = np.concatenate([w2, v2])

        # Joint 3
        w3 = np.array([0, 1, 0])
        q3 = np.array([0, 0.1, -0.3]) # L_link2가 아래로 0.3만큼 내려감
        v3 = -np.cross(w3, q3)
        S3 = np.concatenate([w3, v3])

        self.Slist = np.array([S1, S2, S3]).T # 6x3 Matrix

        # Home Configuration (M): 모든 q=0일 때 End-Effector의 위치와 자세
        # L_link3의 끝부분(tip)이라고 가정 (길이 0.3)
        # Tip Position: (0, 0.1, -0.6)
        self.M = np.eye(4)
        self.M[:3, 3] = np.array([0, 0.1, -0.6])

    def FK_PoE(self, theta_list):
        """Forward Kinematics using Product of Exponentials"""
        T = np.eye(4)
        for i in range(len(theta_list)):
            se3 = PoEUtils.VecTose3(self.Slist[:, i] * theta_list[i])
            # MatrixExp6 구현의 단순화를 위해 scipy 등을 안쓰고 직접 근사 혹은 로드리게스 사용
            # 여기서는 편의상 helper함수 대신 개념적으로 작성합니다.
            # 실제로는 정밀한 Matrix Exp 라이브러리가 필요합니다.
            
            # (약식 구현) 회전과 위치 분리 계산
            w = self.Slist[:3, i]
            v = self.Slist[3:, i]
            th = theta_list[i]
            
            # Rodrigues for Rotation
            w_skew = PoEUtils.Skew(w)
            R = np.eye(3) + np.sin(th)*w_skew + (1-np.cos(th))*np.dot(w_skew, w_skew)
            
            # G matrix for Position
            G = np.eye(3)*th + (1-np.cos(th))*w_skew + (th-np.sin(th))*np.dot(w_skew, w_skew)
            p = np.dot(G, v)
            
            T_i = np.eye(4)
            T_i[:3, :3] = R
            T_i[:3, 3] = p
            
            T = np.dot(T, T_i)
            
        T_final = np.dot(T, self.M)
        return T_final

    def SpaceJacobian(self, theta_list):
        """Calculate Space Jacobian using PoE"""
        Js = np.zeros((6, 3))
        T = np.eye(4)
        Js[:, 0] = self.Slist[:, 0]
        
        for i in range(1, 3):
            # 이전 조인트까지의 변환 행렬 계산
            w = self.Slist[:3, i-1]
            v = self.Slist[3:, i-1]
            th = theta_list[i-1]
            
            w_skew = PoEUtils.Skew(w)
            R = np.eye(3) + np.sin(th)*w_skew + (1-np.cos(th))*np.dot(w_skew, w_skew)
            G = np.eye(3)*th + (1-np.cos(th))*w_skew + (th-np.sin(th))*np.dot(w_skew, w_skew)
            p = np.dot(G, v)
            
            T_i = np.eye(4)
            T_i[:3, :3] = R
            T_i[:3, 3] = p
            
            T = np.dot(T, T_i)
            
            # Adjoint(T) * S_i
            AdT = PoEUtils.Adjoint(T)
            Js[:, i] = np.dot(AdT, self.Slist[:, i])
            
        return Js

# -----------------------------------------------------------
# 3. Main Simulation
# -----------------------------------------------------------

xml = """
<mujoco model="biped_triple_pendulum">
    <compiler angle="radian" coordinate="local" inertiafromgeom="false"/>
    <option gravity="0 0 -9.81" integrator="RK4" timestep="0.002"/>
    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="2 2 0.1" rgba=".9 .9 .9 1"/>
        <body name="base_link" pos="0 0 2.0">
            <inertial pos="0 0 0" mass="2.0" diaginertia="0.01 0.01 0.01"/>
            <geom type="box" size="0.2 0.15 0.05" rgba="0.2 0.2 0.2 1"/>
            <body name="L_link1" pos="0 0.15 0">
                <inertial pos="0 0.05 0" mass="0.5" diaginertia="0.001 0.001 0.001"/>
                <joint name="L_joint1" type="hinge" axis="1 0 0" pos="0 0 0"/> 
                <geom type="capsule" fromto="0 0 0 0 0.1 0" size="0.03" rgba="1 0 0 1"/>
                <body name="L_link2" pos="0 0.1 0">
                    <inertial pos="0 0 -0.15" mass="1.0" diaginertia="0.005 0.005 0.001"/>
                    <joint name="L_joint2" type="hinge" axis="0 1 0" pos="0 0 0"/> 
                    <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" rgba="0 1 0 1"/>
                    <body name="L_link3" pos="0 0 -0.3">
                        <inertial pos="0 0 -0.15" mass="0.8" diaginertia="0.004 0.004 0.001"/>
                        <joint name="L_joint3" type="hinge" axis="0 1 0" pos="0 0 0"/> 
                        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" rgba="0 0 1 1"/>
                        <site name="tip" pos="0 0 -0.3" size="0.02" rgba="1 1 0 1"/>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>
    <actuator>
        <motor name="motor_L1" joint="L_joint1" gear="1"/>
        <motor name="motor_L2" joint="L_joint2" gear="1"/>
        <motor name="motor_L3" joint="L_joint3" gear="1"/>
    </actuator>
    <sensor>
        <actuatorfrc name="Torque_L1" actuator="motor_L1"/>
        <actuatorfrc name="Torque_L2" actuator="motor_L2"/>
        <actuatorfrc name="Torque_L3" actuator="motor_L3"/>
        <framepos name="tip_pos" objtype="site" objname="tip"/>
    </sensor>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
kinematics = RobotKinematics()

# 목표 위치 (Task Space Target)
target_pos = np.array([0.2, 0.4, -0.4]) # Base Frame 기준 목표

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    
    # Mass Matrix 버퍼 미리 할당
    M_mass = np.zeros((model.nv, model.nv))
    
    while viewer.is_running():
        step_start = time.time()
        
        # 1. 현재 상태 읽기
        q = data.qpos[:3] # Joint Positions
        qdot = data.qvel[:3] # Joint Velocities
        
        # 2. Forward Kinematics (FK) 계산 - PoE 사용
        T_curr = kinematics.FK_PoE(q)
        current_pos = T_curr[:3, 3] # End-effector position
        
        # 3. Jacobian (J) 계산 - PoE 사용
        # Space Jacobian이므로 회전 변환 필요할 수 있으나, 여기서는 위치 제어만 수행
        J_space = kinematics.SpaceJacobian(q)
        J_pos = J_space[3:, :] # Linear velocity part (3x3)
        
        # 4. Dynamics: Mass Matrix & Gravity 추출
        mujoco.mj_fullM(model, M_mass, data.qM) # Mass Matrix (Joint Space)
        # qfrc_bias는 Coriolis + Gravity + Spring 힘을 포함함
        # 순수 Gravity Compensation을 위해 사용
        bias_torque = data.qfrc_bias[:3]
        
        # 5. Task Space Control (정확한 토크 계산)
        # 공식: tau = J^T * ( Kp * error + Kd * error_dot ) + g(q)
        # 혹은 Inertia Decoupling: tau = J^T * (Lambda * (acc_des)) + bias + J^T*F_external
        
        # PD Control Gains
        Kp = 20.0
        Kd = 1.0
        
        # Error Calculation
        pos_error = target_pos - current_pos
        vel_error = -np.dot(J_pos, qdot) # Target velocity is 0
        
        # Desired Force in Cartesian Space (가상의 스프링-댐퍼 힘)
        F_des = Kp * pos_error + Kd * vel_error
        
        # Torque Calculation (Jacobian Transpose Control)
        # J_pos.T (3x3) dot F_des (3) -> (3)
        tau_task = np.dot(J_pos.T, F_des)
        
        # Final Torque = Task Torque + Gravity/Coriolis Compensation
        # 이것이 바로 "Exact Torque"를 만들어내는 핵심 식입니다.
        tau_cmd = tau_task + bias_torque
        
        # Apply Torque
        data.ctrl[:3] = tau_cmd
        
        # 6. 시각화 (Target & Current)
        # Target 위치에 녹색 구 그리기
        viewer.user_scn.ngeom = 0
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.05, 0, 0],
            pos=target_pos + np.array([0, 0, 2.0]), # Base body offset(2.0) 고려 (Global 좌표로 변환)
            mat=np.eye(3).flatten(),
            rgba=[0, 1, 0, 0.5] # Green Ghost
        )
        viewer.user_scn.ngeom = 1
        
        mujoco.mj_step(model, data)
        viewer.sync()

        # 실시간 유지
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)