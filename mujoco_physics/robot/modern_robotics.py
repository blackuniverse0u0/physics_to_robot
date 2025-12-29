import numpy as np 

def skew(w):
    skew_w = np.array([
        [0,-w[2],w[1]],
        [w[2],0,-w[0]],
        [-w[1],w[0],0]
    ])
    return skew_w  

def vec_to_so3(skew_w,theta):
    return np.eye(3) + np.sin(theta)*skew_w + (1-np.cos(theta))*skew_w@skew_w


def vec_to_se3(S, theta):
    """rodrigues formula
    Modern Robotics: MatrixExp6 구현
    S: Screw Axis [omega, v] (R^6)
    theta: Joint angle (rad), skew angle
    """
    # 회전축 (Rotation axis / Angular component)
    w = S[:3] # TODO: Rotation matrix를 만들기 위한 screw axis
    
    # 스크류 축의 선속도 성분 (Linear component of the screw axis) -> 주의: 최종 이동 벡터(p)가 아님!
    v = S[3:] # TODO: translation vector와 같은가? 틀리다!!!! linear vel이다!
    # v = - w x p 이다. 
    
    
    # 0 근처 처리 (회전이 거의 없는 경우 대비)
    if np.linalg.norm(w) < 1e-6:
        T = np.eye(4)
        T[:3, 3] = v * theta
        return T

    skew_w = skew(w)
    I = np.eye(3)
    
    # Rodrigues Formula for R
    R = I + np.sin(theta) * skew_w + (1 - np.cos(theta)) * (skew_w @ skew_w)
    # G matrix for translation component p
    G = I * theta + (1 - np.cos(theta)) * skew_w + (theta - np.sin(theta)) * (skew_w @ skew_w)
    p = G @ v

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T



# chapter 6 : velocity kinematics & forward dynamics
# q와 q_dot을 통해 twist의 jacobian을 계산하자. 
# end effector twist 6 dim vector

# x(t)t = f(theta(t)) : R^m

# x_dot = J(theta) @ theta_dot : R^mxn @ R^n

# f^T@x_dot = torque^T @ theta_dot

# torque = J(theta)^T @ f 
# f = [J(theta)^T]^-1 @ torque

# spcae jacobian vs body jacobian
# base 관점 

# E = np.zeros((4, 3))
# E[:3,3] = 1

# x_b = T_be@M@E  
# T_be = exp([S1]theta_1)@...@exp([Sn]theta_n)@M

# screw axis 와 twist의 차이 
# v = S * theta_dot
# 

def twist_matrix(T,T_dot): # R^4x4
    body_twist = np.linalg.inv(T)@T_dot # se(3)
    spatial_twist = T_dot@np.linalg.inv(T)    # se(3)
    
    # [[wb] vb]
    # [ 0   0 ]
    return body_twist,spatial_twist

def twist_vec(w,v): # R^6
    return np.concatenate((w,v))

def adjoint(T):
    R = T[:3,:3]
    p = T[:3,3]
    
    adj = np.eye(6)
    
    adj[:3,:3] = R 
    adj[3:,:3] = skew(p)@R 
    adj[3:,3:] = R
    return adj 
    
#명제 3.5
# R[w]R^T = [Rw]
# [w]p = -[p]w
# TODO : adjoint는 왜 필요한가? 속도기구학, 정역학
def adjoint(T):
    """
    [추가됨] Jacobian 계산을 위한 Adjoint 변환 행렬 (6x6)
    Ad_T = [ R    0 ]
            [ [p]R R ]
    """
    R = T[:3, :3]
    p = T[:3, 3]
    
    adj = np.zeros((6, 6))
    adj[:3, :3] = R
    adj[3:, 3:] = R
    adj[3:, :3] = skew(p) @ R
    return adj

def space_jacobian(S_lists:np.array,thetas:list):
    # Js = [S1,Ad_T1(S2),...]
    Js = np.zeros((6,len(S_lists)))
    T = np.eye(4)
    
    Js[:,0] = S_lists[0]
    for i in range(1,len(S_lists)):
        T = T@vec_to_se3(S_lists[i-1],thetas[i-1])
        Js[:,i] = adjoint(T)@S_lists[i]
    return Js 
        
    
def get_space_jacobian(S_list, thetas):
    """
    [재작성] Space Jacobian J_s 계산 (6xN Matrix)
    
    Js = [S1, Ad_T1(S2), Ad_T1T2(S3), ... ]
    """
    Js = np.zeros((6, len(S_list)))
    T = np.eye(4)
    
    # 첫 번째 컬럼은 항상 첫 번째 Screw Axis 그 자체
    Js[:, 0] = S_list[0]
    
    # 두 번째 컬럼부터 누적 변환 적용
    for i in range(1, len(S_list)):
        # T_i-1 계산: 이전 관절까지의 변환 행렬 누적
        T = T @ vec_to_se3(S_list[i-1], thetas[i-1])
        
        # Adjoint 변환을 통해 현재 Screw를 Space Frame으로 변환하여 저장
        Js[:, i] = adjoint(T) @ S_list[i]
        
    return Js 

import numpy as np 
import math 



class ScrewTheory:
    def __init__(self): 
        pass 
    
    # twist 
    
    # wrench 
    
    # torque 
    
    # power 

    # fk
    
    # ik 
    
from abc import ABC, abstractmethod

class RobotMath:
    @staticmethod
    # def skew(w):

    # w is what? angular velocity 
    # v = w x r 
    
    def vec2skew(v):
        # Ensure v is a 3-element vector
        assert len(v) == 3, "Input must be a 3-element vector"

        # Create the skew-symmetric matrix
        V = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

        return V        
    
    @staticmethod
    def adjoint(T): ...

    
class BaseRobot(ABC):
    def __init__(self, n_joints):
        self.n_joints = n_joints

    @abstractmethod
    def forward_kinematics(self, thetas):
        """Must return the SE(3) matrix of the EE"""
        pass

    @abstractmethod
    def get_jacobian(self, thetas):
        """Must return the 6xN Jacobian matrix"""
        pass
    
    
    