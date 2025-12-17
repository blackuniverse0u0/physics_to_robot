import gymnasium as gym
import gymnasium.vector as gym_vector
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import argparse
import os
from collections import deque
import wandb
import time

MUJOCO_ENVS = [
    "Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Humanoid-v5",
    "InvertedDoublePendulum-v5", "InvertedPendulum-v5",
    "Reacher-v5", "Walker2d-v5"
]


# --- 하이퍼파라미터 ---
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPSILON = 0.2
PPO_EPOCHS = 10
NUM_MINIBATCHES = 32
N_STEPS = 2048
ENTROPY_COEF = 0.01  # 0.0으로 설정하는 경우도 많지만, 탐험을 위해 소량 유지
VF_COEF = 0.5        # Value Function Loss 계수 추가
MAX_GRAD_NORM = 0.5  # Gradient Clipping 추가
MAX_TRAINING_TIMESTEPS = 2_000_000 # 테스트를 위해 약간 줄임 (조절 가능)
SEED = 1
CHECK_INTERVAL_STEPS = 50000
EARLY_STOPPING_PATIENCE = 50 # 윈도우가 크므로 patience 늘림
EARLY_STOPPING_WINDOW = 50   # 윈도우 크기 조절

# --- [개선 1] 환경 생성 및 래퍼 유틸리티 ---
def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(gym_id, render_mode="rgb_array")
        else:
            env = gym.make(gym_id)
        
        # 에피소드 통계 기록
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            
        env = gym.wrappers.ClipAction(env)
        # 관측값 정규화 (중요)
        env = gym.wrappers.NormalizeObservation(env)
        # 관측값 클리핑 (오류 수정됨: observation_space 전달)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
        # 보상 정규화 (중요)
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

# --- [개선 2] 가중치 초기화 유틸리티 ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# --- 네트워크 정의 ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # [개선] Orthogonal Initialization 적용
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01), # 액션 레이어는 작은 std로 초기화
        )
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        mean = self.net(state)
        std = torch.exp(self.log_std.expand_as(mean))
        dist = Normal(mean, std)
        return dist

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        # [개선] Orthogonal Initialization 적용
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, state):
        return self.net(state)

# --- PPO 에이전트 ---
class PPOAgent:
    def __init__(self, envs, device):
        self.device = device
        # envs 객체에서 차원 정보 추출
        state_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)
        
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        
        # Actor와 Critic 파라미터를 하나의 옵티마이저로 관리 (일반적인 구현)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), 
            lr=LEARNING_RATE, 
            eps=1e-5
        )

    def get_action_and_value(self, state, action=None):
        dist = self.actor(state)
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(state)
        
        return action, log_prob, entropy, value

    def train(self, data):
        # Flatten된 배치 데이터
        b_states, b_actions, b_log_probs, b_advantages, b_returns, b_values = data
        
        batch_size = b_states.size(0)
        minibatch_size = batch_size // NUM_MINIBATCHES
        
        clipfracs = []
        
        # 인덱스 생성
        b_inds = np.arange(batch_size)
        
        for epoch in range(PPO_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                # 현재 정책으로 다시 계산
                _, newlogprob, entropy, newvalue = self.get_action_and_value(b_states[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_log_probs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # KL Divergence 모니터링 (디버깅용)
                    # old_approx_kl = (-logratio).mean()
                    # approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > PPO_EPSILON).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                
                # Policy Loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - PPO_EPSILON, 1 + PPO_EPSILON)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss (Clipped) - 안정성을 위해 Clipped Value Loss 사용
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -PPO_EPSILON,
                    PPO_EPSILON,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                # Entropy Loss
                entropy_loss = entropy.mean()
                
                # Total Loss
                loss = pg_loss - ENTROPY_COEF * entropy_loss + v_loss * VF_COEF

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), 
                    MAX_GRAD_NORM
                )
                self.optimizer.step()
        
        return pg_loss.item(), v_loss.item(), loss.item() # 마지막 배치 기준 반환

    def save_model(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


def main():
    parser = argparse.ArgumentParser(description="Vectorized PPO for Massively Parallel Training")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'], help='train or inference')
    parser.add_argument('--env', type=str, default='HalfCheetah-v4', help='MuJoCo environment ID')
    parser.add_argument('--num-envs', type=int, default=1, help='Number of parallel environments')
    parser.add_argument('--capture-video', type=lambda x: x.lower() == 'true', default=False, help='Capture video')
    args = parser.parse_args()

    run_name = f"{args.env}_PPO_Seed{SEED}_{int(time.time())}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- [모드: 학습] ---
    if args.mode == 'train':
        wandb.init(
            project="mujoco_improved_ppo", name=run_name,
            config=vars(args)
        )

        # [개선] SyncVectorEnv + make_env 사용 (wrappers 적용됨)
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env, SEED + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
        )
        
        # Seeding
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

        agent = PPOAgent(envs, device)
        
        save_dir = "checkpoints"
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{args.env}_best.pt")

        # Storage
        obs = torch.zeros((N_STEPS, args.num_envs) + envs.single_observation_space.shape).to(device)
        actions = torch.zeros((N_STEPS, args.num_envs) + envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((N_STEPS, args.num_envs)).to(device)
        rewards = torch.zeros((N_STEPS, args.num_envs)).to(device)
        dones = torch.zeros((N_STEPS, args.num_envs)).to(device)
        values = torch.zeros((N_STEPS, args.num_envs)).to(device)

        global_step = 0
        best_avg_reward = -np.inf
        
        # Start Game
        next_obs, _ = envs.reset(seed=SEED)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)
        
        num_updates = MAX_TRAINING_TIMESTEPS // (N_STEPS * args.num_envs)
        start_time = time.time()

        for update in range(1, num_updates + 1):
            # [개선] Learning Rate Annealing
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * LEARNING_RATE
            agent.optimizer.param_groups[0]["lr"] = lrnow

            # 1. 데이터 수집 (Rollout)
            for step in range(N_STEPS):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                
                actions[step] = action
                logprobs[step] = logprob

                # Env Step
                next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
                done = terminated | truncated
                
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs = torch.Tensor(next_obs).to(device)
                next_done = torch.Tensor(done).to(device)

                # 로깅
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            ep_ret = info['episode']['r']
                            print(f"Global Step: {global_step}, Ep Reward: {ep_ret:.2f}")
                            wandb.log({"episodic_return": ep_ret, "episodic_length": info["episode"]["l"]}, step=global_step)
                            
                            # 모델 저장 (단순화: 에피소드 보상 갱신될 때마다 체크)
                            if ep_ret > best_avg_reward:
                                best_avg_reward = ep_ret
                                agent.save_model(model_path)

            # 2. Advantage 계산 (GAE) - [개선] CleanRL 방식 적용 (매우 중요)
            with torch.no_grad():
                next_value = agent.critic(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(N_STEPS)):
                    if t == N_STEPS - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                returns = advantages + values

            # 3. 데이터 Flatten
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # 4. Advantage Normalization [개선]
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            # 5. 학습 (Train)
            pg_loss, v_loss, total_loss = agent.train((b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values))

            sps = int(global_step / (time.time() - start_time))
            wandb.log({
                "losses/policy_loss": pg_loss, 
                "losses/value_loss": v_loss, 
                "charts/SPS": sps,
                "charts/learning_rate": lrnow
            }, step=global_step)

        envs.close()
        wandb.finish()

    # --- [모드: 추론] ---
    elif args.mode == 'inference':
        # 추론 시에는 Normalize Wrapper가 중요합니다. 학습 때 저장된 통계가 있어야 정확하지만, 
        # 여기서는 단순화를 위해 동일한 구조의 Env를 만들고 실행합니다.
        # 주의: 실제 배포시에는 NormalizeObservation의 running_mean/var를 로드해야 성능이 나옵니다.
        
        env = gym.make(args.env, render_mode='human')
        # 추론 시에도 입력 정규화는 필요하지만, 통계(mean/std)가 업데이트되면 안됩니다.
        # 여기서는 편의상 원본 env 사용 (정규화된 모델을 원본 env에 쓰면 성능이 떨어질 수 있음)
        # 제대로 하려면 VecNormalize 등을 저장하고 불러와야 함.
        
        agent = PPOAgent(gym.vector.SyncVectorEnv([lambda: env]), device)
        model_path = f"checkpoints/{args.env}_best.pt"
        
        if os.path.exists(model_path):
            agent.load_model(model_path)
            print("Model loaded.")
        else:
            print("No model found.")
            return

        state, _ = env.reset()
        done = False
        while not done:
            state_tensor = torch.Tensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(state_tensor)
            
            state, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
            done = terminated or truncated
            env.render()
        env.close()

if __name__ == '__main__':
    # 임시: import random 필요
    import random
    main()