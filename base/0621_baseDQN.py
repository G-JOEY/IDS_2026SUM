# Base DQN Code
# Action: 0(Benign), 1(Attack)
# Reward: +1.0 / -5.0(미탐: Attack을 Benign으로 분류) / -1.0(오탐: Benign을 Attack으로 분류)
# 학습 시 클래스 불균형 문제를 해결하기 위해 정상과 공격 데이터를 항상 50:50 균등 추출
# 4개 층의 MLP 구조를 가진 심층 신경망을 이용하여 Q-value를 예측하고 학습
# Focal Loss: 단순 MSE Loss가 아닌, 어려운 샘플에 가중치를 주는 방법
# Soft Update, epsilon-greedy
# 문자열 레이블 정수 변환, 결측치/무한대 값 처리, log1p 스케일링 후 MinMaxScaler를 적용해 정규화
# Train/Test 셋 분할 시 원본 데이터의 클래스 비율을 유지


import os
import random
import numpy as np
import pandas as pd
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
from gymnasium import spaces
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. NIDS 강화학습 환경
# ==========================================
class NIDSEnv(gym.Env):
    def __init__(self, X_data, y_data, max_steps=1000, is_test=False):
        super(NIDSEnv, self).__init__()
        
        self.X_scaled = X_data
        self.y = y_data
        self.max_steps = max_steps
        self.is_test = is_test
        self.current_step = 0
        
        # 인덱스 분리
        self.benign_indices = np.where(self.y == 0)[0]
        self.attack_indices = np.where(self.y == 1)[0]
        
        self.episode_indices = []
        
        num_features = self.X_scaled.shape[1]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(num_features,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Train 모드일 때는 50:50 균형 샘플링 (클래스 불균형 해결)
        if not self.is_test:
            half_steps = self.max_steps // 2
            sampled_benign = np.random.choice(self.benign_indices, half_steps, replace=True)
            sampled_attack = np.random.choice(self.attack_indices, half_steps, replace=True)
            self.episode_indices = np.concatenate([sampled_benign, sampled_attack])
            np.random.shuffle(self.episode_indices)
        else:
            # Test 모드일 때는 테스트셋 내 실제 비율 반영하여 추출
            available_indices = np.arange(len(self.y))
            if len(available_indices) >= self.max_steps:
                self.episode_indices = np.random.choice(available_indices, self.max_steps, replace=False)
            else:
                self.episode_indices = np.random.choice(available_indices, self.max_steps, replace=True)
        
        state = self.X_scaled[self.episode_indices[self.current_step]].astype(np.float32)
        info = {}
        return state, info

    def step(self, action):
        actual_label = self.y[self.episode_indices[self.current_step]]
        
        if action == actual_label:
            reward = 1.0  
        else:
            if actual_label == 1 and action == 0:
                reward = -5.0  # 미탐 패널티
            else:
                reward = -1.0  # 오탐 패널티
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        if not terminated:
            next_state = self.X_scaled[self.episode_indices[self.current_step]].astype(np.float32)
        else:
            next_state = np.zeros(self.observation_space.shape, dtype=np.float32)
            
        info = {"actual": actual_label, "predicted": action}
        return next_state, reward, terminated, truncated, info

# ==========================================
# 2. DQN 아키텍처
# ==========================================
class StandardQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(StandardQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, x):
        return self.network(x)

# ==========================================
# 3. 에이전트
# ==========================================
class DQNAgent:
    def __init__(self, state_dim, action_dim, is_test=False):
        self.action_dim = action_dim
        self.is_test = is_test
        
        self.policy_net = StandardQNetwork(state_dim, action_dim).to(device)
        self.target_net = StandardQNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.00025)
        self.memory = deque(maxlen=60000)
        
        self.gamma = 0.99
        self.epsilon = 0.01 if is_test else 1.0  # 테스트 시에는 무작위 탐색 최소화
        
        self.epsilon_decay = 0.996  
        self.epsilon_min = 0.01
        
        self.batch_size = 64
        self.tau = 0.005  
        self.focal_gamma = 2.0

    def select_action(self, state):
        if not self.is_test and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).to(device)
                q_values = self.policy_net(state_t)
                return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        if not self.is_test:  
            self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if self.is_test or len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states_t = torch.FloatTensor(np.array(states)).to(device)
        actions_t = torch.LongTensor(actions).view(-1, 1).to(device)
        rewards_t = torch.FloatTensor(rewards).view(-1, 1).to(device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(device)
        dones_t = torch.FloatTensor(dones).view(-1, 1).to(device)
        
        current_q = self.policy_net(states_t).gather(1, actions_t)
        
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(1)[0].view(-1, 1)
            target_q = rewards_t + (self.gamma * max_next_q * (1 - dones_t))
            
        td_error = torch.abs(current_q - target_q)
        p_t = torch.exp(-td_error)
        focal_weight = (1 - p_t) ** self.focal_gamma
        
        loss = (focal_weight * nn.MSELoss(reduction='none')(current_q, target_q)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ==========================================
# 4. 데이터 로드 및 층화 추출 (Stratified Split) 고도화
# ==========================================
def load_and_split_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"지정한 경로에 CSV 파일이 존재하지 않습니다: {file_path}")
        
    print(f"데이터 로드 시작: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path, low_memory=False)
    
    X = df.drop(columns=['Label', 'Timestamp'], errors='ignore')
    X = X.select_dtypes(include=[np.number]).astype(np.float32)
    y = df['Label'].apply(lambda x: 0 if str(x).strip() == 'Benign' else 1).values.astype(np.int8)
    del df
    
    X_np = X.to_numpy(dtype=np.float32)
    X_np[np.isinf(X_np)] = np.nan
    col_means = np.nanmean(X_np, axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)
    inds = np.where(np.isnan(X_np))
    X_np[inds] = np.take(col_means, inds[1])
    
    np.clip(X_np, a_min=0, a_max=None, out=X_np)
    X_log = np.log1p(X_np)
    del X_np
    
    max_float32 = np.finfo(np.float32).max * 0.9
    np.clip(X_log, a_min=None, a_max=max_float32, out=X_log)
    
    print("층화 추출(Stratified Split) 진행 중...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_log, y, test_size=0.2, stratify=y, random_state=42
    )
    del X_log
    
    print("데이터 스케일링 진행 중...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)  
    
    print(f"학습셋 정상: {np.sum(y_train==0):,}개, 공격: {np.sum(y_train==1):,}개")
    print(f"테스트셋 정상: {np.sum(y_test==0):,}개, 공격: {np.sum(y_test==1):,}개")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# ==========================================
# 5. 메인 루프 실행 (Train -> Test)
# ==========================================
if __name__ == "__main__":
    clean_file_path = r"C:\ids2018_data\nids_advanced_cleaned.csv"
    
    # 1. 데이터 로드 및 분리 완료
    X_train, X_test, y_train, y_test = load_and_split_data(clean_file_path)
    
    # 2. Train용 환경 및 에이전트 선언
    train_env = NIDSEnv(X_train, y_train, max_steps=1000, is_test=False)
    agent = DQNAgent(state_dim=train_env.observation_space.shape[0], action_dim=train_env.action_space.n, is_test=False)
    
    num_train_episodes = 800 
    history_rewards = []
    history_accuracy = []
    history_fpr = []
    history_fnr = []
    
    # ------------------------------------------
    # [STAGE 1] 모델 학습 (Train)
    # ------------------------------------------
    print("\n========= [STAGE 1] DQN 기반 NIDS 학습 시작 =========")
    for episode in range(num_train_episodes):
        state, info = train_env.reset()
        episode_reward = 0
        tp, tn, fp, fn = 0, 0, 0, 0
        
        for step in range(train_env.max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()
            
            episode_reward += reward
            state = next_state
            
            act = info["actual"]
            pred = info["predicted"]
            
            if act == 1 and pred == 1: tp += 1
            elif act == 0 and pred == 0: tn += 1
            elif act == 0 and pred == 1: fp += 1
            elif act == 1 and pred == 0: fn += 1
                
            if done:
                break
                
        total_steps = tp + tn + fp + fn
        accuracy = ((tp + tn) / total_steps) * 100 if total_steps > 0 else 0
        fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0.0
        fnr = (fn / (fn + tp)) * 100 if (fn + tp) > 0 else 0.0
        attack_ratio = ((tp + fn) / total_steps) * 100 if total_steps > 0 else 0
        
        history_rewards.append(episode_reward)
        history_accuracy.append(accuracy)
        history_fpr.append(fpr)
        history_fnr.append(fnr)
        
        if (episode + 1) % 20 == 0 or episode == 0:
            print(f"에피소드 {episode+1:3d}/{num_train_episodes} | "
                  f"보상: {episode_reward:7.1f} | "
                  f"정확도: {accuracy:5.1f}% | "
                  f"오탐율(FPR): {fpr:5.1f}% | "
                  f"미탐율(FNR): {fnr:5.1f}% | "
                  f"공격비율: {attack_ratio:5.1f}% | "
                  f"입실론: {agent.epsilon:.3f}")
        
        agent.decay_epsilon()
            
    print("========= 학습 완료 =========")
    
    # ------------------------------------------
    # [STAGE 2] 모델 검증 (Test) - 데이터 분포 출력 추가
    # ------------------------------------------
    print("\n========= [STAGE 2] 테스트 데이터셋 평가 시작 =========")
    agent.is_test = True
    test_env = NIDSEnv(X_test, y_test, max_steps=1000, is_test=True)
    
    num_test_episodes = 50  
    test_fpr, test_fnr, test_acc = [], [], []
    
    # 누적 데이터 분포 확인용 변수
    total_test_benign = 0
    total_test_attack = 0
    
    for episode in range(num_test_episodes):
        state, info = test_env.reset()
        tp, tn, fp, fn = 0, 0, 0, 0
        
        for step in range(test_env.max_steps):
            action = agent.select_action(state)  
            next_state, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            
            state = next_state
            
            act = info["actual"]
            pred = info["predicted"]
            
            if act == 1 and pred == 1: tp += 1
            elif act == 0 and pred == 0: tn += 1
            elif act == 0 and pred == 1: fp += 1
            elif act == 1 and pred == 0: fn += 1
            
            if done:
                break
                
        total_steps = tp + tn + fp + fn
        acc = ((tp + tn) / total_steps) * 100
        fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0.0
        fnr = (fn / (fn + tp)) * 100 if (fn + tp) > 0 else 0.0
        
        # 현재 에피소드 분포 계산 및 누적
        ep_benign = tn + fp
        ep_attack = tp + fn
        total_test_benign += ep_benign
        total_test_attack += ep_attack
        
        test_acc.append(acc)
        test_fpr.append(fpr)
        test_fnr.append(fnr)
        
        # 10개 에피소드마다 실시간 데이터 분포와 결과 출력
        if (episode + 1) % 10 == 0 or episode == 0:
            print(f"테스트 에피소드 {episode+1:2d}/{num_test_episodes} | "
                  f"정상 데이터: {ep_benign:3d}개 | 공격 데이터: {ep_attack:3d}개 | "
                  f"정확도: {acc:5.1f}% | FPR: {fpr:5.1f}% | FNR: {fnr:5.1f}%")

    # 전체 에피소드 누적 기반 데이터 최종 비율 계산
    all_test_steps = total_test_benign + total_test_attack
    avg_benign_ratio = (total_test_benign / all_test_steps) * 100 if all_test_steps > 0 else 0
    avg_attack_ratio = (total_test_attack / all_test_steps) * 100 if all_test_steps > 0 else 0

    print("\n========= 최종 테스트 결과 요약 =========")
    print(f"테스트 전체 데이터 분포 : 정상 {total_test_benign:,}개({avg_benign_ratio:.1f}%) / 공격 {total_test_attack:,}개({avg_attack_ratio:.1f}%)")
    print(f"평균 정확도 (Test Accuracy) : {np.mean(test_acc):.2f}%")
    print(f"평균 오탐율 (False Alarm Rate) : {np.mean(test_fpr):.2f}%")
    print(f"평균 미탐율 (Missed Attack Rate) : {np.mean(test_fnr):.2f}%")
    print("=========================================")

    # ------------------------------------------
    # 6. 결과 시각화 그래프 출력 (Training 기록 기반 3단 그래프)
    # ------------------------------------------
    plt.figure(figsize=(18, 5))

    # 1) 누적 보상 그래프
    plt.subplot(1, 3, 1)
    plt.plot(history_rewards, color='blue', alpha=0.6)
    plt.title('Training Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)

    # 2) 정확도 그래프
    plt.subplot(1, 3, 2)
    plt.plot(history_accuracy, color='green', alpha=0.6)
    plt.title('Training Accuracy (%)')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.ylim(0, 105)
    plt.grid(True)

    # 3) 학습 과정 오탐율 & 미탐율 비교 그래프
    plt.subplot(1, 3, 3)
    plt.plot(history_fpr, label='FPR (False Alarm)', color='orange', alpha=0.7)
    plt.plot(history_fnr, label='FNR (Missed Attack)', color='red', alpha=0.7)
    plt.title('Training FPR vs FNR (%)')
    plt.xlabel('Episode')
    plt.ylabel('Rate (%)')
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()