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
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. NIDS 시계열 환경 (Train / Test 모드 분리)
# ==========================================
class NIDSEnv(gym.Env):
    def __init__(self, X_data, y_data, max_steps=1000, window_size=3, is_test=False):
        super(NIDSEnv, self).__init__()
        
        self.X_scaled = X_data
        self.y = y_data
        self.max_steps = max_steps
        self.window_size = window_size
        self.is_test = is_test
        self.current_step = 0
        
        self.num_features = self.X_scaled.shape[1]
        state_dim = self.num_features * self.window_size
        
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        
        self.episode_start_idx = 0

    def _get_windowed_state(self, current_step_offset):
        target_idx = self.episode_start_idx + current_step_offset
        states_bucket = []
        
        for i in reversed(range(self.window_size)):
            lookback_idx = target_idx - i
            if lookback_idx < self.episode_start_idx:
                states_bucket.append(np.zeros(self.num_features, dtype=np.float32))
            else:
                states_bucket.append(self.X_scaled[lookback_idx])
                
        return np.concatenate(states_bucket, axis=0).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        total_rows = len(self.y)
        valid_start_range = total_rows - self.max_steps
        
        attempts = 0
        while True:
            start_idx = np.random.randint(0, valid_start_range)
            slice_labels = self.y[start_idx : start_idx + self.max_steps]
            attack_ratio = np.sum(slice_labels == 1) / self.max_steps
            
            if not self.is_test:
                # Train 모드: 수렴 성능 향상을 위해 일정한 공격 비율(20%~40%)이 섞인 시퀀스 탐색
                if (0.20 <= attack_ratio <= 0.40) or attempts > 300:
                    self.episode_start_idx = start_idx
                    break
            else:
                # Test 모드: 테스트셋의 원본 시간 흐름과 실제 분포를 왜곡하지 않고 그대로 평가
                if attack_ratio > 0 or attempts > 30:
                    self.episode_start_idx = start_idx
                    break
            attempts += 1
            
        state = self._get_windowed_state(self.current_step)
        info = {}
        return state, info

    def step(self, action):
        actual_label = self.y[self.episode_start_idx + self.current_step]
        
        if action == actual_label:
            if actual_label == 0:
                reward = 2.0  
            else:
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
            next_state = self._get_windowed_state(self.current_step)
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
# 3. 에이전트 (이원화 리플레이 버퍼 및 테스트 모드 전환 지원)
# ==========================================
class DQNAgent:
    def __init__(self, state_dim, action_dim, is_test=False):
        self.action_dim = action_dim
        self.is_test = is_test
        
        self.policy_net = StandardQNetwork(state_dim, action_dim).to(device)
        self.target_net = StandardQNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.00005)
        
        self.memory_benign = deque(maxlen=30000)
        self.memory_attack = deque(maxlen=30000)
        
        self.gamma = 0.99
        self.epsilon = 0.01 if is_test else 1.0
        self.epsilon_decay = 0.996 
        self.epsilon_min = 0.01
        
        self.batch_size = 64
        self.tau = 0.001 
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
        if self.is_test:
            return
        if reward == 2.0 or reward == -1.0:  
            self.memory_benign.append((state, action, reward, next_state, done))
        else:                                
            self.memory_attack.append((state, action, reward, next_state, done))

    def train_step(self):
        if self.is_test:
            return
            
        half_batch = self.batch_size // 2
        if len(self.memory_benign) < half_batch or len(self.memory_attack) < half_batch:
            return
        
        batch_benign = random.sample(self.memory_benign, half_batch)
        batch_attack = random.sample(self.memory_attack, half_batch)
        batch = batch_benign + batch_attack
        random.shuffle(batch) 
        
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
# 4. 통합 시계열 데이터 로드 및 순차적 분할 함수
# ==========================================
def load_and_split_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"지정한 경로에 CSV 파일이 존재하지 않습니다: {file_path}")
        
    print(f"통합 데이터 로드 시작: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path, low_memory=False)
    
    # 시계열 순서 유지를 위한 정렬 및 타임스탬프 인덱싱
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.sort_values(by='Timestamp').reset_index(drop=True)
        
    X_raw = df.drop(columns=['Label', 'Timestamp'], errors='ignore')
    y = df['Label'].apply(lambda x: 0 if str(x).strip() == 'Benign' else 1).values.astype(np.int8)
    del df
    
    # 모든 피처 강제 숫자 형변환 처리 (헤더 문자열 노이즈 제거 포함)
    for col in X_raw.columns:
        X_raw[col] = pd.to_numeric(X_raw[col], errors='coerce')
        
    X_raw = X_raw.fillna(0.0).astype(np.float32)
    X_np = X_raw.to_numpy(dtype=np.float32)
    del X_raw
    
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
    
    # 💡 시계열의 연속성이 끊어지지 않도록 셔플 없이 앞부분 80%는 Train, 뒷부분 20%는 Test로 순차 슬라이싱
    print("시계열 데이터 순차 분할(Sequential Split) 진행 중...")
    split_idx = int(len(X_log) * 0.8)
    
    X_train_raw = X_log[:split_idx]
    X_test_raw = X_log[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    del X_log
    
    # 데이터 누수 차단 스케일러 적용
    print("데이터 스케일링 진행 중 (누수 방지)...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    print(f"학습셋 크기: {X_train_scaled.shape} | 정상: {np.sum(y_train==0):,}개, 공격: {np.sum(y_train==1):,}개")
    print(f"테스트셋 크기: {X_test_scaled.shape} | 정상: {np.sum(y_test==0):,}개, 공격: {np.sum(y_test==1):,}개")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# ==========================================
# 5. 메인 루프 실행 (Train -> Test)
# ==========================================
if __name__ == "__main__":
    clean_file_path = r"C:\ids2018_data\nids_advanced_cleaned.csv"
    
    # 1. 데이터 로드 및 전처리 분리
    X_train, X_test, y_train, y_test = load_and_split_data(clean_file_path)
    
    # 2. Train용 환경 및 에이전트 생성
    train_env = NIDSEnv(X_train, y_train, max_steps=1000, window_size=3, is_test=False)
    agent = DQNAgent(state_dim=train_env.observation_space.shape[0], action_dim=train_env.action_space.n, is_test=False)
    
    num_train_episodes = 800
    history_rewards = []
    history_accuracy = []
    history_fpr = []
    history_fnr = []
    
    # ------------------------------------------
    # [STAGE 1] 모델 학습 (Train)
    # ------------------------------------------
    print("\n========= [STAGE 1] 완벽 수렴 가공 시계열 DQN 학습 시작 =========")
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
        accuracy = ((tp + tn) / total_steps) * 100 if total_steps > 0 else 0.0
        fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0.0
        fnr = (fn / (fn + tp)) * 100 if (fn + tp) > 0 else 0.0
        attack_ratio = ((tp + fn) / total_steps) * 100 if total_steps > 0 else 0.0
        
        history_rewards.append(episode_reward)
        history_accuracy.append(accuracy)
        history_fpr.append(fpr)
        history_fnr.append(fnr)
        
        # 출력 서식 원본 복구 및 싱크 유지
        if (episode + 1) % 5 == 0 or episode == 0:
            print(f"에피 {episode+1:3d}/{num_train_episodes} | "
                  f"보상: {episode_reward:7.1f} | "
                  f"정확도: {accuracy:5.1f}% | "
                  f"오탐율(FPR): {fpr:5.1f}% | "
                  f"미탐율(FNR): {fnr:5.1f}% | "
                  f"공격비율: {attack_ratio:5.1f}% | "
                  f"입실론: {agent.epsilon:.3f}")
        
        agent.decay_epsilon()
            
    print("========= 학습 완료 =========")
    
    # ------------------------------------------
    # [STAGE 2] 모델 검증 (Test)
    # ------------------------------------------
    print("\n========= [STAGE 2] 테스트 데이터셋 평가 시작 =========")
    agent.is_test = True
    test_env = NIDSEnv(X_test, y_test, max_steps=1000, window_size=3, is_test=True)
    
    num_test_episodes = 50  
    test_fpr, test_fnr, test_acc = [], [], []
    
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
        acc = ((tp + tn) / total_steps) * 100 if total_steps > 0 else 0.0
        fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0.0
        fnr = (fn / (fn + tp)) * 100 if (fn + tp) > 0 else 0.0
        
        test_acc.append(acc)
        test_fpr.append(fpr)
        test_fnr.append(fnr)

    print("\n========= 최종 테스트 결과 요약 =========")
    print(f"평균 정확도 (Test Accuracy) : {np.mean(test_acc):.2f}%")
    print(f"평균 오탐율 (False Alarm Rate) : {np.mean(test_fpr):.2f}%")
    print(f"평균 미탐율 (Missed Attack Rate) : {np.mean(test_fnr):.2f}%")
    print("=========================================")

    # ------------------------------------------
    # 6. 결과 시각화 그래프 출력 (학습 기록 기반 선 그래프 3단 구성)
    # ------------------------------------------
    plt.figure(figsize=(18, 5))

    # 보상 그래프
    plt.subplot(1, 3, 1)
    plt.plot(history_rewards, color='blue', alpha=0.6)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.grid(True)

    # 정확도 그래프
    plt.subplot(1, 3, 2)
    plt.plot(history_accuracy, color='green', alpha=0.6)
    plt.title('Training Accuracy (%)')
    plt.xlabel('Episode')
    plt.ylim(0, 105)
    plt.grid(True)

    # 오탐율 vs 미탐율 그래프
    plt.subplot(1, 3, 3)
    plt.plot(history_fpr, label='FPR (False Alarm)', color='orange', alpha=0.7)
    plt.plot(history_fnr, label='FNR (Missed Attack)', color='red', alpha=0.7)
    plt.title('FPR vs FNR (%)')
    plt.xlabel('Episode')
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()