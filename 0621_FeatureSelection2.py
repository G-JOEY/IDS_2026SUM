# FeatureSelection2 Code
# RFECV(재귀적 특성 제거 및 교차 검증)를 통한 최적 피처 개수 자동 탐색
# ExtraTreesClassifier를 평가 엔진으로 삼아 피처를 하나씩 제거하며 성능을 평가
# 3-Fold 교차 검증을 수행하면서 최고의 정확도를 내는 최적의 특성 개수를 AI가 스스로 찾아내어 선택
# 격리 서브 샘플링: 오직 학습셋(y_train) 내부 영역에서만 최대 30,000개의 샘플을 무작위 추출하여 RFECV를 학습시킴


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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFECV  # 자동으로 최적 개수를 찾는 RFECV
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
        
        self.final_n_features = self.X_scaled.shape[1]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.final_n_features,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Train 모드일 때는 50:50 균형 샘플링
        if not self.is_test:
            half_steps = self.max_steps // 2
            sampled_benign = np.random.choice(self.benign_indices, half_steps, replace=True)
            sampled_attack = np.random.choice(self.attack_indices, half_steps, replace=True)
            self.episode_indices = np.concatenate([sampled_benign, sampled_attack])
            np.random.shuffle(self.episode_indices)
        else:
            # Test 모드일 때는 테스트셋 내 실제 비율과 자연스러운 가동 흐름 반영
            available_indices = np.arange(len(self.y))
            if len(available_indices) >= self.max_steps:
                self.episode_indices = np.random.choice(available_indices, self.max_steps, replace=False)
            else:
                self.episode_indices = np.random.choice(available_indices, self.max_steps, replace=True)
        
        state = self.X_scaled[self.episode_indices[self.current_step]].astype(np.float32)
        return state, {}

    def step(self, action):
        actual_label = self.y[self.episode_indices[self.current_step]]
        if action == actual_label:
            reward = 1.0  
        else:
            reward = -5.0 if (actual_label == 1 and action == 0) else -1.0  # 미탐/오탐 차등 패널티
        
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
# 2. DQN 아키텍처 및 3. 에이전트
# ==========================================
class StandardQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(StandardQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x): return self.network(x)


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
        self.epsilon = 0.01 if is_test else 1.0
        self.epsilon_decay = 0.996
        self.epsilon_min = 0.01
        
        self.batch_size, self.tau, self.focal_gamma = 64, 0.005, 2.0

    def select_action(self, state):
        if not self.is_test and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(device)
            return torch.argmax(self.policy_net(state_t)).item()

    def store_transition(self, state, action, reward, next_state, done):
        if not self.is_test:
            self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if self.is_test or len(self.memory) < self.batch_size: return
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
        focal_weight = (1 - torch.exp(-td_error)) ** self.focal_gamma
        loss = (focal_weight * nn.MSELoss(reduction='none')(current_q, target_q)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ==========================================
# 4. 통합 데이터 가공 및 누수 차단 RFECV 검증
# ==========================================
def load_and_split_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"지정한 경로에 CSV 파일이 존재하지 않습니다: {file_path}")
        
    print(f"통합 데이터 로드 시작: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path, low_memory=False)
    
    X = df.drop(columns=['Label', 'Timestamp'], errors='ignore')
    X = X.select_dtypes(include=[np.number]).astype(np.float32)
    y = df['Label'].apply(lambda x: 0 if str(x).strip() == 'Benign' else 1).values.astype(np.int8)
    del df
    
    # inf 및 NaN 정제
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
    
    # 데이터셋 층화 추출 분리 (Train 8: Test 2)
    print("층화 추출(Stratified Split) 진행 중...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_log, y, test_size=0.2, stratify=y, random_state=42
    )
    del X_log
    
    print("데이터 스케일링 진행 중...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    del X_train_raw, X_test_raw

    # -------------------------------------------------------------
    # [RFECV 기반 특성 최적화 - Train 데이터로만 엄격히 격리 학습]
    # -------------------------------------------------------------
    orig_feature_count = X_train_scaled.shape[1]
    print(f"RFECV 분석 시작 (원본 피처: {orig_feature_count}개)")
    print("AI가 Train 데이터를 분석하여 최적의 특성 개수를 스스로 탐색합니다...")
    
    # 교차 검증 속도를 위해 균형잡힌 서브 샘플 선별 (오직 Train 인덱스 영역 내부에서만 샘플링)
    sample_size = min(30000, len(y_train))
    sample_indices = np.random.choice(len(y_train), sample_size, replace=False)
    
    base_estimator = ExtraTreesClassifier(n_estimators=30, random_state=42, n_jobs=-1)
    
    # step=5, cv=3 교차검증 세팅 가동
    rfecv = RFECV(estimator=base_estimator, step=5, cv=3, scoring='accuracy', n_jobs=-1)
    rfecv.fit(X_train_scaled[sample_indices], y_train[sample_indices])
    
    # 최적의 특성 선택 마스크 획득 및 격리 마스킹 처리
    X_train_final = X_train_scaled[:, rfecv.support_]
    X_test_final = X_test_scaled[:, rfecv.support_] # Test 데이터는 변환 정보 교류 없이 차단 마스킹만 수용
    
    print(f"-> [결과] AI가 찾아낸 최적의 특성 개수: {rfecv.n_features_}개")
    print(f"-> 최종 신경망 입력 피처 결정 완료: {orig_feature_count}개 -> {X_train_final.shape[1]}개")
    # -------------------------------------------------------------
    
    print(f"학습셋 정상: {np.sum(y_train==0):,}개, 공격: {np.sum(y_train==1):,}개")
    print(f"테스트셋 정상: {np.sum(y_test==0):,}개, 공격: {np.sum(y_test==1):,}개")
    
    return X_train_final, X_test_final, y_train, y_test

# ==========================================
# 5. 메인 루프 실행 (Train -> Test 스테이지 분리)
# ==========================================
if __name__ == "__main__":
    clean_file_path = r"C:\ids2018_data\nids_advanced_cleaned.csv"
    
    # 고도화 데이터 로드 가동
    X_train, X_test, y_train, y_test = load_and_split_data(clean_file_path)
    
    # Train용 환경 및 에이전트 인스턴스 빌드
    train_env = NIDSEnv(X_train, y_train, max_steps=1000, is_test=False)
    agent = DQNAgent(state_dim=train_env.final_n_features, action_dim=train_env.action_space.n, is_test=False)
    
    num_train_episodes = 800
    history_rewards = []
    history_accuracy = []
    history_fpr = []
    history_fnr = []
    
    # ------------------------------------------
    # [STAGE 1] 모델 학습 (Train)
    # ------------------------------------------
    print(f"\n========= [STAGE 1] RFECV 메인 본 학습 시작 (총 {num_train_episodes} 에피소드) =========")
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
            
            act, pred = info["actual"], info["predicted"]
            if act == 1 and pred == 1: tp += 1
            elif act == 0 and pred == 0: tn += 1
            elif act == 0 and pred == 1: fp += 1
            elif act == 1 and pred == 0: fn += 1
            if done: break
                
        total_steps = tp + tn + fp + fn
        accuracy = ((tp + tn) / total_steps) * 100 if total_steps > 0 else 0.0
        fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0.0
        fnr = (fn / (fn + tp)) * 100 if (fn + tp) > 0 else 0.0
        attack_ratio = ((tp + fn) / total_steps) * 100 if total_steps > 0 else 0.0
        
        history_rewards.append(episode_reward)
        history_accuracy.append(accuracy)
        history_fpr.append(fpr)
        history_fnr.append(fnr)
        
        if (episode + 1) % 5 == 0 or episode == 0:
            print(f"에피 {episode+1:3d}/{num_train_episodes} | "
                  f"보상: {episode_reward:7.1f} | "
                  f"정확도: {accuracy:5.1f}% | "
                  f"오탐율(FPR): {fpr:5.1f}% | "
                  f"미탐율(FNR): {fnr:5.1f}% | "
                  f"공격비율: {attack_ratio:5.1f}% | "
                  f"입실론: {agent.epsilon:.3f}")
        
        agent.decay_epsilon()
            
    print("========= 메인 본 학습 완료 =========")
    
    # ------------------------------------------
    # [STAGE 2] 모델 검증 (Test)
    # ------------------------------------------
    print("\n========= [STAGE 2] 테스트 데이터셋 평가 시작 =========")
    agent.is_test = True
    test_env = NIDSEnv(X_test, y_test, max_steps=1000, is_test=True)
    
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
            
            act, pred = info["actual"], info["predicted"]
            if act == 1 and pred == 1: tp += 1
            elif act == 0 and pred == 0: tn += 1
            elif act == 0 and pred == 1: fp += 1
            elif act == 1 and pred == 0: fn += 1
            if done: break
                
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
    # 6. 결과 시각화 그래프 출력
    # ------------------------------------------
    plt.figure(figsize=(18, 5))

    # 보상 추이
    plt.subplot(1, 3, 1)
    plt.plot(history_rewards, color='blue', alpha=0.6)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)

    # 정확도 추이
    plt.subplot(1, 3, 2)
    plt.plot(history_accuracy, color='green', alpha=0.6)
    plt.title('Training Accuracy (%)')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.ylim(0, 105)
    plt.grid(True)

    # 오탐 및 미탐 추이 비교 그래프
    plt.subplot(1, 3, 3)
    plt.plot(history_fpr, label='FPR (False Alarm)', color='orange', alpha=0.7)
    plt.plot(history_fnr, label='FNR (Missed Attack)', color='red', alpha=0.7)
    plt.title('FPR vs FNR (%)')
    plt.xlabel('Episode')
    plt.ylabel('Rate (%)')
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()