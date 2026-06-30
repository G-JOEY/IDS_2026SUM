# 데이터 스케일러: MinMaxScaler
# 오토인코더 구조: 일반 MLP
# 테스트 차단 기준: 상위 90%선 단일 임계치 차단
# Reward: 고정 미탐 페널티


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
# 0. 오토인코더(Autoencoder) 정의 및 학습
# ==========================================
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)  # 압축 잠재 공간
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()  # 0~1 값 복원
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_autoencoder(ae_model, benign_data, epochs=25, batch_size=1024, lr=0.001):
    """정상 패턴을 더 깐깐하게 학습하도록 Epochs를 25로 상향 조정"""
    print(f"\n>>> [공정 1] 정상 데이터 기반 오토인코더 훈련 시작 (총 {epochs} Epochs)...")
    ae_model.train()
    optimizer = optim.Adam(ae_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    dataset_size = benign_data.shape[0]
    indices = np.arange(dataset_size)
    
    for epoch in range(epochs):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        num_batches = int(np.ceil(dataset_size / batch_size))
        
        for i in range(num_batches):
            batch_idx = indices[i * batch_size : (i + 1) * batch_size]
            batch_x = torch.FloatTensor(benign_data[batch_idx]).to(device)
            
            reconstructed = ae_model(batch_x)
            loss = criterion(reconstructed, batch_x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_idx)
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"AE Epoch {epoch+1:2d}/{epochs} | Avg Loss: {epoch_loss / dataset_size:.6f}")
    
    ae_model.eval()
    print(">>> 오토인코더 가중치 고정 완료.\n")
    return ae_model

def compute_recon_errors(ae_model, data, batch_size=2048):
    """각 패킷 데이터 샘플의 독립적인 재구성 에러 수치를 계산합니다."""
    ae_model.eval()
    errors = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch_x = torch.FloatTensor(data[i:i+batch_size]).to(device)
            reconstructed = ae_model(batch_x)
            loss_per_sample = torch.mean((reconstructed - batch_x) ** 2, dim=1)
            errors.extend(loss_per_sample.cpu().numpy())
    return np.array(errors, dtype=np.float32).reshape(-1, 1)


# ==========================================
# 1. NIDS 강화학습 환경
# ==========================================
class NIDSEnv(gym.Env):
    def __init__(self, X_data, y_data, is_zeroday_list, max_steps=1000, is_test=False):
        super(NIDSEnv, self).__init__()
        
        self.X_scaled = X_data
        self.y = y_data               # 0: 정상, 1: 공격
        self.is_zeroday = is_zeroday_list  # True: 제로데이 공격
        
        self.max_steps = max_steps
        self.is_test = is_test
        self.current_step = 0
        
        # 인덱스 분리
        self.benign_indices = np.where((self.y == 0) & (~self.is_zeroday))[0]
        self.known_attack_indices = np.where((self.y == 1) & (~self.is_zeroday))[0]
        self.zeroday_indices = np.where(self.is_zeroday)[0]
        
        self.episode_indices = []
        
        num_features = self.X_scaled.shape[1]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(num_features,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        if not self.is_test:
            half_steps = self.max_steps // 2
            sampled_benign = np.random.choice(self.benign_indices, half_steps, replace=True)
            sampled_attack = np.random.choice(self.known_attack_indices, half_steps, replace=True)
            self.episode_indices = np.concatenate([sampled_benign, sampled_attack])
            np.random.shuffle(self.episode_indices)
        else:
            available_indices = np.arange(len(self.y))
            if len(available_indices) >= self.max_steps:
                self.episode_indices = np.random.choice(available_indices, self.max_steps, replace=False)
            else:
                self.episode_indices = np.random.choice(available_indices, self.max_steps, replace=True)
        
        state = self.X_scaled[self.episode_indices[self.current_step]].astype(np.float32)
        info = {}
        return state, info

    def step(self, action):
        idx = self.episode_indices[self.current_step]
        actual_label = self.y[idx]
        is_zd = self.is_zeroday[idx]
        
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
            
        info = {"actual": actual_label, "predicted": action, "is_zeroday": is_zd}
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
    def __init__(self, state_dim, action_dim, is_test=False, ae_threshold=0.01):
        self.action_dim = action_dim
        self.is_test = is_test
        self.ae_threshold = ae_threshold # 제로데이 판정용 수학적 임계치
        
        self.policy_net = StandardQNetwork(state_dim, action_dim).to(device)
        self.target_net = StandardQNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.00025)
        self.memory = deque(maxlen=60000)
        
        self.gamma = 0.99
        self.epsilon = 0.01 if is_test else 1.0
        
        self.epsilon_decay = 0.996  
        self.epsilon_min = 0.01
        
        self.batch_size = 64
        self.tau = 0.005  
        self.focal_gamma = 2.0

    def select_action(self, state):
        """DQN 예측값과 오토인코더 이상치 점수를 결합한 의사결정을 수행합니다."""
        # state의 가장 마지막 원소[-1]는 연동된 오토인코더 에러 점수
        ae_error = state[-1]
        
        # [하이브리드 핵심] 테스트 단계에서 에러가 튜닝된 임계치를 넘으면 즉시 차단 가동
        if self.is_test and ae_error > self.ae_threshold:
            return 1  # DQN의 오판 유무와 상관없이 물리적으로 강력 차단
            
        # 일반적인 탐색 및 DQN 예측 루프
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
# 4. 데이터 전처리 및 통합 피처 맵 빌드
# ==========================================
def load_and_split_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"지정한 경로에 CSV 파일이 존재하지 않습니다: {file_path}")
        
    print(f"데이터 로드 시작: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path, low_memory=False)
    
    known_labels = ['Benign', 'DDoS attacks-HOIC', 'DDoS attacks-LOIC-HTTP', 'DoS attacks-Hulk']
    zeroday_labels = ['Infiltration']
    
    df_filtered = df[df['Label'].isin(known_labels + zeroday_labels)].copy()
    del df
    
    X = df_filtered.drop(columns=['Label', 'Timestamp'], errors='ignore')
    X = X.select_dtypes(include=[np.number]).astype(np.float32)
    
    y = df_filtered['Label'].apply(lambda x: 0 if str(x).strip() == 'Benign' else 1).values.astype(np.int8)
    is_zeroday = df_filtered['Label'].apply(lambda x: True if str(x).strip() in zeroday_labels else False).values.astype(bool)
    del df_filtered
    
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
    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
    
    X_train_raw, X_test_raw = X_log[idx_train], X_log[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    is_zd_train, is_zd_test = is_zeroday[idx_train], is_zeroday[idx_test]
    del X_log
    
    print("데이터 스케일링 진행 중...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)  
    
    # -----------------------------------------------------
    # [공정 2] 고도화된 오토인코더 훈련
    # -----------------------------------------------------
    train_benign_indices = np.where((y_train == 0) & (~is_zd_train))[0]
    X_train_benign = X_train_scaled[train_benign_indices]
    
    ae = Autoencoder(input_dim=X_train_scaled.shape[1]).to(device)
    ae = train_autoencoder(ae, X_train_benign, epochs=25, batch_size=1024)
    
    print("오토인코더 기반 이상치 분포(Reconstruction Error) 생성 중...")
    train_recon_errors = compute_recon_errors(ae, X_train_scaled)
    test_recon_errors = compute_recon_errors(ae, X_test_scaled)
    
    # 미세 침투 공격을 필터링하기 위해 경계선을 상위 90%선으로 하향 조정
    benign_train_errors = train_recon_errors[train_benign_indices]
    ae_threshold = float(np.percentile(benign_train_errors, 90))
    print(f">>> [임계치 계산 완료] 제로데이 탐지 강화를 위한 튜닝된 AE 경계선: {ae_threshold:.6f}")
    
    # 데이터 피처 맵 최후방에 에러 차원 병렬 결합
    X_train_final = np.hstack((X_train_scaled, train_recon_errors))
    X_test_final = np.hstack((X_test_scaled, test_recon_errors))
    
    print(f"학습셋 정상: {np.sum((y_train==0) & (~is_zd_train)):,}개 | 알려진 공격: {np.sum((y_train==1) & (~is_zd_train)):,}개")
    print(f"테스트셋 정상: {np.sum(y_test==0):,}개 | 알려진 공격: {np.sum((y_test==1) & (~is_zd_test)):,}개 | [제로데이]: {np.sum(is_zd_test):,}개")
    
    return X_train_final, X_test_final, y_train, y_test, is_zd_train, is_zd_test, ae_threshold


# ==========================================
# 5. 메인 루프 실행 (Train -> Test)
# ==========================================
if __name__ == "__main__":
    clean_file_path = r"C:\ids2018_data\nids_advanced_cleaned.csv"
    
    # 데이터셋 구성 및 자동 계산된 오토인코더 임계치 확보
    X_train, X_test, y_train, y_test, is_zd_train, is_zd_test, calculated_threshold = load_and_split_data(clean_file_path)
    
    # 환경 구축 및 자동 계산된 임계치 에이전트에 이식
    train_env = NIDSEnv(X_train, y_train, is_zd_train, max_steps=1000, is_test=False)
    agent = DQNAgent(state_dim=train_env.observation_space.shape[0], action_dim=train_env.action_space.n, is_test=False, ae_threshold=calculated_threshold)
    
    num_train_episodes = 800 
    history_rewards = []
    history_accuracy = []
    
    # ------------------------------------------
    # [STAGE 1] 모델 학습 (Train)
    # ------------------------------------------
    print("\n========= [STAGE 1] 하이브리드 NIDS 학습 시작 (Zero-day 완전 격리 상태) =========")
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
        
        history_rewards.append(episode_reward)
        history_accuracy.append(accuracy)
        
        if (episode + 1) % 20 == 0 or episode == 0:
            print(f"에피소드 {episode+1:3d}/{num_train_episodes} | 보상: {episode_reward:7.1f} | 알려진 정확도: {accuracy:5.1f}% | 입실론: {agent.epsilon:.3f}")
        
        agent.decay_epsilon()
            
    print("========= 학습 완료 =========")
    
    # ------------------------------------------
    # [STAGE 2] 미지의 제로데이 어택 검증 및 하이브리드 방어 실증 (Test)
    # ------------------------------------------
    print("\n========= [STAGE 2] 미지의 제로데이 어택 평가 시작 (하이브리드 제어 필터 가동) =========")
    agent.is_test = True
    test_env = NIDSEnv(X_test, y_test, is_zd_test, max_steps=1000, is_test=True)
    
    num_test_episodes = 50  
    test_acc, test_fpr, test_fnr = [], [], []
    
    total_zd_count = 0
    total_zd_detected = 0
    
    for episode in range(num_test_episodes):
        state, info = test_env.reset()
        tp, tn, fp, fn = 0, 0, 0, 0
        zd_attack_ep, zd_detect_ep = 0, 0
        
        for step in range(test_env.max_steps):
            action = agent.select_action(state)  
            next_state, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            
            state = next_state
            
            act = info["actual"]
            pred = info["predicted"]
            is_zd = info["is_zeroday"]
            
            if act == 1 and pred == 1: tp += 1
            elif act == 0 and pred == 0: tn += 1
            elif act == 0 and pred == 1: fp += 1
            elif act == 1 and pred == 0: fn += 1
            
            if is_zd:
                zd_attack_ep += 1
                if pred == 1:  
                    zd_detect_ep += 1
            
            if done:
                break
                
        total_steps = tp + tn + fp + fn
        acc = ((tp + tn) / total_steps) * 100
        fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0.0
        fnr = (fn / (fn + tp)) * 100 if (fn + tp) > 0 else 0.0
        
        test_acc.append(acc)
        test_fpr.append(fpr)
        test_fnr.append(fnr)
        
        total_zd_count += zd_attack_ep
        total_zd_detected += zd_detect_ep
        
        if (episode + 1) % 10 == 0 or episode == 0:
            zd_rate_ep = (zd_detect_ep / zd_attack_ep) * 100 if zd_attack_ep > 0 else 0.0
            print(f"테스트 에피소드 {episode+1:2d}/{num_test_episodes} | 전체 정확도: {acc:5.1f}% | 유입된 제로데이: {zd_attack_ep:3d}개 | 제로데이 탐지율: {zd_rate_ep:5.1f}%")

    final_zd_detection_rate = (total_zd_detected / total_zd_count) * 100 if total_zd_count > 0 else 0.0

    print("\n========= 하이브리드 제로데이 실증 테스트 결과 요약 =========")
    print(f"평균 테스트 정확도 (Total Test Accuracy)  : {np.mean(test_acc):.2f}%")
    print(f"평균 오탐율 (False Alarm Rate)            : {np.mean(test_fpr):.2f}%")
    print(f"평균 미탐율 (Missed Attack Rate)          : {np.mean(test_fnr):.2f}%")
    print("-------------------------------------------------")
    print(f"테스트 중 유입된 총 제로데이(Infiltration) 수 : {total_zd_count:,}개")
    print(f"하이브리드 시스템이 탐지해낸 제로데이 수        : {total_zd_detected:,}개")
    print(f"★ 최종 제로데이 탐지율 (Zero-day Detection Rate) : {final_zd_detection_rate:.2f}% ★")
    print("=================================================")