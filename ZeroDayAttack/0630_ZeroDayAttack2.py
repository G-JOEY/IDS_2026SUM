# Zero-Day Attack 2
# RobustScaler 도입(이상치 제거): 중앙값(Median)과 사분위수(IQR)를 기준으로 스케일링을 수행
# 1D-CNN 오토인코더와의 결합: 오직 정상(Benign) 데이터로만 학습하는 1D-CNN 기반 오토인코더를 훈련시킨 후, 재구성 에러(Reconstruction Error)를 기존 피처 맨 마지막 컬럼에 새로운 특징 공간으로 결합
# 정상 데이터 중에서 오토인코더의 재구성 에러 점수가 상위 10%에 해당하는 샘플들을 강제로 공격 레이블로 바꾸어 DQN에게 주입함으로써, 유사한 변칙 패턴(Infiltration)에 당황하지 않고 공격으로 판단할 수 있게 함
# 오토인코더의 재구성 에러 점수가 정상 기준 최상위 1%를 초과하는 극단적인 이상 징후를 보이면 강제로 공격 처리하여, 지능적인 제로데이 공격이 정상인 척하더라도 기하학적 이상치를 잡아내는 오토인코더가 최후의 방화벽 역할을 수행함


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
from sklearn.preprocessing import RobustScaler  # 극단적 이상치 방어용 스케일러로 업그레이드
import matplotlib.pyplot as plt

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# [하이퍼파라미터 설정] 스위트 스팟 제어 타깃
# ==========================================
PSEUDO_RATE = 0.10     # 정상 데이터 중 에러 상위 몇 %를 함정(가상 공격)으로 주입할지 결정
HARD_CUT_RATE = 0.01   # 테스트 단계에서 DQN 오판을 무시하고 강제 차단할 오토인코더 최상위 에러 범위 (상위 1%)


# ==========================================
# 0. 1D-CNN 오토인코더 아키텍처 및 학습
# ==========================================
class CNN1DAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(CNN1DAutoencoder, self).__init__()
        self.input_dim = input_dim
        
        # 인코더
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        )
        
        # 출력 차원 동적 계산
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            dummy_encoded = self.encoder(dummy_input)
            self.encoded_channels = dummy_encoded.size(1)
            self.encoded_length = dummy_encoded.size(2)
            self.flatten_dim = self.encoded_channels * self.encoded_length

        # 디코더
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.flatten_dim, self.flatten_dim),
            nn.ReLU()
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=2, stride=2, padding=0)
        )
        
        # 최종 차원 복원 및 스케일 아웃 (RobustScaler 대응을 위해 Sigmoid 제거 및 Linear 결합)
        self.final_reconstruct = nn.Sequential(
            nn.Linear(self.encoded_length * 4, input_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1) 
        encoded = self.encoder(x)
        
        batch_size = encoded.size(0)
        encoded_flat = encoded.view(batch_size, -1)
        decoded_fc = self.decoder_fc(encoded_flat)
        
        decoded_conv_input = decoded_fc.view(batch_size, self.encoded_channels, self.encoded_length)
        decoded_output = self.decoder_conv(decoded_conv_input)
        
        decoded_flat = decoded_output.squeeze(1)
        final_output = self.final_reconstruct(decoded_flat)
        return final_output

def train_autoencoder(ae_model, benign_data, epochs=25, batch_size=1024, lr=0.001):
    print(f"\n>>> [공정 1] 1D-CNN 오토인코더 패턴 학습 시작 (Robust 특징 공간 추출, 총 {epochs} Epochs)...")
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
            print(f"CNN-AE Epoch {epoch+1:2d}/{epochs} | Avg Loss: {epoch_loss / dataset_size:.6f}")
    
    ae_model.eval()
    print(">>> 1D-CNN 오토인코더 가중치 잠금 완료.\n")
    return ae_model

def compute_recon_errors(ae_model, data, batch_size=2048):
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
    def __init__(self, X_data, y_data, is_zeroday_list, pseudo_threshold, max_steps=1000, is_test=False):
        super(NIDSEnv, self).__init__()
        
        self.X_scaled = X_data
        self.y = y_data               
        self.is_zeroday = is_zeroday_list  
        self.pseudo_threshold = pseudo_threshold 
        
        self.max_steps = max_steps
        self.is_test = is_test
        self.current_step = 0
        
        self.benign_indices = np.where((self.y == 0) & (~self.is_zeroday))[0]
        self.known_attack_indices = np.where((self.y == 1) & (~self.is_zeroday))[0]
        self.zeroday_indices = np.where(self.is_zeroday)[0]
        
        self.episode_indices = []
        
        num_features = self.X_scaled.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32)
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
        
        current_ae_error = self.X_scaled[idx][-1]
        
        if not self.is_test and actual_label == 0 and current_ae_error > self.pseudo_threshold:
            effective_label = 1  
        else:
            effective_label = actual_label
        
        if action == effective_label:
            reward = 1.0  
        else:
            if effective_label == 1 and action == 0:
                # 동적 패널티: 오토인코더 에러 수치가 높을수록, 미탐 시 DQN에게 가하는 벌점을 가중시킴
                reward = -5.0 * (1.0 + current_ae_error * 100.0)
            else:
                reward = -1.0  
        
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
# 2. DQN 아키텍처 및 에이전트
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

class DQNAgent:
    def __init__(self, state_dim, action_dim, is_test=False, hard_cut_threshold=99.9):
        self.action_dim = action_dim
        self.is_test = is_test
        self.hard_cut_threshold = hard_cut_threshold # 최후방 물리 차단용 하드 경계선
        
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
        ae_error = state[-1]
        
        # [하이브리드 핵심 2중 필터 가동] 실전(Test) 단계에서 에러 점수가 최상위 임계치를 넘으면 즉시 차단
        if self.is_test and ae_error > self.hard_cut_threshold:
            return 1  # DQN의 신경망 예측값과 무관하게 강제 드롭 조치
            
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
# 3. 데이터 로드 및 전처리 파이프라인
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
    
    # [개선] 대규모 이상치 노이즈에 견고한 RobustScaler 전격 도입
    print("Robust 스케일링 진행 중 (이상치 압착 방지)...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)  
    
    train_benign_indices = np.where((y_train == 0) & (~is_zd_train))[0]
    X_train_benign = X_train_scaled[train_benign_indices]
    
    ae = CNN1DAutoencoder(input_dim=X_train_scaled.shape[1]).to(device)
    ae = train_autoencoder(ae, X_train_benign, epochs=25, batch_size=1024)
    
    print("1D-CNN 오토인코더 기반 로버스트 이상치 점수 생성 중...")
    train_recon_errors = compute_recon_errors(ae, X_train_scaled)
    test_recon_errors = compute_recon_errors(ae, X_test_scaled)
    
    # 임계치 2종 자동 산정
    benign_train_errors = train_recon_errors[train_benign_indices]
    
    # 1) 가상 제로데이 면역용 기준선 (PSEUDO_RATE 연동)
    pseudo_percentile = 100 * (1.0 - PSEUDO_RATE)
    pseudo_threshold = float(np.percentile(benign_train_errors, pseudo_percentile))
    
    # 2) 테스트 단계 물리적 강제 차단용 기준선 (HARD_CUT_RATE 연동)
    hard_cut_percentile = 100 * (1.0 - HARD_CUT_RATE)
    hard_cut_threshold = float(np.percentile(benign_train_errors, hard_cut_percentile))
    
    print(f">>> [함정 경계선 (상위 {PSEUDO_RATE*100:.1f}%)] : {pseudo_threshold:.6f}")
    print(f">>> [강제 차단선 (상위 {HARD_CUT_RATE*100:.1f}%)] : {hard_cut_threshold:.6f}")
    
    X_train_final = np.hstack((X_train_scaled, train_recon_errors))
    X_test_final = np.hstack((X_test_scaled, test_recon_errors))
    
    return X_train_final, X_test_final, y_train, y_test, is_zd_train, is_zd_test, pseudo_threshold, hard_cut_threshold


# ==========================================
# 4. 메인 루프 실행 (Train -> Test)
# ==========================================
if __name__ == "__main__":
    clean_file_path = r"C:\ids2018_data\nids_advanced_cleaned.csv"
    
    X_train, X_test, y_train, y_test, is_zd_train, is_zd_test, pseudo_border, hard_border = load_and_split_data(clean_file_path)
    
    train_env = NIDSEnv(X_train, y_train, is_zd_train, pseudo_threshold=pseudo_border, max_steps=1000, is_test=False)
    # 에이전트 내부로 최후방 물리 차단선(hard_border) 이식
    agent = DQNAgent(state_dim=train_env.observation_space.shape[0], action_dim=train_env.action_space.n, is_test=False, hard_cut_threshold=hard_border)
    
    num_train_episodes = 800 
    history_rewards = []
    history_accuracy = []
    
    # ------------------------------------------
    # [STAGE 1] 모델 면역 학습
    # ------------------------------------------
    print(f"\n========= [STAGE 1] 하이브리드 NIDS 최적화 면역 훈련 시작 =========")
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
            print(f"에피소드 {episode+1:3d}/{num_train_episodes} | 보상: {episode_reward:7.1f} | 겉보기 학습 정확도: {accuracy:5.1f}% | 입실론: {agent.epsilon:.3f}")
        
        agent.decay_epsilon()
            
    print("========= 1차 연산 완료 =========")
    
    # ------------------------------------------
    # [STAGE 2] 미지의 제로데이 어택 검증 및 하이브리드 실증
    # ------------------------------------------
    print("\n========= [STAGE 2] 미지의 제로데이 어택 평가 시작 (2중 필터 제어 필드 가동) =========")
    agent.is_test = True
    test_env = NIDSEnv(X_test, y_test, is_zd_test, pseudo_threshold=0.0, max_steps=1000, is_test=True)
    
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

    print("\n========= 2중 하이브리드 최적화 실증 결과 요약 =========")
    print(f"적용된 가상 주입 비율 (PSEUDO_RATE)      : {PSEUDO_RATE * 100}%")
    print(f"적용된 강제 차단 범위 (HARD_CUT_RATE)    : 최상위 {HARD_CUT_RATE * 100}%")
    print(f"평균 테스트 정확도 (Total Test Accuracy)  : {np.mean(test_acc):.2f}%")
    print(f"평균 오탐율 (False Alarm Rate)            : {np.mean(test_fpr):.2f}%")
    print(f"평균 미탐율 (Missed Attack Rate)          : {np.mean(test_fnr):.2f}%")
    print("-------------------------------------------------")
    print(f"테스트 중 유입된 총 제로데이(Infiltration) 수 : {total_zd_count:,}개")
    print(f"2중 시스템이 걸러낸 총 제로데이 수              : {total_zd_detected:,}개")
    print(f"★ 최종 제로데이 탐지율 (Zero-day Detection Rate) : {final_zd_detection_rate:.2f}% ★")
    print("=================================================")