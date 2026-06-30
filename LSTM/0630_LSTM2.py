# LSTM 2
# 상한선 규제: 날짜별 최대 에피소드 수 제한
# 이중 풀 인덱싱: 에피소드 구성 시 일반/혼합 구간용 에피소드와 희귀 공격이 단 1개라도 포함된 희귀 구간용 에피소드를 따로 보관
# 학습 에피소드 진행 단계에 따라 희귀 공격을 주입하는 확률을 동적으로 조절
# 윈도우 크기 확장


import os
import glob
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

# ROCm MIOpen 커널 에러 방지 및 재현성 설정
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. NIDS 멀티파일 시계열 환경
# ==========================================
class NIDSMultiFileEnv(gym.Env):
    def __init__(self, folder_path, max_steps=1000, window_size=5, stride=200, is_test=False):
        super(NIDSMultiFileEnv, self).__init__()
        
        self.window_size = window_size
        self.max_steps = max_steps
        self.stride = stride
        self.is_test = is_test
        self.current_step = 0
        
        # 파일 목록 확인
        file_list = glob.glob(os.path.join(folder_path, "*.csv"))
        if not file_list:
            raise FileNotFoundError(f"경로에 CSV 파일이 없습니다: {folder_path}")
        
        # 가독성을 위한 정렬
        self.file_list = sorted(file_list)
        print(f"\n[환경 초기화] 총 {len(self.file_list)}개의 독립 일자별 파일을 감지했습니다.")
        
        self.files_data = {}
        self.num_features = None
        self.feature_columns = None
        
        # 희귀 공격 목록 사전 정의 (분석 결과 기반: 노출 에피소드가 극도로 적은 타겟)
        self.rare_attack_labels = ['Brute Force -Web', 'Brute Force -XSS', 'SQL Injection', 'DDOS attack-LOIC-UDP', 'Label']
        
        # 데이터 순차 사전 로드 및 인덱스 주머니(Pool) 횔터링
        for idx, file_path in enumerate(self.file_list):
            f_name = os.path.basename(file_path)
            print(f"[{idx+1}/{len(self.file_list)}] 인덱싱 및 가공 중: {f_name}")
            
            df = pd.read_csv(file_path, low_memory=False)
            if 'Dst Port' in df.columns:
                df = df[df['Dst Port'] != 'Dst Port'].reset_index(drop=True)
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
                df = df.sort_values(by='Timestamp').reset_index(drop=True)
                
            X_raw = df.drop(columns=['Label', 'Timestamp'], errors='ignore')
            
            # 다중 클래스(공격명) 및 이진 라벨 분리 보존
            df['Label'] = df['Label'].str.strip()
            y_raw_strings = df['Label'].values
            y_binary = df['Label'].apply(lambda x: 0 if x == 'Benign' else 1).values.astype(np.int8)
            del df
            
            for col in X_raw.columns:
                if X_raw[col].dtype == object:
                    X_raw[col] = pd.to_numeric(X_raw[col], errors='coerce')
                    
            if self.feature_columns is None:
                self.feature_columns = X_raw.select_dtypes(include=[np.number]).columns.tolist()
                self.num_features = len(self.feature_columns)
                print(f"   -> 마스터 피처 고정: 총 {self.num_features}개 수립")
                
            X_raw = X_raw.reindex(columns=self.feature_columns).fillna(0.0).astype(np.float32)
            X_np = X_raw.to_numpy(dtype=np.float32)
            del X_raw
            
            # 결측치 정제 및 로그 스케일링
            X_np[np.isinf(X_np)] = np.nan
            col_means = np.nan_to_num(np.nanmean(X_np, axis=0), nan=0.0)
            inds = np.where(np.isnan(X_np))
            X_np[inds] = np.take(col_means, inds[1])
            np.clip(X_np, a_min=0, a_max=None, out=X_np)
            X_log = np.log1p(X_np)
            del X_np
            
            # 스케일러 적용
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_log)
            del X_log
            
            # --- 실질 주머니(Pool) 횔터링 알고리즘 ---
            total_rows = len(y_binary)
            valid_start_range = total_rows - self.max_steps
            
            pool_normal = []
            pool_rare = []
            
            # 누적합 기반 초고속 검색 구조
            prefix_sum = np.zeros(total_rows + 1, dtype=np.int32)
            np.cumsum(y_binary, out=prefix_sum[1:])
            
            for start_idx in range(0, valid_start_range, self.stride):
                end_idx = start_idx + self.max_steps
                attack_cnt = prefix_sum[end_idx] - prefix_sum[start_idx]
                attack_ratio = attack_cnt / self.max_steps
                
                # 테스트 모드일 때는 제약 조건 없이 전체 분할 탐색
                if self.is_test:
                    if attack_ratio > 0:
                        pool_normal.append(start_idx)
                    continue
                
                # [주머니 1] 일반/혼합 및 정상 구간 필터링 (공격 1% 이상인 유효 구간)
                if 0.01 <= attack_ratio <= 0.60:
                    pool_normal.append(start_idx)
                
                # [주머니 2] 희귀 공격 절대 사수 강제 인덱싱 규칙
                slice_strings = y_raw_strings[start_idx:end_idx]
                if any(rare in slice_strings for rare in self.rare_attack_labels):
                    pool_rare.append(start_idx)
            
            # 파일당 무지성 편향 독점을 방지하기 위한 최대 슬롯 상한 조절 (정규화 패치)
            if not self.is_test and len(pool_normal) > 150:
                pool_normal = random.sample(pool_normal, 150)
                
            self.files_data[idx] = {
                "X": X_scaled,
                "y": y_binary,
                "pool_normal": pool_normal,
                "pool_rare": pool_rare,
                "valid_start_range": valid_start_range
            }
            print(f"   -> 인덱싱 완료 (일반 풀: {len(pool_normal)}개, 희귀 풀: {len(pool_rare)}개)")
            
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.window_size, self.num_features), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        
        self.current_X = None
        self.current_y = None
        self.episode_start_idx = 0

    def _get_windowed_state(self, current_step_offset):
        target_idx = self.episode_start_idx + current_step_offset
        states_bucket = []
        for i in reversed(range(self.window_size)):
            lookback_idx = target_idx - i
            if lookback_idx < self.episode_start_idx:
                states_bucket.append(np.zeros(self.num_features, dtype=np.float32))
            else:
                states_bucket.append(self.current_X[lookback_idx])
        return np.stack(states_bucket, axis=0).astype(np.float32)

    def set_rare_probability(self, episode):
        """에스컬레이터 확률 제어 전략 함수"""
        if episode < 150:
            return 0.00   # 기초 체력 다지기 (안정화)
        elif episode < 450:
            return 0.10  # 심화 과외 집중 기간
        else:
            return 0.04  # 실전형 밸런스 유지

    def reset(self, seed=None, options=None, current_episode_idx=0):
        super().reset(seed=seed)
        self.current_step = 0
        
        # 1. 파일 독립성 보장을 위해 무작위 파일 고르기
        chosen_file_idx = np.random.choice(list(self.files_data.keys()))
        f_info = self.files_data[chosen_file_idx]
        
        self.current_X = f_info["X"]
        self.current_y = f_info["y"]
        
        if self.is_test:
            # 테스트 시에는 안전하게 탐색 풀 내부에서 추출
            if f_info["pool_normal"]:
                self.episode_start_idx = np.random.choice(f_info["pool_normal"])
            else:
                self.episode_start_idx = np.random.randint(0, max(1, f_info["valid_start_range"]))
        else:
            rare_prob = self.set_rare_probability(current_episode_idx)
            
            # 2. 확률적 주머니 주입 및 과적합 방지 랜덤 오프셋 적용
            if random.random() < rare_prob and len(f_info["pool_rare"]) > 0:
                base_idx = np.random.choice(f_info["pool_rare"])
                # 타임라인 암기 분쇄용 오프셋 노이즈 추가
                offset = np.random.randint(-50, 50)
                self.episode_start_idx = int(np.clip(base_idx + offset, 0, f_info["valid_start_range"]))
            else:
                if len(f_info["pool_normal"]) > 0:
                    self.episode_start_idx = np.random.choice(f_info["pool_normal"])
                else:
                    self.episode_start_idx = np.random.randint(0, max(1, f_info["valid_start_range"]))
                    
        return self._get_windowed_state(self.current_step), {}

    def step(self, action):
        actual_label = self.current_y[self.episode_start_idx + self.current_step]
        
        if action == actual_label:
            reward = 2.0 if actual_label == 0 else 1.0
        else:
            reward = -5.0 if (actual_label == 1 and action == 0) else -1.0
            
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        if not terminated:
            next_state = self._get_windowed_state(self.current_step)
        else:
            next_state = np.zeros(self.observation_space.shape, dtype=np.float32)
            
        return next_state, reward, terminated, truncated, {"actual": actual_label, "predicted": action}

# ==========================================
# 2. LSTM Q-Network 아키텍처
# ==========================================
class LSTMQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, num_layers=1):
        super(LSTMQNetwork, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# ==========================================
# 3. 에이전트
# ==========================================
class DQNAgent:
    def __init__(self, input_dim, action_dim, is_test=False):
        self.action_dim = action_dim
        self.is_test = is_test
        
        hidden_dim = 128
        self.policy_net = LSTMQNetwork(input_dim, hidden_dim, action_dim).to(device)
        self.target_net = LSTMQNetwork(input_dim, hidden_dim, action_dim).to(device)
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
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                return torch.argmax(self.policy_net(state_t)).item()

    def store_transition(self, state, action, reward, next_state, done):
        if self.is_test: return
        if reward == 2.0 or reward == -1.0:
            self.memory_benign.append((state, action, reward, next_state, done))
        else:
            self.memory_attack.append((state, action, reward, next_state, done))

    def train_step(self):
        if self.is_test: return
        half_batch = self.batch_size // 2
        if len(self.memory_benign) < half_batch or len(self.memory_attack) < half_batch:
            return
            
        batch = random.sample(self.memory_benign, half_batch) + random.sample(self.memory_attack, half_batch)
        random.shuffle(batch)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        states_t = torch.FloatTensor(np.array(states)).to(device)
        actions_t = torch.LongTensor(actions).view(-1, 1).to(device)
        rewards_t = torch.FloatTensor(rewards).view(-1, 1).to(device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(device)
        dones_t = torch.FloatTensor(dones).view(-1, 1).to(device)
        
        try:
            current_q = self.policy_net(states_t).gather(1, actions_t)
            with torch.no_grad():
                max_next_q = self.target_net(next_states_t).max(1)[0].view(-1, 1)
                target_q = rewards_t + (self.gamma * max_next_q * (1 - dones_t))
                
            td_error = torch.abs(current_q - target_q)
            loss = ((1 - torch.exp(-td_error)) ** self.focal_gamma * nn.MSELoss(reduction='none')(current_q, target_q)).mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        except RuntimeError as e:
            if "miopenStatusUnknownError" in str(e):
                # ROCm 커널 크래시 긴급 CPU 대피 장치
                self.policy_net.cpu(); self.target_net.cpu()
                current_q = self.policy_net(states_t.cpu()).gather(1, actions_t.cpu())
                with torch.no_grad():
                    max_next_q = self.target_net(next_states_t.cpu()).max(1)[0].view(-1, 1)
                    target_q = rewards_t.cpu() + (self.gamma * max_next_q * (1 - dones_t.cpu()))
                td_error = torch.abs(current_q - target_q)
                loss = ((1 - torch.exp(-td_error)) ** self.focal_gamma * nn.MSELoss(reduction='none')(current_q, target_q)).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.policy_net.to(device); self.target_net.to(device)
            else:
                raise e

        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ==========================================
# 4. 체크포인트 관리 기능 함수
# ==========================================
def save_checkpoint(agent, episode, file_path="nids_lstm_dqn.pth"):
    checkpoint = {
        'episode': episode,
        'policy_net_state': agent.policy_net.state_dict(),
        'target_net_state': agent.target_net.state_dict(),
        'optimizer_state': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon
    }
    torch.save(checkpoint, file_path)

def load_checkpoint(agent, file_path="nids_lstm_dqn.pth"):
    if os.path.exists(file_path):
        checkpoint = torch.load(file_path, map_location=device)
        agent.policy_net.load_state_dict(checkpoint['policy_net_state'])
        agent.target_net.load_state_dict(checkpoint['target_net_state'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
        agent.epsilon = checkpoint['epsilon']
        return checkpoint['episode'] + 1
    return 0

# ==========================================
# 5. 메인 루프 실행 제어부
# ==========================================
if __name__ == "__main__":
    # 데이터 폴더 경로 설정 (각 10개의 CSV 파일이 보존되어 있는 경로)
    data_folder_path = r"C:\ids2018_data"
    model_path = "nids_lstm_dqn.pth"
    
    # 훈련 환경 구축
    train_env = NIDSMultiFileEnv(data_folder_path, max_steps=1000, window_size=5, stride=200, is_test=False)
    agent = DQNAgent(input_dim=train_env.num_features, action_dim=train_env.action_space.n, is_test=False)
    
    num_train_episodes = 800
    start_episode = load_checkpoint(agent, model_path)
    
    history_rewards = []
    history_accuracy = []
    history_fpr = []
    history_fnr = []
    
    # ------------------------------------------
    # [STAGE 1] 모델 학습 및 복원 루프
    # ------------------------------------------
    if start_episode < num_train_episodes:
        print(f"\n========= [STAGE 1] LSTM-DQN 아키텍처 학습 연동 시작 (재개 지점: 에피 {start_episode+1}) =========")
        for episode in range(start_episode, num_train_episodes):
            state, info = train_env.reset(current_episode_idx=episode)
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
            
            if (episode + 1) % 5 == 0 or episode == start_episode:
                print(f"에피 {episode+1:3d}/{num_train_episodes} | 보상: {episode_reward:7.1f} | "
                      f"정확도: {accuracy:5.1f}% | FPR: {fpr:5.1f}% | FNR: {fnr:5.1f}% | "
                      f"공격비율: {attack_ratio:5.1f}% | 입실론: {agent.epsilon:.3f}")
            
            agent.decay_epsilon()
            
            # 💡 10 에피소드마다 주기적 백업 자동화
            if (episode + 1) % 10 == 0:
                save_checkpoint(agent, episode, model_path)
                
        # 가독성을 위한 최종 저장
        save_checkpoint(agent, num_train_episodes - 1, model_path)
        print("========= 전 프로세스 학습 수렴 완료 =========")
    else:
        print(f"\n[안내] 이미 {num_train_episodes} 에피소드 학습이 완료된 가중치 파일이 발견되었습니다. 곧바로 검증 단계로 진입합니다.")

    # ------------------------------------------
    # [STAGE 2] 최종 모델 평가 (Test)
    # ------------------------------------------
    print("\n========= [STAGE 2] 저장된 최적 모델 로드 및 실전 평가 시작 =========")
    agent.is_test = True
    load_checkpoint(agent, model_path)
    
    test_env = NIDSMultiFileEnv(data_folder_path, max_steps=1000, window_size=5, stride=200, is_test=True)
    num_test_episodes = 50
    test_fpr, test_fnr, test_acc = [], [], []
    total_test_benign, total_test_attack = 0, 0
    
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
            if done: break
                
        total_steps = tp + tn + fp + fn
        acc = ((tp + tn) / total_steps) * 100 if total_steps > 0 else 0.0
        fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0.0
        fnr = (fn / (fn + tp)) * 100 if (fn + tp) > 0 else 0.0
        
        ep_benign = tn + fp
        ep_attack = tp + fn
        total_test_benign += ep_benign
        total_test_attack += ep_attack
        
        test_acc.append(acc)
        test_fpr.append(fpr)
        test_fnr.append(fnr)
        
        if (episode + 1) % 10 == 0 or episode == 0:
            print(f"테스트 {episode+1:2d}/{num_test_episodes} | 정상: {ep_benign:3d}개 | 공격: {ep_attack:3d}개 | "
                  f"정확도: {acc:5.1f}% | FPR: {fpr:5.1f}% | FNR: {fnr:5.1f}%")

    all_test_steps = total_test_benign + total_test_attack
    avg_benign_ratio = (total_test_benign / all_test_steps) * 100 if all_test_steps > 0 else 0
    avg_attack_ratio = (total_test_attack / all_test_steps) * 100 if all_test_steps > 0 else 0

    print("\n========= 최종 테스트 결과 요약 =========")
    print(f"전체 테스트 본문 분포 : 정상 {total_test_benign:,}개({avg_benign_ratio:.1f}%) / 공격 {total_test_attack:,}개({avg_attack_ratio:.1f}%)")
    print(f"평균 정확도 (Test Accuracy) : {np.mean(test_acc):.2f}%")
    print(f"평균 오탐율 (False Alarm Rate) : {np.mean(test_fpr):.2f}%")
    print(f"평균 미탐율 (Missed Attack Rate) : {np.mean(test_fnr):.2f}%")
    print("=========================================")

    # ------------------------------------------
    # 6. 학습 결과 차트 출력 (처음부터 학습한 경우에 한함)
    # ------------------------------------------
    if history_rewards:
        plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 1)
        plt.plot(history_rewards, color='blue', alpha=0.6)
        plt.title('Training Episode Rewards')
        plt.xlabel('Episode'); plt.ylabel('Total Reward'); plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(history_accuracy, color='green', alpha=0.6)
        plt.title('Training Accuracy (%)')
        plt.xlabel('Episode'); plt.ylabel('Accuracy'); plt.ylim(0, 105); plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(history_fpr, label='FPR', color='orange', alpha=0.7)
        plt.plot(history_fnr, label='FNR', color='red', alpha=0.7)
        plt.title('Training FPR vs FNR (%)')
        plt.xlabel('Episode'); plt.ylabel('Rate (%)'); plt.ylim(0, 105); plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.show()