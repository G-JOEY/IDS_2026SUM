# EDA for Zero-Day Attack
# 주요 네트워크 피처 분포도 생성: 변별력이 높은 4개의 피처를 자동으로 선정하여, 5개 레이블의 밀도 분포를 시각화
# PCA 기반 2차원 전역 특징 공간 시각화: 수십 개에 달하는 고차원 네트워크 피처 공간을 눈으로 볼 수 있게 주성분 분석(PCA)을 통해 2차원 공간으로 압축한 뒤 산점도로 그림
# 클래스 간 유클리드 거리 행렬 계산


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def analyze_all_nids_classes(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"지정한 경로에 CSV 파일이 존재하지 않습니다: {file_path}")
        
    print("1. 전체 데이터 로드 및 다중 레이블 필터링...")
    df = pd.read_csv(file_path, low_memory=False)
    
    # 연구 설계에 포함된 모든 레이블 대상
    target_labels = [
        'Benign', 
        'DDoS attacks-HOIC', 
        'DDoS attacks-LOIC-HTTP', 
        'DoS attacks-Hulk', 
        'Infiltration'
    ]
    df_filtered = df[df['Label'].isin(target_labels)].copy()
    df_filtered['Label'] = df_filtered['Label'].str.strip()
    del df
    
    # 수치형 피처 전처리
    X = df_filtered.drop(columns=['Label', 'Timestamp'], errors='ignore')
    X = X.select_dtypes(include=[np.number]).astype(np.float32)
    y = df_filtered['Label'].values
    
    # 결측치 및 인피니티 처리
    X_np = X.to_numpy()
    X_np[np.isinf(X_np)] = np.nan
    col_means = np.nanmean(X_np, axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)
    inds = np.where(np.isnan(X_np))
    X_np[inds] = np.take(col_means, inds[1])
    
    # 모델 학습 환경과 동일한 스케일링 적용 (Log1p -> MinMaxScaler)
    np.clip(X_np, a_min=0, a_max=None, out=X_np)
    X_log = np.log1p(X_np)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_log)
    
    feature_names = X.columns
    df_scaled = pd.DataFrame(X_scaled, columns=feature_names)
    df_scaled['Label'] = y
    
    print("\n2. 클래스별 주요 네트워크 피처 분포도 생성 중...")
    # 변별력이 높은 주요 통계적 피처 4개 선정
    potential_features = [col for col in feature_names if 'Length' in col or 'Packets' in col or 'Duration' in col][:4]
    
    plt.figure(figsize=(16, 12))
    for i, col in enumerate(potential_features):
        plt.subplot(2, 2, i+1)
        # 5개 레이블의 밀도 분포를 한 그래프에 겹쳐서 시각화
        sns.kdeplot(data=df_scaled, x=col, hue='Label', common_norm=False, fill=True, alpha=0.15)
        plt.title(f"Global Distribution: {col}")
        plt.xlabel("Scaled Value")
    plt.tight_layout()
    plt.savefig("global_feature_distributions.png")
    plt.show()
    
    print("\n3. 전역 특징 공간 시각화 (PCA 다중 클래스 클러스터 맵)...")
    # 클래스 불균형이 심하므로 시각화 품질을 위해 클래스별 균등 샘플링 진행
    sampled_dfs = []
    # 각 클래스당 최대 3000개씩 추출 (데이터가 적은 클래스는 존재하는 만큼만)
    max_sample_per_class = 3000 
    
    for label in target_labels:
        class_subset = df_scaled[df_scaled['Label'] == label]
        if len(class_subset) > 0:
            sample_n = min(max_sample_per_class, len(class_subset))
            sampled_dfs.append(class_subset.sample(n=sample_n, random_state=42))
            
    df_global_sample = pd.concat(sampled_dfs)
    X_sample = df_global_sample.drop(columns=['Label'])
    y_sample = df_global_sample['Label']
    
    # 2차원 공간 투영
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sample)
    
    plt.figure(figsize=(12, 9))
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1], 
        hue=y_sample, 
        alpha=0.7, 
        style=y_sample,
        palette='Set1'
    )
    plt.title("2D Global PCA Map: Benign vs All Attack Classes")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("global_pca_space_map.png")
    plt.show()
    
    print("\n4. 클래스 중심점 간 기하학적 거리 행렬 계산...")
    unique_labels = np.unique(y_sample)
    centers = {label: np.mean(X_pca[y_sample == label], axis=0) for label in unique_labels}
    
    dist_matrix = pd.DataFrame(index=unique_labels, columns=unique_labels, dtype=np.float32)
    for l1 in unique_labels:
        for l2 in unique_labels:
            dist_matrix.loc[l1, l2] = np.linalg.norm(centers[l1] - centers[l2])
            
    print("\n[각 클래스 중심점 간 유클리드 거리 행렬]")
    print(dist_matrix.round(4))

if __name__ == "__main__":
    clean_file_path = r"C:\ids2018_data\nids_advanced_cleaned.csv"
    analyze_all_nids_classes(clean_file_path)