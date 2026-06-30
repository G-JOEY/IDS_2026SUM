# EDA for LSTM
# 황금 밸런스 (20%~40%) 구간 탐색
# 희귀 공격 유실 진단
# 일자별 황금 에피소드 생성 가능 수, 공격 종류별 실질 노출 에피소드 수 그래프 출력


import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze_nids_dataset(folder_path, max_steps=1000, stride=200, low_ratio=0.20, high_ratio=0.40):
    """
    폴더 내의 일자별 CSV 파일을 순회하며 데이터 분포와 
    우리가 설계한 에피소드 구성 조건 만족 여부를 시뮬레이션합니다.
    """
    file_list = glob.glob(os.path.join(folder_path, "*.csv"))
    if not file_list:
        raise FileNotFoundError(f"지정한 경로에 CSV 파일이 존재하지 않습니다: {folder_path}")
        
    print(f"================================================================")
    print(f" 데이터 분석 및 에피소드 풀 시뮬레이션 시작 (총 {len(file_list)}개 파일)")
    print(f" 설정값: Episode Steps={max_steps} | Stride={stride} | Target Ratio={low_ratio*100}% ~ {high_ratio*100}%")
    print(f"================================================================")
    
    summary_data = []
    all_rare_attacks_info = {}
    
    for idx, file_path in enumerate(file_list):
        file_name = os.path.basename(file_path)
        print(f"\n[{idx+1}/{len(file_list)}] 파일 분석 중: {file_name}")
        
        # 1. 고속 로드를 위해 필요한 컬럼만 추출
        df_label = pd.read_csv(file_path, low_memory=False, usecols=['Label'])
        df_label['Label'] = df_label['Label'].str.strip()
        
        total_rows = len(df_label)
        if total_rows < max_steps:
            print(f"  -> [경고] 데이터 행 수({total_rows})가 max_steps({max_steps})보다 작아 스킵합니다.")
            continue
            
        # 2. 클래스 기본 통계
        label_counts = df_label['Label'].value_counts()
        benign_count = label_counts.get('Benign', 0)
        total_attack_count = total_rows - benign_count
        attack_ratio_pct = (total_attack_count / total_rows) * 100
        
        # 3. 이진 라벨 및 누적합 생성
        y_binary = (df_label['Label'] != 'Benign').astype(np.int8).values
        prefix_sum = np.zeros(total_rows + 1, dtype=np.int32)
        np.cumsum(y_binary, out=prefix_sum[1:])
        
        # 4. 세부 공격 종류 파악
        attack_types = {k: v for k, v in label_counts.to_dict().items() if k != 'Benign'}
        
        # 5. 에피소드 풀 시뮬레이션 (Stride 적용 슬라이딩)
        valid_start_range = total_rows - max_steps
        pool_normal_count = 0  # 20% ~ 40% 만족하는 일반/혼합 구간
        pool_rare_count = 0    # 희귀 공격이 '단 1개라도' 들어있는 구간 수
        
        # 이 파일에서 발견된 공격 종류별로 유효 구간 내 포함 횟수 트래킹
        attack_inclusion = {k: 0 for k in attack_types.keys()}
        raw_labels_np = df_label['Label'].values
        
        for start_idx in range(0, valid_start_range, stride):
            end_idx = start_idx + max_steps
            bin_attack_cnt = prefix_sum[end_idx] - prefix_sum[start_idx]
            bin_attack_ratio = bin_attack_cnt / max_steps
            
            # [조건 A] 황금 밸런스 일반/혼합 구간 카운트
            if low_ratio <= bin_attack_ratio <= high_ratio:
                pool_normal_count += 1
                
            # [조건 B] 희귀 공격 포함 구간 시뮬레이션
            # 해당 구간에 포함된 세부 라벨들 확인
            slice_labels = raw_labels_np[start_idx:end_idx]
            unique_slice_labels = set(slice_labels)
            
            for atk in attack_types.keys():
                if atk in unique_slice_labels:
                    attack_inclusion[atk] += 1
                    
        # 6. 통계 저장
        summary_data.append({
            "file": file_name,
            "total_rows": total_rows,
            "benign": benign_count,
            "attack": total_attack_count,
            "attack_ratio": attack_ratio_pct,
            "pool_normal": pool_normal_count,
            "attack_details": attack_types,
            "attack_inclusion": attack_inclusion
        })
        
        # 출력 요약
        print(f"  -> 총 데이터: {total_rows:,}건 (정상: {benign_count:,}건 / 공격: {total_attack_count:,}건 [{attack_ratio_pct:.2f}%])")
        print(f"  -> 발견된 공격 종류: {list(attack_types.keys())}")
        print(f"  -> [시뮬레이션] 만족하는 황금 밸런스(20~40%) 에피소드 수: {pool_normal_count:,}개")
        for atk, inc_cnt in attack_inclusion.items():
            print(f"     * 공격 [{atk}]이(가) 포함되는 실질 에피소드 수 (Stride={stride}): {inc_cnt:,}개")
            
    # ========================================================
    # 7. 종합 리포트 및 시각화
    # ========================================================
    print("\n" + "="*80)
    print(f"{'전체 데이터셋 종합 요약 리포트':^75}")
    print("="*80)
    print(f"{'파일명':<25} | {'총 행수':<12} | {'공격 비율':<10} | {'황금에피소드 풀 (20~40%)':<15}")
    print("-" * 80)
    
    file_names_plot = []
    gold_pools_plot = []
    total_attacks_distribution = {}
    total_inclusion_distribution = {}
    
    for s in summary_data:
        print(f"{s['file']:<25} | {s['total_rows']:<12,} | {s['attack_ratio']:<9.2f}% | {s['pool_normal']:<15,}개")
        file_names_plot.append(s['file'])
        gold_pools_plot.append(s['pool_normal'])
        
        for k, v in s['attack_details'].items():
            total_attacks_distribution[k] = total_attacks_distribution.get(k, 0) + v
        for k, v in s['attack_inclusion'].items():
            total_inclusion_distribution[k] = total_inclusion_distribution.get(k, 0) + v
            
    print("="*80)
    
    print("\n[위험 징후 탐지: 영원히 학습 불가능한 공격 검출]")
    missing_any = False
    for atk, total_cnt in total_attacks_distribution.items():
        inc_cnt = total_inclusion_distribution.get(atk, 0)
        if inc_cnt == 0:
            print(f"  ⚠️ [위험] 공격 종류 [{atk}] (총 {total_cnt:,}건 존재)은 현재 20~40% 조건 하에 에피소드가 0개 생성됩니다! (유실 확정)")
            missing_any = True
        else:
            print(f"  ✅ [안전] 공격 종류 [{atk}] ({total_cnt:,}건) -> 실질 노출 가능 에피소드 수: {inc_cnt:,}개")
            
    if not missing_any:
        print("  -> 다행히 현재 조건에서 영원히 외면받는 공격 종류는 없습니다.")
    print("="*80)
    
    # 8. 대시보드 시각화
    plt.figure(figsize=(16, 6))
    
    # 그래프 1: 일자별 황금 밸런스 에피소드 생성 가능 수
    plt.subplot(1, 2, 1)
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(file_names_plot)))
    plt.barh([f[:10] for f in file_names_plot], gold_pools_plot, color=colors)
    plt.title(f'Available Gold Episodes (20-40%) per File\n(Stride={stride}, Window=1000)')
    plt.xlabel('Number of Unique Scenario Episodes')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 그래프 2: 세부 공격 라벨별 실질 노출 가능 에피소드 수
    plt.subplot(1, 2, 2)
    atks = list(total_inclusion_distribution.keys())
    inc_counts = list(total_inclusion_distribution.values())
    
    if atks:
        plt.bar(atks, inc_counts, color='salmon', edgecolor='black')
        plt.title(f'Real Episode Exposure Count per Attack Type\n(How many episodes contain this attack?)')
        plt.ylabel('Episode Count')
        plt.xticks(rotation=45, ha='right')
        plt.yscale('log') # 공격 편향이 심할 수 있으므로 로그 스케일 적용
        plt.ylabel('Episode Count (Log Scale)')
        plt.grid(True, linestyle='--', alpha=0.4)
    else:
        plt.text(0.5, 0.5, 'No Attack Data Found', ha='center', va='center')
        
    plt.tight_layout()
    plt.show()

# ==========================================
# 실행부
# ==========================================
if __name__ == "__main__":
    # 데이터가 저장되어 있는 루트 폴더 경로
    TARGET_FOLDER = r"C:\ids2018_data"
    
    # Stride를 200으로 설정하여 실제 강화학습 환경과 동일하게 시뮬레이션
    analyze_nids_dataset(
        folder_path=TARGET_FOLDER, 
        max_steps=1000, 
        stride=200, 
        low_ratio=0.20, 
        high_ratio=0.40
    )