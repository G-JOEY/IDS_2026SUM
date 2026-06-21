# DQN based IDS with Feature Selection
- 1: Rule based feature selection
- 2(Improved Version): RFECV(Recursive Feature Elimination with Cross-Validation)

## State
$s_t = [x_1, x_2, ..., x_n]$
- $n$: feature 개수
- $x_i \in [0, 1]$ (normalized)

## Action
$a_t \in \{0,1\}$
- 0: Benign
- 1: Attack

## Reward
$$
R_t(s_t, a_t) =
\begin{cases}
+1, & a_t = y_t \\
-5, & a_t = 0, \ y_t = 1 \\
-1, & a_t = 1, \ y_t = 1
\end{cases}
$$
- $a_t$: 예측, $y_t$: 정답
- 미탐 시 -5, 오탐 시 -1

## DQN
### Architecture
- (input) -> 256 -> 128 -> 64 -> (output)
- Policy Net $\theta$, Target net $\theta^-$
- Replay Memory

### Curren Q-Value
$Q(s_t, a_t) = Q_{\theta}(s_t, a_t)$

### Next Q-Value
$Q(s_{t+1}, a_{t+1}) = \max_{a'}Q_{\theta^-}(s_{t+1}, a_{t+1})$

### Target Q-Value
$y_t = r_t + \gamma\max_{a'}Q(s_{t+1}, a_{t+1})(1-d_t)$
- $d_t$: Done flag, $d_t \in \{0, 1\}$

### Loss
#### TD Error
$\delta_t = |Q_{\theta}(s_t, a_t) - y_t|$

#### Focal Weight
$w_t = (1-e^{-\delta_t})^{\gamma_f}$

#### Loss
$Loss = \frac{1}{N}\sum^N_{i=1}w_i(Q(s_t, a_t) - y_t)^2$

## Hyperparameters
|Hyperparameter|Value|
|---|---|
|Learning Rate|0.00025|
|Replay Memory|60,000|
|Discount Factor $\gamma$|0.99|
|$epsilon$|0.01|
|$epsilon$ decay|0.996|
|$epsilon$ min|0.01|
|batch size|64|
|Target Update $\tau$|0.005|
|focal gamma $\gamma_f$|2.0|

## 구현상 특징
- 두 코드의 차이는 Feature selection에 있음

### Version 1
#### 1. 분산이 0.01 미만인 feature 제거
```python
# 1) 분산 기준 필터링: 오직 Train 데이터 기준으로만 학습(fit)
    var_thresh = 0.01
    selector = VarianceThreshold(threshold=var_thresh)
    
    X_train_var = selector.fit_transform(X_train_scaled)
    X_test_var = selector.transform(X_test_scaled) # Test는 통계치 유출 없이 변환만
    
    remaining_indices = selector.get_support(indices=True)
    df_train_filtered = pd.DataFrame(X_train_var, columns=remaining_indices)
    df_test_filtered = pd.DataFrame(X_test_var, columns=remaining_indices)
    print(f"-> 1단계 [분산 필터링(기준: < {var_thresh})]: {df_train_filtered.shape[1]}개 남음")
```
#### 2. 상관계수(Correlation Coefficient)가 0.95인 쌍 중 하나 제거
```python
# 2) 상관관계 기반 필터링: 오직 Train 데이터 상관관계 기준으로 탈락 항목 수집
    corr_thresh = 0.95
    corr_matrix = df_train_filtered.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_thresh)]
    
    # 최종 연산 데이터 확보
    X_train_final = df_train_filtered.drop(columns=to_drop).to_numpy(dtype=np.float32)
    X_test_final = df_test_filtered.drop(columns=to_drop).to_numpy(dtype=np.float32)
    
    print(f"-> 2단계 [상관관계 필터링(기준: > {corr_thresh})]: {X_train_final.shape[1]}개 남음")
    print(f"신경망 입력 피처 축소 정밀 완료: {orig_feature_count}개 -> {X_train_final.shape[1]}개")
```

### Version 2(Improved)
- RFECV(Recursive Feature Elimination with Cross-Validation)
- 최적의 Feature 개수를 자동으로 찾는 Feature Selection 기법

```python
    rfecv = RFECV(estimator=base_estimator, step=5, cv=3, scoring='accuracy', n_jobs=-1)
    rfecv.fit(X_train_scaled[sample_indices], y_train[sample_indices])
```

## 실험결과
