# 나눔고딕 폰트 설치
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm -rf ~/.cache/matplotlib

# 런타임 재시작 후 이 셀을 다시 실행하여 폰트 설정 적용
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 나눔고딕 폰트 경로 설정
fm.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.fft import rfft, rfftfreq

try:
    from scipy.io import loadmat
except Exception:
    loadmat = None

plt.rcParams["axes.unicode_minus"] = False
np.random.seed(42)

DATASET_NAME = "TODO: 사용한 공개 데이터셋명"
DATASET_URL = "TODO: 데이터 출처 URL"
NORMAL_FILE = "TODO: 정상 데이터 파일명"
FAULT_FILE = "TODO: 이상 데이터 파일명"
FS = 12000  # TODO: 데이터셋 설명서에 맞게 샘플링 주파수 수정


# 방법 C: MAT 파일 구조 확인
# CWRU 계열 데이터는 .mat 형식인 경우가 많습니다.

RUN_MAT_INSPECT = True
MAT_PATH = "/content/raw/Time_Normal_1_098.mat"

if RUN_MAT_INSPECT:
    if loadmat is None:
        raise RuntimeError("scipy.io.loadmat을 사용할 수 없습니다.")
    mat_norm = loadmat(MAT_PATH)
    keys = [k for k in mat_norm.keys() if not k.startswith("__")]
    print("MAT 변수 목록:")
    for k in keys:
        arr = np.asarray(mat_norm[k])
        print(k, arr.shape, arr.dtype)
else:
    print("MAT 구조 확인을 사용하려면 RUN_MAT_INSPECT = True로 바꾸세요.")

# 방법 C: MAT 파일 구조 확인
# CWRU 계열 데이터는 .mat 형식인 경우가 많습니다.

RUN_MAT_INSPECT = True
MAT_PATH = "/content/raw/OR007_6_1_136.mat"

if RUN_MAT_INSPECT:
    if loadmat is None:
        raise RuntimeError("scipy.io.loadmat을 사용할 수 없습니다.")
    mat_OR007 = loadmat(MAT_PATH)
    keys = [k for k in mat_OR007.keys() if not k.startswith("__")]
    print("MAT 변수 목록:")
    for k in keys:
        arr = np.asarray(mat_OR007[k])
        print(k, arr.shape, arr.dtype)
else:
    print("MAT 구조 확인을 사용하려면 RUN_MAT_INSPECT = True로 바꾸세요.")

MAT_PATH = "/content/raw/B007_1_123.mat"

if loadmat is None:
    raise RuntimeError("scipy.io.loadmat을 사용할 수 없습니다.")
mat_B007 = loadmat(MAT_PATH)
keys = [k for k in mat_B007.keys() if not k.startswith("__")]
print("MAT 변수 목록:")
for k in keys:
    arr = np.asarray(mat_B007[k])
    print(k, arr.shape, arr.dtype)

MAT_PATH = "/content/raw/B014_1_190.mat"

if loadmat is None:
    raise RuntimeError("scipy.io.loadmat을 사용할 수 없습니다.")
mat_B014 = loadmat(MAT_PATH)
keys = [k for k in mat_B014.keys() if not k.startswith("__")]
print("MAT 변수 목록:")
for k in keys:
    arr = np.asarray(mat_B014[k])
    print(k, arr.shape, arr.dtype)

# TODO: 실제 데이터 로딩 후 아래 두 배열을 교체하세요.
# 스켈레톤이 바로 실행되도록 임시 예제 신호를 넣어둡니다.

duration = 2.0
t = np.arange(0, duration, 1 / FS)

normal_signal = mat_norm["X098_DE_time"].ravel()
OR007_fault_signal = mat_OR007["X136_DE_time"].ravel()
B007_fault_signal = mat_B007["X123_DE_time"].ravel()
B014_fault_signal = mat_B014["X190_DE_time"].ravel()

# impact_positions = np.arange(0, len(t), int(FS / 90))
# for pos in impact_positions:
#     if pos + 40 < len(fault_signal):
#         fault_signal[pos:pos+40] += np.hanning(40) * np.random.uniform(1.5, 2.2)

print("normal:", normal_signal.shape)
print("fault:", OR007_fault_signal.shape)
print("fault:", B007_fault_signal.shape)
print("fault:", B014_fault_signal.shape)

def plot_time_waveform(signal, fs, title, seconds=0.2):
    n = min(len(signal), int(fs * seconds))
    x = np.arange(n) / fs
    plt.figure(figsize=(12, 3))
    plt.plot(x, signal[:n])
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(alpha=0.3)
    plt.show()

plot_time_waveform(normal_signal, FS, "정상 진동 신호")
plot_time_waveform(OR007_fault_signal, FS, "이상 진동 신호")
plot_time_waveform(B007_fault_signal, FS, "이상 진동 신호")
plot_time_waveform(B014_fault_signal, FS, "이상 진동 신호")


def calculate_features(signal):
    signal = np.asarray(signal).ravel()
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    kurtosis = stats.kurtosis(signal, fisher=False)
    skewness = stats.skew(signal)
    crest_factor = peak / rms if rms > 0 else np.nan
    std = np.std(signal)
    mean_abs = np.mean(np.abs(signal))
    return {
        "mean": np.mean(signal),
        "std": std,
        "rms": rms,
        "peak": peak,
        "kurtosis": kurtosis,
        "skewness": skewness,
        "crest_factor": crest_factor,
        "mean_abs": mean_abs,
    }

feature_df = pd.DataFrame([
    {"state": "normal", **calculate_features(normal_signal)},
    {"state": "fault_OR007", **calculate_features(OR007_fault_signal)},
    {"state": "fault_B007", **calculate_features(B007_fault_signal)},
    {"state": "fault_B014", **calculate_features(B014_fault_signal)},
])

display(feature_df)

plot_cols = ["rms", "peak", "kurtosis", "crest_factor"]
feature_df.set_index("state")[plot_cols].T.plot(kind="bar", figsize=(10, 4))
plt.title("정상/이상 특징값 비교")
plt.ylabel("Feature value")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.show()

def compute_fft(signal, fs):
    signal = np.asarray(signal).ravel()
    signal = signal - np.mean(signal)
    n = len(signal)
    window = np.hanning(n)
    spectrum = np.abs(rfft(signal * window)) / n
    freq = rfftfreq(n, 1 / fs)
    return freq, spectrum

def plot_fft(signal, fs, title, max_freq=1000):
    freq, spectrum = compute_fft(signal, fs)
    mask = freq <= max_freq
    plt.figure(figsize=(12, 4))
    plt.plot(freq[mask], spectrum[mask])
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(alpha=0.3)
    plt.show()

plot_fft(normal_signal, FS, "정상 신호 FFT")
plot_fft(OR007_fault_signal, FS, "이상 신호 FFT_OR007")
plot_fft(B007_fault_signal, FS, "이상 신호 FFT_B007")
plot_fft(B014_fault_signal, FS, "이상 신호 FFT_B014")

def window_features(signal, fs, window_sec=0.2, step_sec=0.1):
    signal = np.asarray(signal).ravel()
    window = int(fs * window_sec)
    step = int(fs * step_sec)
    rows = []
    for start in range(0, len(signal) - window + 1, step):
        seg = signal[start:start + window]
        rows.append({
            "time_sec": start / fs,
            **calculate_features(seg),
        })
    return pd.DataFrame(rows)

normal_win = window_features(normal_signal, FS)
fault_OR007_win = window_features(OR007_fault_signal, FS)
fault_B007_win = window_features(B007_fault_signal, FS)
fault_B014_win = window_features(B014_fault_signal, FS)
normal_win["state"] = "normal"
fault_OR007_win["state"] = "fault_OR007"
fault_B007_win["state"] = "fault_B007"
fault_B014_win["state"] = "fault_B014"
trend_df = pd.concat([normal_win, fault_OR007_win,fault_B007_win,fault_B014_win], ignore_index=True)

display(trend_df.head())

for col in ["rms", "kurtosis", "crest_factor"]:
    plt.figure(figsize=(12, 3))
    for state, group in trend_df.groupby("state"):
        plt.plot(group["time_sec"], group[col], label=state)
    plt.title(f"구간별 {col} 추세")
    plt.xlabel("Time (s)")
    plt.ylabel(col)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

normal_baseline = normal_win[["rms", "kurtosis", "crest_factor"]].agg(["mean", "std"])

rms_threshold = normal_baseline.loc["mean", "rms"] + 3 * normal_baseline.loc["std", "rms"]
kurtosis_threshold = 5.0
crest_threshold = 4.0

def diagnose(row):
    reasons = []
    if row["rms"] > rms_threshold:
        reasons.append("RMS 증가")
    if row["kurtosis"] > kurtosis_threshold:
        reasons.append("충격성 증가")
    if row["crest_factor"] > crest_threshold:
        reasons.append("Crest Factor 증가")

    if len(reasons) >= 2:
        return "위험", ", ".join(reasons)
    if len(reasons) == 1:
        return "주의", reasons[0]
    return "정상", "-"

diagnosis = fault_B007_win.copy()
diagnosis[["diagnosis", "reason"]] = diagnosis.apply(
    lambda row: pd.Series(diagnose(row)),
    axis=1,
)

display(diagnosis[["time_sec", "rms", "kurtosis", "crest_factor", "diagnosis", "reason"]].head(20))
print(diagnosis["diagnosis"].value_counts())


summary = f'''
# 공개 진동 데이터 분석 결과 요약

## 사용 데이터
- 데이터셋: {DATASET_NAME}
- 출처: {DATASET_URL}
- 샘플링 주파수: {FS} Hz

## 특징값 비교
- 정상 RMS: {feature_df.loc[feature_df['state']=='normal', 'rms'].iloc[0]:.4f}
- 이상 RMS: {feature_df.loc[feature_df['state']=='fault', 'rms'].iloc[0]:.4f}
- 정상 Kurtosis: {feature_df.loc[feature_df['state']=='normal', 'kurtosis'].iloc[0]:.4f}
- 이상 Kurtosis: {feature_df.loc[feature_df['state']=='fault', 'kurtosis'].iloc[0]:.4f}
- 정상 Crest Factor: {feature_df.loc[feature_df['state']=='normal', 'crest_factor'].iloc[0]:.4f}
- 이상 Crest Factor: {feature_df.loc[feature_df['state']=='fault', 'crest_factor'].iloc[0]:.4f}

## 진단 기준 예시
- RMS 주의 기준: 정상 RMS 평균 + 3σ = {rms_threshold:.4f}
- Kurtosis 주의 기준: {kurtosis_threshold}
- Crest Factor 주의 기준: {crest_threshold}

## CBM 해석
- TODO: 어떤 특징값이 이상을 가장 잘 설명했는지 작성
- TODO: 점검 또는 정비를 지시할 기준 작성
- TODO: 실제 현장 적용 시 필요한 추가 데이터 작성
'''

print(summary)


