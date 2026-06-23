import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from scipy import stats
from scipy.fft import rfft, rfftfreq
from scipy.io import loadmat


st.set_page_config(
    page_title="CWRU Bearing Dataset 분석",
    layout="wide"
)

plt.rcParams["axes.unicode_minus"] = False


# =========================
# 기본 설정
# =========================

DATASET_NAME = "CWRU Bearing Dataset"
DATASET_URL = "https://engineering.case.edu/bearingdatacenter"
DEFAULT_FS = 12000  # CWRU 12kHz 데이터 기준. 사용하는 파일에 따라 48kHz 여부 확인 필요


DEFAULT_FILES = {
    "normal": {
        "path": "raw/Time_Normal_1_098.mat",
        "key": "X098_DE_time",
        "label": "정상"
    },
    "fault_OR007": {
        "path": "raw/OR007_6_1_136.mat",
        "key": "X136_DE_time",
        "label": "OR007 불량"
    },
    "fault_B007": {
        "path": "raw/B007_1_123.mat",
        "key": "X123_DE_time",
        "label": "B007 불량"
    },
    "fault_B014": {
        "path": "raw/B014_1_190.mat",
        "key": "X190_DE_time",
        "label": "B014 불량"
    },
}


# =========================
# 함수 정의
# =========================

@st.cache_data
def load_mat_signal(path: str, key: str) -> np.ndarray:
    mat = loadmat(path)

    if key not in mat:
        available_keys = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(
            f"{path} 파일에서 {key} 변수를 찾을 수 없습니다. "
            f"사용 가능한 변수: {available_keys}"
        )

    return np.asarray(mat[key]).ravel()


@st.cache_data
def inspect_mat(path: str) -> pd.DataFrame:
    mat = loadmat(path)

    rows = []
    for key in mat.keys():
        if key.startswith("__"):
            continue

        arr = np.asarray(mat[key])
        rows.append({
            "variable": key,
            "shape": str(arr.shape),
            "dtype": str(arr.dtype)
        })

    return pd.DataFrame(rows)


def calculate_features(signal: np.ndarray) -> dict:
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


def window_features(
    signal: np.ndarray,
    fs: int,
    window_sec: float = 0.2,
    step_sec: float = 0.1
) -> pd.DataFrame:
    signal = np.asarray(signal).ravel()

    window = int(fs * window_sec)
    step = int(fs * step_sec)

    if window <= 0 or step <= 0:
        raise ValueError("window_sec와 step_sec는 0보다 커야 합니다.")

    rows = []

    for start in range(0, len(signal) - window + 1, step):
        seg = signal[start:start + window]
        rows.append({
            "time_sec": start / fs,
            **calculate_features(seg),
        })

    return pd.DataFrame(rows)


def compute_fft(signal: np.ndarray, fs: int):
    signal = np.asarray(signal).ravel()
    signal = signal - np.mean(signal)

    n = len(signal)
    window = np.hanning(n)

    spectrum = np.abs(rfft(signal * window)) / n
    freq = rfftfreq(n, 1 / fs)

    return freq, spectrum


def plot_time_waveform(signal: np.ndarray, fs: int, title: str, seconds: float = 0.2):
    n = min(len(signal), int(fs * seconds))
    x = np.arange(n) / fs

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(x, signal[:n])
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)

    return fig


def plot_fft(signal: np.ndarray, fs: int, title: str, max_freq: int = 1000):
    freq, spectrum = compute_fft(signal, fs)
    mask = freq <= max_freq

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(freq[mask], spectrum[mask])
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)

    return fig


def diagnose(row, rms_threshold: float, kurtosis_threshold: float, crest_threshold: float):
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


def show_missing_file_guide(missing_files):
    st.error("실행에 필요한 .mat 파일이 없습니다.")
    st.write("아래 파일들을 `raw/` 폴더에 넣은 뒤 다시 실행하세요.")
    st.code("\n".join(missing_files), language="text")

    st.info(
        "GitHub에 데이터 파일을 같이 올리지 않을 경우, "
        "사용자가 직접 raw 폴더에 파일을 넣어야 합니다."
    )


# =========================
# 화면 구성
# =========================

st.title("CWRU Bearing Dataset 진동 분석")
st.caption(
    "정상/불량 베어링 진동 신호의 파형, FFT, 특징값, 간단한 진단 기준을 확인하는 Streamlit 앱입니다."
)

with st.sidebar:
    st.header("설정")

    fs = st.number_input(
        "샘플링 주파수 FS (Hz)",
        min_value=1000,
        max_value=100000,
        value=DEFAULT_FS,
        step=1000
    )

    waveform_seconds = st.slider(
        "파형 표시 구간 (초)",
        min_value=0.05,
        max_value=2.0,
        value=0.2,
        step=0.05
    )

    max_freq = st.slider(
        "FFT 최대 주파수 (Hz)",
        min_value=100,
        max_value=6000,
        value=1000,
        step=100
    )

    window_sec = st.slider(
        "특징 추출 윈도우 (초)",
        min_value=0.05,
        max_value=1.0,
        value=0.2,
        step=0.05
    )

    step_sec = st.slider(
        "특징 추출 간격 (초)",
        min_value=0.05,
        max_value=1.0,
        value=0.1,
        step=0.05
    )

    kurtosis_threshold = st.number_input(
        "Kurtosis 기준",
        value=5.0,
        step=0.5
    )

    crest_threshold = st.number_input(
        "Crest Factor 기준",
        value=4.0,
        step=0.5
    )


st.subheader("1. 데이터 파일 확인")

missing_files = []
for item in DEFAULT_FILES.values():
    if not os.path.exists(item["path"]):
        missing_files.append(item["path"])

if missing_files:
    show_missing_file_guide(missing_files)
    st.stop()


selected_name = st.selectbox(
    "MAT 구조를 확인할 파일 선택",
    options=list(DEFAULT_FILES.keys()),
    format_func=lambda x: DEFAULT_FILES[x]["label"]
)

selected_info = DEFAULT_FILES[selected_name]
inspect_df = inspect_mat(selected_info["path"])
st.dataframe(inspect_df, use_container_width=True)


st.subheader("2. 신호 로딩")

signals = {}

for name, info in DEFAULT_FILES.items():
    try:
        signals[name] = load_mat_signal(info["path"], info["key"])
    except Exception as e:
        st.error(f"{info['path']} 로딩 중 오류 발생: {e}")
        st.stop()

signal_summary = pd.DataFrame([
    {
        "state": name,
        "label": DEFAULT_FILES[name]["label"],
        "length": len(signal),
        "duration_sec": len(signal) / fs
    }
    for name, signal in signals.items()
])

st.dataframe(signal_summary, use_container_width=True)


st.subheader("3. 시간 영역 파형")

tabs = st.tabs([DEFAULT_FILES[name]["label"] for name in signals.keys()])

for tab, name in zip(tabs, signals.keys()):
    with tab:
        fig = plot_time_waveform(
            signals[name],
            fs,
            f"{DEFAULT_FILES[name]['label']} 진동 신호",
            seconds=waveform_seconds
        )
        st.pyplot(fig)


st.subheader("4. 특징값 비교")

feature_df = pd.DataFrame([
    {
        "state": name,
        "label": DEFAULT_FILES[name]["label"],
        **calculate_features(signal)
    }
    for name, signal in signals.items()
])

st.dataframe(feature_df, use_container_width=True)

plot_cols = ["rms", "peak", "kurtosis", "crest_factor"]

fig, ax = plt.subplots(figsize=(10, 4))
feature_df.set_index("label")[plot_cols].T.plot(kind="bar", ax=ax)
ax.set_title("정상/이상 특징값 비교")
ax.set_ylabel("Feature value")
ax.tick_params(axis="x", rotation=0)
ax.grid(axis="y", alpha=0.3)
st.pyplot(fig)


st.subheader("5. FFT 분석")

fft_tabs = st.tabs([DEFAULT_FILES[name]["label"] for name in signals.keys()])

for tab, name in zip(fft_tabs, signals.keys()):
    with tab:
        fig = plot_fft(
            signals[name],
            fs,
            f"{DEFAULT_FILES[name]['label']} FFT",
            max_freq=max_freq
        )
        st.pyplot(fig)


st.subheader("6. 구간별 특징 추세")

trend_list = []

for name, signal in signals.items():
    temp = window_features(
        signal,
        fs,
        window_sec=window_sec,
        step_sec=step_sec
    )
    temp["state"] = name
    temp["label"] = DEFAULT_FILES[name]["label"]
    trend_list.append(temp)

trend_df = pd.concat(trend_list, ignore_index=True)

st.dataframe(trend_df.head(50), use_container_width=True)

for col in ["rms", "kurtosis", "crest_factor"]:
    fig, ax = plt.subplots(figsize=(12, 3))

    for label, group in trend_df.groupby("label"):
        ax.plot(group["time_sec"], group[col], label=label)

    ax.set_title(f"구간별 {col} 추세")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(col)
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)


st.subheader("7. 간단 진단 예시")

normal_win = trend_df[trend_df["state"] == "normal"]

normal_baseline = normal_win[["rms", "kurtosis", "crest_factor"]].agg(["mean", "std"])

rms_threshold = (
    normal_baseline.loc["mean", "rms"]
    + 3 * normal_baseline.loc["std", "rms"]
)

st.write(f"RMS 기준: 정상 RMS 평균 + 3σ = `{rms_threshold:.4f}`")
st.write(f"Kurtosis 기준: `{kurtosis_threshold}`")
st.write(f"Crest Factor 기준: `{crest_threshold}`")

diagnosis_target = st.selectbox(
    "진단 대상 선택",
    options=[name for name in signals.keys() if name != "normal"],
    format_func=lambda x: DEFAULT_FILES[x]["label"]
)

diagnosis_df = trend_df[trend_df["state"] == diagnosis_target].copy()

diagnosis_df[["diagnosis", "reason"]] = diagnosis_df.apply(
    lambda row: pd.Series(
        diagnose(
            row,
            rms_threshold,
            kurtosis_threshold,
            crest_threshold
        )
    ),
    axis=1
)

st.dataframe(
    diagnosis_df[
        ["time_sec", "rms", "kurtosis", "crest_factor", "diagnosis", "reason"]
    ],
    use_container_width=True
)

st.write("진단 결과 개수")
diagnosis_count_df = (
    diagnosis_df["diagnosis"]
    .value_counts()
    .rename_axis("diagnosis")
    .reset_index(name="count")
)

st.dataframe(diagnosis_count_df, use_container_width=True)


st.subheader("8. 요약")

normal_row = feature_df[feature_df["state"] == "normal"].iloc[0]
target_row = feature_df[feature_df["state"] == diagnosis_target].iloc[0]

summary = f"""
## 공개 진동 데이터 분석 결과 요약

### 사용 데이터
- 데이터셋: {DATASET_NAME}
- 출처: {DATASET_URL}
- 샘플링 주파수: {fs} Hz

### 특징값 비교
- 정상 RMS: {normal_row["rms"]:.4f}
- 선택 불량 RMS: {target_row["rms"]:.4f}
- 정상 Kurtosis: {normal_row["kurtosis"]:.4f}
- 선택 불량 Kurtosis: {target_row["kurtosis"]:.4f}
- 정상 Crest Factor: {normal_row["crest_factor"]:.4f}
- 선택 불량 Crest Factor: {target_row["crest_factor"]:.4f}

### 진단 기준 예시
- RMS 주의 기준: 정상 RMS 평균 + 3σ = {rms_threshold:.4f}
- Kurtosis 주의 기준: {kurtosis_threshold}
- Crest Factor 주의 기준: {crest_threshold}

### 해석
- RMS는 진동 에너지 크기 변화를 보는 지표입니다.
- Kurtosis는 충격성 신호가 얼마나 강한지 보는 지표입니다.
- Crest Factor는 순간 피크가 RMS 대비 얼마나 큰지 보는 지표입니다.
- 실제 예지보전 모델에서는 이 기준값만으로 판단하기보다, 여러 파일을 윈도우 단위로 나누고 라벨을 붙여 분류 모델을 학습시키는 방식이 적합합니다.
"""

st.markdown(summary)
