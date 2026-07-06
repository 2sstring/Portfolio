# %% [1] 라이브러리 임포트 및 시각화 설정
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import librosa
import sounddevice as sd

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
print("라이브러리 로드 완료")

# %% [2] 실제 MIMII 오디오 데이터 로드
normal_dir = '0_dB_valve/valve/id_02/normal'
abnormal_dir = '0_dB_valve/valve/id_02/abnormal'

normal_files = sorted(glob.glob(os.path.join(normal_dir, '*.wav')))
abnormal_files = sorted(glob.glob(os.path.join(abnormal_dir, '*.wav')))

if not normal_files or not abnormal_files:
    raise FileNotFoundError("정상/비정상 wav 파일 경로를 확인하세요.")

normal_audio_path = normal_files[0]
abnormal_audio_path = abnormal_files[0]

sr_target = 16000
y_normal, sr = librosa.load(normal_audio_path, sr=sr_target)
y_abnormal, _ = librosa.load(abnormal_audio_path, sr=sr_target)

print(f"정상 파일: {normal_audio_path}")
print(f"비정상 파일: {abnormal_audio_path}")
print(f"샘플링 레이트: {sr}")
print(f"정상 길이: {len(y_normal)/sr:.2f} sec")
print(f"비정상 길이: {len(y_abnormal)/sr:.2f} sec")

# %% [3] 시간영역 파형 시각화 및 청음
fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ax[0].plot(np.arange(len(y_normal))/sr, y_normal)
ax[0].set_title('Normal Valve - Waveform')
ax[1].plot(np.arange(len(y_abnormal))/sr, y_abnormal)
ax[1].set_title('Abnormal Valve - Waveform')
ax[1].set_xlabel('Time (sec)')
plt.tight_layout()
plt.show()

print("정상 소리 재생")
sd.play(y_normal, sr)
sd.wait()

print("비정상 소리 재생")
sd.play(y_abnormal, sr)
sd.wait()

# %% [4] FFT 기반 특징 추출 함수
def positive_fft(signal, sr=16000, length=16000):
    if len(signal) < length:
        signal = np.pad(signal, (0, length - len(signal)))
    else:
        signal = signal[:length]

    signal = np.hanning(length) * signal
    Y = np.fft.fft(signal)
    N = len(Y) // 2
    amplitude = 2 * np.abs(Y[:N]) / (length / 2)
    freq = np.linspace(0, sr/2, N, endpoint=True)
    return amplitude, freq

def fft_result(data, sr=16000, window_sec=1.0):
    stride = int(sr * window_sec)
    windows = [data[i:i+stride] for i in range(0, len(data), stride)]
    amps = []
    for w in windows:
        amp, freq = positive_fft(w, sr=sr, length=stride)
        amps.append(amp)
    amps = np.array(amps)
    return freq, amps.mean(axis=0)

# %% [5] FFT 특징 비교 시각화
freq_n, fft_normal = fft_result(y_normal, sr=sr)
freq_a, fft_abnormal = fft_result(y_abnormal, sr=sr)

plt.figure(figsize=(12, 5))
plt.plot(freq_n, fft_normal, label='Normal')
plt.plot(freq_a, fft_abnormal, label='Abnormal')
plt.title('FFT Average Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim([0, 4000])
plt.legend()
plt.show()

print(f"FFT feature shape: {fft_normal.shape}")  # 보통 (8000,)
