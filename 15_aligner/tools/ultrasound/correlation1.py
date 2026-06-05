import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import re
import random
import os
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import seasonal_decompose
import scipy.io.wavfile as wav
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.ndimage import gaussian_filter1d
from einops import rearrange
from scipy.signal import stft
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

C = 343.00 #声速
Fc = 17000 # 开始频率
Tw = 0.01 #chirp时长
Tf = 0.0
PRT1 = 0.0
PRT2 = 0.0
B = 4000  #带宽
Fs = 48000 #采样频率
Ts = 1 / Fs
k = B / Tw
len_flag = round(Fs * Tf)
len_cycle = round(Fs * (Tw + PRT1))
len_chirp = round(Fs * Tw)
len_blank = round(6 * PRT2 * Fs)
Nfft = 1*round(Tw * Fs)  # fft值，根据需要确定
dist_min = 0.01  # 最小距离 m
dist_max = 1 # 最大距离 m
bi_flag = 0


padding_length = 421  # run meta.py to get


def generate_chirp(sample_rate, chirp_duration, start_freq, band_width):
    amp = 1
    B = band_width
    Tw = chirp_duration
    init_phase = 0.5*np.pi
    Fc = start_freq
    step = 1.0 / sample_rate
    t = np.arange(0.0, Tw, step, dtype='float')
    trans_sw_sin = amp * np.sin(init_phase + 2 * np.pi * (Fc * t + B / Tw / 2 * t ** 2))
    trans_sw_cos = amp * np.cos(init_phase + 2 * np.pi * (Fc * t + B / Tw / 2 * t ** 2))
    # plt.figure()
    # plt.plot(trans_sw_sin)
    return (trans_sw_sin, trans_sw_cos, t)


def generate_chirp_down(sample_rate, chirp_duration, start_freq, band_width):
    amp = 1
    B = band_width
    Tw = chirp_duration
    init_phase = 0.25*np.pi
    Fc = start_freq
    step = 1.0 / sample_rate
    t = np.arange(0.0, Tw, step, dtype='float')
    trans_sw_sin = amp * np.sin(init_phase + 2 * np.pi * ((Fc + B) * t - B / Tw * t ** 2))
    trans_sw_cos = amp * np.cos(init_phase + 2 * np.pi * ((Fc + B) * t - B / Tw * t ** 2))
    # plt.figure()
    # plt.plot(trans_sw_sin)
    return (trans_sw_sin, trans_sw_cos, t)


def generate_chirp_bilateral(sample_rate, chirp_duration, start_freq, band_width):
  amp = 1
  B = band_width
  Tw = chirp_duration
  init_phase = 0.0
  Fc = start_freq
  step = 1.0 / sample_rate
  t = np.arange(0.0, Tw/2, step, dtype='float')
  chirp_sin_up = amp * np.sin(init_phase + 2*np.pi * (Fc*t + B/Tw*t**2))
  chirp_cos_up = amp * np.cos(init_phase + 2*np.pi * (Fc*t + B/Tw*t**2))
  chirp_sin_down = amp * np.sin(init_phase + 2*np.pi * ((Fc+B)*t - B/Tw*t**2))
  chirp_cos_down = amp * np.cos(init_phase + 2*np.pi * ((Fc+B)*t - B/Tw*t**2))
  trans_sw_sin = np.hstack((chirp_sin_up, chirp_sin_down))
  trans_sw_cos = np.hstack((chirp_cos_up, chirp_cos_down))

  return trans_sw_sin, trans_sw_cos, t
def correlation_lags(in1_len, in2_len, mode='full'):
    if mode == "full":
        lags = np.arange(-in2_len + 1, in1_len)
    elif mode == "same":
        lags = np.arange(-in2_len + 1, in1_len)
        mid = lags.size // 2
        lag_bound = in1_len // 2
        if in1_len % 2 == 0:
            lags = lags[(mid - lag_bound):(mid + lag_bound)]
        else:
            lags = lags[(mid - lag_bound):(mid + lag_bound) + 1]
    elif mode == "valid":
        lag_bound = in1_len - in2_len
        if lag_bound >= 0:
            lags = np.arange(lag_bound + 1)
        else:
            lags = np.arange(lag_bound, 1)
    return lags


# return data and ref_sig alignment lag
def delay_cal(data, ref_sig):
    correlation = signal.correlate(data, ref_sig,
                                   mode="full")  # calculate corelation of data and ref_sig with shift (-in2_len, in1_len)
    lags = correlation_lags(data.size, ref_sig.size, mode="full")
    lag = lags[np.argmax(correlation)]
    return lag


def butter_bandpass(lowcut, highcut, Fs, order=5):
    nyq = 0.5 * Fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, Fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, Fs, order)
    y = lfilter(b, a, data)
    return y


def get_lag(original_data, trans_sw):
    interested_signal = original_data
    if interested_signal.dtype == 'int16':
        interested_signal = interested_signal / 32768
    ref_data = trans_sw  # 发射信号
    lag = delay_cal(interested_signal[48000:48000 + len_chirp * 6], ref_data)  # 求直达信号和发射信号的延迟

    # 过滤接收信号，只取FMCW带宽内的信号
    lowcut = Fc - 10
    highcut = Fc + B + 10
    ref_data_filtered = butter_bandpass_filter(ref_data, lowcut, highcut, Fs, order=5)
    lag2 = delay_cal(ref_data_filtered, ref_data)
    lag = int((lag + lag2) % (Fs * Tw))  # 确定延迟
    return lag


def get_matrix(original_data, lag, fixed_cycles = None):
    interested_signal = original_data
    if interested_signal.dtype == 'int16':
        interested_signal = interested_signal / 32768

    lowcut = Fc - 10
    highcut = Fc + B + 10
    # 计算chirp数量
    interested_signal_filtered = butter_bandpass_filter(interested_signal, lowcut, highcut, Fs, order=5)
    sig_cycles = int((len(interested_signal_filtered) - (lag)) / len_chirp)
    fixed_cycles = sig_cycles if fixed_cycles == None else fixed_cycles
    matrix_data = np.zeros((fixed_cycles, len_cycle))
    for i in range(0, min(fixed_cycles, sig_cycles)):  # 取每个chirp的接收信号
        matrix_data[i] = interested_signal_filtered[i * len_cycle + lag: i * len_cycle + lag + len_chirp]

    return matrix_data

# data: (fixed_cycles, len_cycle)
def get_corr(data, ref_data):
    corrs = []
    for mtx in data:
        correlation = signal.correlate(mtx, ref_data, mode="full")
        l = len(correlation) // 2
        correlation = correlation[l:l + 100]
        corrs.append(correlation)

    corrs = np.array(corrs)
    corrs = corrs[:, :]

    corrs_diff = np.diff(corrs, axis=0)
    # plt.pcolormesh(np.abs(np.transpose(corrs_diff)), vmin = 0, vmax = np.max(np.abs(corrs_diff))/8)
    # plt.show()
    return corrs[:], corrs_diff[:]



def filter(data):
    data = data.transpose(1, 0)
    # print(data.shape)
    for i in range(data.shape[0]):
        data[i] = gaussian_filter1d(data[i], sigma=3)
    return data.transpose(1, 0)


def normalize(data):
    data = data / np.linalg.norm(data)
    data = data / np.median(np.abs(data))
    return data

def preprocess_corr(original_data_mc):
    # chirp 生成
    trans_up_sin, trans_up_cos, _ = generate_chirp_bilateral(Fs, Tw, Fc, B)
    # trans_down_sin, trans_down_cos, _ = generate_chirp_down(Fs, Tw, Fc, B)

    corrs_all = []
    if original_data_mc.dtype == 'int16':
        original_data_mc = original_data_mc / 32768
    # 遍历2个通道
    for ch in range(1):

        data = original_data_mc[:,ch]

        # 上变频相关处理
        lag_up = get_lag(data, trans_up_cos)
        data_up = get_matrix(data, lag_up)
        corrs_up, corrs_diff_up = get_corr(data_up, trans_up_cos)
        print(corrs_up.shape, corrs_diff_up.shape)
        corrs_diff_up = corrs_diff_up.T
        corrs_up = corrs_up.T
        print(corrs_up.shape, corrs_diff_up.shape)
        for i in range(corrs_diff_up.shape[0]):
            corrs_diff_up[i] = gaussian_filter1d(corrs_diff_up[i], sigma=2)
        corrs_diff_up = corrs_diff_up / np.linalg.norm(corrs_diff_up)
        print(corrs_up.shape, corrs_diff_up.shape)
        corrs_all.append(corrs_up)
        corrs_all.append(corrs_diff_up)

        # 下变频相关处理
        # lag_down = get_lag(data, trans_down_cos)
        # data_down = get_matrix(data, lag_down)
        # corrs_down, corrs_diff_down = get_corr(data_down, trans_down_cos)
        # corrs_diff_down = corrs_diff_down.T
        # for i in range(corrs_diff_down.shape[0]):
        #     corrs_diff_down[i] = gaussian_filter1d(corrs_diff_down[i], sigma=2)
        # corrs_diff_down = corrs_diff_down / np.linalg.norm(corrs_diff_down)
        # corrs_all.append(corrs_diff_down)
    min_len = min(c.shape[1] for c in corrs_all)
    corrs_all = [c[:, :min_len] for c in corrs_all]
    print(corrs_all)
    return corrs_all

def preprocess_corr1(original_data_mc):
    # chirp 生成
    trans_up_sin, trans_up_cos, _ = generate_chirp(Fs, Tw, Fc, B)
    trans_down_sin, trans_down_cos, _ = generate_chirp_down(Fs, Tw, Fc, B)

    corrs_all = []
    if original_data_mc.dtype == 'int16':
        original_data_mc = original_data_mc / 32768
    # 遍历2个通道
    for ch in range(2):

        data = original_data_mc[:,ch]

        # 上变频相关处理
        lag_up = get_lag(data, trans_up_cos)
        data_up = get_matrix(data, lag_up)
        corrs_up, corrs_diff_up = get_corr(data_up, trans_up_cos)
        corrs_diff_up = corrs_diff_up.T
        for i in range(corrs_diff_up.shape[0]):
            corrs_diff_up[i] = gaussian_filter1d(corrs_diff_up[i], sigma=2)
        corrs_diff_up = corrs_diff_up / np.linalg.norm(corrs_diff_up)
        corrs_all.append(corrs_diff_up)

        # 下变频相关处理
        lag_down = get_lag(data, trans_down_cos)
        data_down = get_matrix(data, lag_down)
        corrs_down, corrs_diff_down = get_corr(data_down, trans_down_cos)
        corrs_diff_down = corrs_diff_down.T
        for i in range(corrs_diff_down.shape[0]):
            corrs_diff_down[i] = gaussian_filter1d(corrs_diff_down[i], sigma=2)
        corrs_diff_down = corrs_diff_down / np.linalg.norm(corrs_diff_down)
        corrs_all.append(corrs_diff_down)
    min_len = min(c.shape[1] for c in corrs_all)
    corrs_all = [c[:, :min_len] for c in corrs_all]
    return np.stack(corrs_all, axis=0)
