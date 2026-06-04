import numpy as np
import scipy.fftpack as fftp
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.fftpack import fft
import os
from tools.ultrasound.correlation import preprocess_corr

# params
C = 343.00 #声速
Fc = 17000 # 开始频率
Tw = 0.1 #chirp时长
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

#生成单边信号的发射信号
def generate_chirp(sample_rate, chirp_duration, start_freq, band_width):
    amp = 1
    B = band_width
    Tw = chirp_duration
    init_phase = 0.0
    Fc = start_freq
    step = 1.0 / sample_rate
    t = np.arange(0.0, Tw, step, dtype='float')
    trans_sw_sin = amp * np.sin(init_phase + 2 * np.pi * (Fc * t + B / Tw / 2 * t ** 2))
    trans_sw_cos = amp * np.cos(init_phase + 2 * np.pi * (Fc * t + B / Tw / 2 * t ** 2))

    # plt.figure()
    # plt.plot(trans_sw_sin)
    return (trans_sw_sin, trans_sw_cos, t)

#生成双边信号的发射信号
def generate_chirp_bilateral(sample_rate, chirp_duration, start_freq, band_width):
    amp = 1
    B = band_width
    Tw = chirp_duration
    init_phase = 0.0
    Fc = start_freq
    step = 1.0 / sample_rate
    t = np.arange(0.0, Tw / 2, step, dtype='float')
    t = t[:len_cycle//2]
    chirp_sin_up = amp * np.sin(init_phase + 2 * np.pi * (Fc * t + B / Tw * t ** 2))
    chirp_cos_up = amp * np.cos(init_phase + 2 * np.pi * (Fc * t + B / Tw * t ** 2))
    chirp_sin_down = amp * np.sin(init_phase + 2 * np.pi * ((Fc + B) * t - B / Tw * t ** 2))
    chirp_cos_down = amp * np.cos(init_phase + 2 * np.pi * ((Fc + B) * t - B / Tw * t ** 2))
    trans_sw_sin = np.hstack((chirp_sin_up, chirp_sin_down))

    trans_sw_cos = np.hstack((chirp_cos_up, chirp_cos_down))
    return (trans_sw_sin, trans_sw_cos, t)


#巴特沃斯带通
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

#求相关性
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

#求接收信号和发射信号的延迟，以实现对齐
def delay_cal(data, ref_sig):
    correlation = signal.correlate(data, ref_sig, mode="full")
    lags = correlation_lags(data.size, ref_sig.size, mode="full")
    lag = lags[np.argmax(correlation)]
    # plt.figure()
    # plt.plot(lags, correlation)
    return lag

#发射信号和接收信号在对齐后进行相乘，得到混频信号
def mixed_sw(data, trans_sw_cos, trans_sw_sin, dist_min, dist_max, B, C, Tw):
    mix_sw_cos = data * trans_sw_cos
    mix_sw_sin = data * trans_sw_sin
    lowcut_deltaF = 2 * dist_min * B / C / Tw
    highcut_deltaF = 2 * dist_max * B / C / Tw

    mix_sw_cos_bpf = butter_bandpass_filter(mix_sw_cos, lowcut_deltaF, highcut_deltaF, Fs, order=3)
    mix_sw_sin_bpf = butter_bandpass_filter(mix_sw_sin, lowcut_deltaF, highcut_deltaF, Fs, order=3)

    array_mix_sw = mix_sw_cos_bpf + 1j * mix_sw_sin_bpf
    return mix_sw_cos_bpf, mix_sw_sin_bpf, array_mix_sw

#对混频信号作傅里叶变换可以得到各个bin的IQ值
def compute_IQ(mixed_chirp, dist_idx, Nfft):
    N = round(Nfft // 2)

    phasor = fftp.fft(mixed_chirp, Nfft)
    phasor = phasor[0: N]
    phasor = phasor[dist_idx]

    est_I_vec = np.real(phasor)
    est_Q_vec = np.imag(phasor)
    est_I_Q_vec = est_I_vec + 1j * est_Q_vec

    return est_I_vec, est_Q_vec, est_I_Q_vec

#移走异常点
def remove_outlier(data):
    for i in range(6, len(data) - 6):
        if abs(data[i] - data[i - 1]) > 2 * (max(data[i - 6:i]) - min(data[i - 6:i])) \
                and abs(data[i + 3] - data[i]) > 2 * (
                max(data[i + 3:i + 8]) - min(data[i + 3:i + 8])):
            data[i] = data[i - 1]
    return data


def select_bin(IQ, file):   #生成每个bin的相位和幅度信号
    # I = np.real(IQ)
    # Q = np.imag(IQ)
    # IQ = []
    # for i in range(I.shape[0]):
    #     decompositionI = seasonal_decompose(I[i], freq=5, two_sided=False)
    #     trendI = decompositionI.trend
    #     decompositionQ = seasonal_decompose(Q[i], freq=5, two_sided=False)
    #     trendQ = decompositionQ.trend
    #     trendI = trendI[5:]
    #     trendQ = trendQ[5:]
    #     trendIQ = trendI + 1j*trendQ
    #     IQ.append(trendIQ)
    # IQ = np.array(IQ)
    # (num_of_bins, num_of_chirps) = IQ.shape
    # I = np.real(IQ)
    # Q = np.imag(IQ)
    am_map = np.abs(IQ)  # 幅度
    # plt.figure()
    # plt.pcolormesh(np.abs(am_map[:, :]), vmin=None, vmax=np.max(np.abs(am_map[:, :]))/1)
    # plt.colorbar()
    corr_scores = []   #表示前后两个chirp所有bin的幅度值的相关系数
    for i in range(1, IQ.shape[1]):
        cur = am_map[:, i]
        pre = am_map[:, i-1]
        corr_score = np.corrcoef(cur, pre)[0][1]
        corr_scores.append(corr_score)
    # plt.figure()
    # plt.plot(corr_scores)
    mincorr = np.min(corr_scores)
    # if mincorr < 0.95:
        # print(file)
    phase_map = np.angle(IQ)  # 相位
    phase_map = np.unwrap(phase_map, axis=1)
    am_map2 = []
    phase_map2 = []
    for bin_idx in range(0, IQ.shape[0]):
        am_bin = am_map[bin_idx, :]
        phase_bin = phase_map[bin_idx, :]
        phase_bin = np.diff(phase_bin)
        am_bin = np.diff(am_bin)
        phase_bin = remove_outlier(phase_bin)
        # phase_bin = np.unwrap(phase_bin)
        am_bin = remove_outlier(am_bin)
        am_bin = am_bin
        phase_map2.append(phase_bin)
        am_map2.append(am_bin)
    phase_map2 = np.array(phase_map2)  #幅度和相位的差分
    am_map2 = np.array(am_map2)
    # plt.figure()
    # for i in range(10):
    #     plt.plot(phase_map2[i*4],'.-')
    return am_map, phase_map




def gen_IQ(original_data, return_data):
    if bi_flag == 0:   #根据单边还是双边信号选择发射信号
        trans_sw_sin, trans_sw_cos, t = generate_chirp(Fs, Tw, Fc, B)
    else:
        trans_sw_sin, trans_sw_cos, t = generate_chirp_bilateral(Fs, Tw, Fc, B)
    interested_signal = original_data
    if interested_signal.dtype == 'int16':
        interested_signal = interested_signal / 32768
    ref_data = trans_sw_cos   #发射信号
    lag = delay_cal(interested_signal[:len_chirp * 3], ref_data) #求直达信号和发射信号的延迟

#过滤接收信号，只取FMCW带宽内的信号
    lowcut = Fc - 10
    highcut = Fc + B + 10
    interested_signal_filtered = butter_bandpass_filter(interested_signal, lowcut, highcut, Fs, order=5)
    ref_data_filtered = butter_bandpass_filter(ref_data, lowcut, highcut, Fs, order=5)
    lag2 = delay_cal(ref_data_filtered, ref_data)
    # print(lag2, interested_signal_filtered.shape)
    freq_search = np.linspace(0, Fs // 2, Nfft // 2)   #选择混频信号的搜索频率
    if bi_flag == 0:
        dist_search = freq_search * C * Tw / (2 * B)
    else:
        dist_search = freq_search * C * Tw / (2 * B * 2)
    dist_idx = (dist_search >= dist_min) & (dist_search <= dist_max) #跟据距离的最大值最小值确定bin的区间范围

    lag = int((lag + lag2) % (Fs * Tw)) #确定延迟

    # 计算chirp数量
    sig_cycles = int((len(interested_signal_filtered) - (lag)) / len_chirp)
    # print(sig_cycles)
    matrix_data = np.zeros((sig_cycles, len_cycle))
    for i in range(0, sig_cycles):  #取每个chirp的接收信号
        matrix_data[i] = interested_signal_filtered[ i* len_cycle + lag: i* len_cycle + lag + len_chirp]
    # 解调：与发射信号相乘
    mix_sw_cos = matrix_data * trans_sw_cos
    mix_sw_sin = matrix_data * trans_sw_sin

    lowcut_deltaF = 4 * dist_min * B / C / Tw   #根据最小最大距离确定混频信号转换成频域后的频率区间
    highcut_deltaF = 4 * dist_max * B / C / Tw
    mix_sw_cos_bpf = butter_bandpass_filter(mix_sw_cos, lowcut_deltaF, highcut_deltaF, Fs, order=3)
    mix_sw_sin_bpf = butter_bandpass_filter(mix_sw_sin, lowcut_deltaF, highcut_deltaF, Fs, order=3)
    array_mix_sw = mix_sw_cos_bpf + 1j * mix_sw_sin_bpf
    # 计算IQ
    N = round(Nfft // 2)
    phasor = fft(array_mix_sw[:, :], Nfft)
    phasor = phasor[:, 0: N]
    phasor = phasor[:, dist_idx]
    # shape = (bins, sig_cycles)
    est_I_Q_vec_sequence = phasor.T   #得到IQ信号
    # print(est_I_Q_vec_sequence.shape)
    return est_I_Q_vec_sequence


# def fmcw_pro(file_path, user_number, counter, gesture_category,session_counter, base_save_path='wo'):
#
#     fs = 48000
#
#     num_channels = 2  # 双通道音频
#     data = np.memmap(file_path, dtype=np.float32, mode='r')
#     data = data.reshape(-1, num_channels)
#
#     mics = data[int(0.5 * fs):int(4.5 * fs)]
#
#     am_maps = preprocess_corr(mics)
#
#     save_path = os.path.join(base_save_path, f"{gesture_category}-{counter % 1000}-{user_number}.npz")
#     #save_path = os.path.join(base_save_path, f"{user_number}-{gesture_category}-{session_counter}-{counter%5}.npz")
#     os.makedirs(base_save_path, exist_ok=True)
#     np.savez_compressed(save_path, datapre=np.array(am_maps[:, :, :]))
#     #print(np.array(am_maps).shape)
#     print(f"Saved: {save_path}")
def fmcw_pro(file_path, offset: float = 0.0):
    fs = 48000
    num_channels =1 # 双通道音频

    # 读入数据
    data = np.memmap(file_path, dtype=np.float32, mode='r')
    data = data.reshape(-1, num_channels)
    data = data[int(Fs * (0.5 + max((offset-0.5), 0))):int(-(Fs * 0.5))]
    # os.makedirs(base_save_path, exist_ok=True)



    am_maps = preprocess_corr(data)

    return np.array(am_maps[:, :, :])

        # 命名规则：0-idx-0.npz
    # save_path = f"{gesture_category}-{counter}-0.npz"

    # np.savez_compressed(save_path, datapre=np.array(am_maps[:, :, :]))
    # print(f"Saved: {save_path}")


# i = 0
# if __name__ == '__main__':
#     user_number = 0
#     base_dir = f'data/heart'

#     files = os.listdir(base_dir)

#     # 只取 .pcm 文件，并且按数字顺序排序
#     files = [f for f in files if f.endswith('.pcm')]
#     files.sort(key=lambda x: int(os.path.splitext(x)[0]))  # 去掉扩展名转成数字排序

#     i = 0
#     for session_counter in files:
#         session_path = os.path.join(base_dir, session_counter)

#         gesture_category = 0  # 每5个一类
#         class_local_index = i  # 类内编号

#         fmcw_pro(
#             session_path,
#             user_number,
#             class_local_index,  # index: 每类从0开始
#             gesture_category,  # label
#             i
#         )
#         i += 1

if __name__ == "__main__":
    fmcw_pro("/Users/lily/Documents/myApps/Capstone_Saveme/100_data_530/ywj/ywj531/1.pcm", 0, 0, 0, 0)