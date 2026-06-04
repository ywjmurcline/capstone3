# Bio-data Aligner — 多模态生理数据对齐工具

## 这个工具是做什么的？

这个工具的作用是：**把一次情绪实验里采集到的所有传感器数据，整理、对齐，最终打包成可以直接用于 AI 训练的 `.npz` 文件。**

每次实验中，参与者会观看一段情绪激发视频，同时佩戴多种传感器。这个工具负责把这些分散的原始文件"拼在一起"，让每条实验数据（trial）都对应同一段时间窗口里的所有传感器信号。

---

## 实验数据结构背景

每个 trial（试次）分为两个时间段：

| 时间段 | 说明 |
|--------|------|
| **washout（基线期）** | 视频播放前的安静等待阶段，用于消除上一段情绪的影响 |
| **video（刺激期）** | 观看情绪视频的阶段 |

每个 trial 采集的传感器信号：

| 传感器 | 文件类型 | 说明 |
|--------|----------|------|
| **Finger PPG** | `.csv` | 手指光电容积脉搏波，反映心率 |
| **Glass PPG** | `.csv` | 眼镜框上的 PPG，同上 |
| **Face Video** | `.mov/.mp4` | 参与者脸部录像，用于提取眼动（EAR、眨眼） |
| **Ultrasound PCM** | `.pcm` | 超声传感器原始音频数据 |
| **PsychoPy CSV** | `.csv` | 实验软件记录，含情绪标签（valence、arousal、emotion tag） |

---

## 三个脚本的分工

```
aligner.py        ← 第一步：整理元数据，生成 trials.json
aligner_data.py   ← 第二步：读取原始信号，输出 .npz 文件
aligner_check.py  ← 第三步（可选）：可视化检查 .npz 文件是否正常
```

`aligner_check.py` 不是必须跑的，但它是验证数据质量非常好的手段，建议在第一次使用或遇到问题时运行。

---


## 输入文件说明

### 1. `bio_data/` 文件夹结构

```
bio_data/
├── ppg/
│   ├── ppg_finger_emotion_video_task_2_ywj_2026-05-31 14:54:25.254380.csv
│   ├── ppg_glass_emotion_video_task_2_ywj_2026-05-31 14:54:25.254380.csv
│   └── ...（每个 trial 对应两个文件，finger 和 glass）
└── psychopy/
    ├── psychopy_emotion_video_task_2_ywj_2026-05-31 14:54:25.254380.csv
    └── ...（每个 trial 一个文件，有时同一 trial 会有多个重试版本）
```

文件名规律（程序依赖此规律做匹配）：

```
ppg_finger_emotion_video_task_{video_id}_{participant}_{timestamp}.csv
ppg_glass_emotion_video_task_{video_id}_{participant}_{timestamp}.csv
psychopy_emotion_video_task_{video_id}_{participant}_{timestamp}.csv
```

**关键原理**：同一个 trial 的三个文件，文件名末尾的 timestamp **完全相同**，程序就是用这个时间戳来配对的。

### 2. `participants.json`

记录被试英文缩写和数字 ID 的对应关系。

```json
{"ywj": 1, "lmk": 2}
```

### 3. `all_videos_select_1.csv`

刺激视频素材库。关键列：`video_name`、`tag`、`duration_seconds`、`absolute_path`。

程序用它查找每个 trial 的视频类别标签（如 `"Fear"`, `"Amusing"`）和视频时长。

### 4. PCM 文件夹

包含超声原始文件，文件名为纯数字：`0.pcm`, `1.pcm`, `2.pcm` …

```
0.pcm  → trial 0 的 washout 期
1.pcm  → trial 0 的 video 期
2.pcm  → trial 1 的 washout 期
3.pcm  → trial 1 的 video 期
...
规律：行号 // 2 = trial index，偶数行 = washout，奇数行 = video
```

### 5. 面部录像视频

采集整个实验过程的连续视频，格式 `.mov` 或 `.mp4`。  
**文件名必须是 `YYYY-MM-DD HH-MM-SS.mov` 格式**（例如 `2026-05-31 14-46-52.mov`），程序从文件名解析录像开始时刻。

> 注意：复制文件会改变系统 ctime，所以尽量别移动文件。实在不行，程序会fall back到文件名时间，所以复制后文件名不变即可。
这里没有写其他的align的办法，所以如果当时没有存视频时间，可以想办法倒推一下视频的开始时间，把文件名改成开始时间。

---

## 第一步：`aligner.py` — 生成 trials.json

### 运行方式

```bash
python aligner.py \
    --bio_dir   /path/to/bio_data \
    --video_csv /path/to/all_videos_select_1.csv \
    --pcm_dir   /path/to/pcm_folder \
    --run_dir   /path/to/run_output
```

### 交互式步骤

脚本会逐步引导你，每一步都可以选择跳过（输入 `Y`）或运行（输入 `N`）。每步结束时暂停让你检查，按 `Y` 继续。这样你可以在步骤之间手动编辑 `trials.json` 修正问题。

| 步骤 | 说明 |
|------|------|
| Step 1 | 扫描 PsychoPy CSV，按 video_id 分组，自动选最新的有数据文件 |
| Step 2 | 根据时间戳匹配 finger PPG 和 glass PPG 文件 |
| Step 3 | 检查 PPG 信号质量（连续 10 个相同值 = 传感器可能脱落）|
| Step 4 | 从 PsychoPy CSV 提取 valence / arousal / emotion_tag 标签 |
| Step 5 | 从视频素材 CSV 补充视频时长、参考情绪标签 |
| Step 6 | 扫描超声 PCM 文件，计算各文件时长，输出 `pcm_durations.csv` 供人工检查 |
| Step 7 | 按行号规则把 PCM 分配给各 trial（偶数行=washout，奇数行=video）|
| Step 8 | 为每条 trial 添加 `ultrasound_offset` 字段（默认为 0）|

**Step 6 需要人工检查**：如果某个 trial 的超声采集失败，请把 `pcm_durations.csv` 里对应行的 filename 清空，程序会把空行理解为"该 PCM 缺失"。
可以参考data/ywj里的例子。

### 输出

```
run_dir/
├── trials.json       ← 主要输出，aligner_data.py 的输入
└── pcm_durations.csv ← 供人工检查的超声时长表
```

`trials.json` 单条示例：

```json
{
    "index": 2,
    "video_file": "/path/to/videos/1408.mp4",
    "participant": "ywj",
    "duration_seconds": 124.533,
    "ultrasound_offset": {"washout": 0, "video": 0},
    "ori_file_path": {
        "psychopy_path":   "/path/to/psychopy/psychopy_...csv",
        "finger_ppg_path": "/path/to/ppg/ppg_finger_...csv",
        "glass_ppg_path":  "/path/to/ppg/ppg_glass_...csv",
        "ultra_sound_washout": "2.pcm",
        "ultra_sound_video":   "3.pcm"
    },
    "ground_truth": {
        "valence": 3.0,
        "arousal": 5.0,
        "emotion_tag": {"fear": 2.0, "happy": 1.0, "neutral": 0.0},
        "emotion_tag_reference": "Fear"
    },
    "error": []
}
```

`error` 字段可能出现的值：

| 值 | 含义 |
|----|------|
| `"finger_ppg_error"` | 手指 PPG 有平段（可能传感器脱落）|
| `"glass_ppg_error"` | 眼镜 PPG 有平段 |
| `"ultra_sound_washout_missing"` | washout 期超声缺失 |
| `"ultra_sound_video_missing"` | video 期超声缺失 |
| `"video_meta_not_found"` | 在素材库 CSV 里找不到对应视频 |

**有 `error` 的 trial 默认仍会尝试输出 npz**，有问题的信号会是空数组。训练时可根据 `error` 字段决定是否过滤。

---

## 第二步：`aligner_data.py` — 生成 .npz 文件

### 运行方式

```bash
python aligner_data.py \
    --trials_json  /path/to/trials.json \
    --video_path   /path/to/face_recording.mov \
    --participants /path/to/participants.json \
    --pcm_dir      /path/to/pcm_dir \
    --run_dir      /path/to/run_output \
    --output_dir   /path/to/npz_output \
    --segment_cooldown 30
```

参数说明：

| 参数 | 含义 |
|------|------|
| `--trials_json` | 第一步生成的 trials.json |
| `--video_path` | 参与者脸部录像（文件名须为 `YYYY-MM-DD HH-MM-SS.mov`）|
| `--participants` | 参与者 ID 映射表 |
| `--pcm_dir` | 超声 PCM 文件夹 |
| `--run_dir` | 第一步的运行输出文件夹（眼动缓存也存在这里）|
| `--output_dir` | `.npz` 文件输出文件夹 |
| `--segment_cooldown` | 眼动识别片段之间的冷却秒数（默认 30）|

### 关于 `--segment_cooldown`

EAR（眼睛纵横比）提取是视觉计算程序，非常吃 CPU。在 Mac 上长时间运行会导致电脑发烫，冷却时间是为了让 CPU 降温，避免过热降频或损伤硬件。

- **如果你的电脑散热没问题，可以设置为 `--segment_cooldown 0` 关闭等待。**
- 默认值 30 秒是保守的安全设置。

注意：时间对齐的误差大约在 **1–2 秒**左右。

### 关于 `ultrasound_offset`

如果超声信号的起止时间点需要微调，可以在 `trials.json` 的 `ultrasound_offset` 字段手动设置：

```json
"ultrasound_offset": {"washout": 0, "video": 1.5}
```

**重要规则：**
- offset **只能裁开始部分**，且只能是正数，也就是多裁掉一点。
- 超声处理时**默认前后各裁掉 0.5 秒**（去掉边缘噪声）。
- `offset` 的值**包含那个 0.5 秒**。也就是说：
  - `offset = 0` → 前面裁掉 0.5 秒（默认行为）
  - `offset = 1` → 前面裁掉 1 秒（只比默认多裁 0.5 秒，而不是多裁 1 秒）

### 对齐原理

**PPG 是时间基准**：PPG 文件的 `marker` 列里有 Arduino 硬件发出的 `VIDEO_START` / `VIDEO_END` 标记，程序用这些标记精确切出 washout 和 video 两段的绝对时间窗口。

**眼动对齐**：脸部录像从实验开始就一直在录，每帧的绝对时刻 = 视频文件的创建时间 + 帧的相对时间戳。程序用 PPG 给出的绝对时间窗口切出对应的眼动帧。

**超声**：每段 washout / video 各对应一个独立的 PCM 文件，直接整体转换为矩阵。

### 输出格式

每个 trial 生成一个 `.npz` 文件，命名为 `p{participant_id}_{trial_index}.npz`，内含 10 个数组：

| 数组名 | 含义 | 数据类型 |
|--------|------|----------|
| `washout_finger_ppg` | 基线期手指 PPG | float32 1D |
| `washout_glass_ppg` | 基线期眼镜 PPG | float32 1D |
| `washout_ultrasound` | 基线期超声矩阵 | float32 2D |
| `washout_ear` | 基线期逐帧 EAR 值 | float32 1D |
| `washout_blink` | 基线期逐帧眨眼标记（0/1）| int8 1D |
| `video_finger_ppg` | 视频期手指 PPG | float32 1D |
| `video_glass_ppg` | 视频期眼镜 PPG | float32 1D |
| `video_ultrasound` | 视频期超声矩阵 | float32 2D |
| `video_ear` | 视频期逐帧 EAR 值 | float32 1D |
| `video_blink` | 视频期逐帧眨眼标记（0/1）| int8 1D |

同时输出 `progress.json`，记录每个 trial 的处理状态（成功 / 跳过 / 错误原因）。

### 断点续跑

`aligner_data.py` 支持断点续跑。每处理完一个 trial，状态立刻写入 `progress.json`。重新运行时，程序会读取已有的 `progress.json`，**跳过 `"skipped": true` 或 `"npz"` 已有值的条目**，只处理还没完成的 trial。

如果你想**手动重跑某个 trial**，直接编辑 `progress.json`，把对应条目改成：

```json
{
    "index": 5,
    "skipped": false,
    "npz": null,
    ...
}
```

然后重新运行 `aligner_data.py`，它就会重新处理这条数据。

**读取 .npz 示例：**

```python
import numpy as np

data = np.load("p1_2.npz")
print(data.files)                  # 查看所有 key

ear  = data["video_ear"]           # shape: (n_frames,)
blink = data["video_blink"]        # shape: (n_frames,)，0 或 1
ppg  = data["video_finger_ppg"]    # shape: (n_samples,)
```


---

## 第三步：`aligner_check.py` — 可视化检查（可选）

这一步不是必须的，但是验证数据好不好非常直观的办法。

### 运行方式

```bash
# 检查单个文件
python aligner_check.py path/to/p2_2.npz

# 检查整个文件夹里所有 .npz
python aligner_check.py path/to/npz_folder/

# 自定义参数
python aligner_check.py path/to/p2_2.npz --seconds 30 --output_dir ./plots
```

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--seconds` | 只画前多少秒 | 20 |
| `--output_dir` | 图片输出目录 | 与 npz 同目录 |
| `--ppg_fs` | PPG 采样率 (Hz) | 25 |
| `--us_fs` | 超声采样率 (Hz) | 100 |
| `--ear_fs` | 视频帧率 (fps) | 60 |

每个 `.npz` 文件生成三张 PNG：

| 文件 | 内容 |
|------|------|
| `{stem}_ppg_plot.png` | 4 路 PPG 波形（2×2 网格）|
| `{stem}_ultrasound_plot.png` | 2 段超声热力图 |
| `{stem}_ear_plot.png` | EAR 曲线 + 眨眼红色阴影 |

---

## 完整流程图

```
原始数据
├── bio_data/
│   ├── psychopy/  (*.csv)
│   └── ppg/       (*.csv)
├── pcm/           (*.pcm, 0.pcm / 1.pcm / 2.pcm ...)
├── face_video.mov
├── all_videos_select.csv
└── participants.json

          ↓ python aligner.py

run_dir/
├── trials.json       ← 元数据总表（可手动编辑）
└── pcm_durations.csv ← 超声文件时长（需人工检查）

          ↓ python aligner_data.py

output_dir/
├── p1_0.npz
├── p1_1.npz
├── ...
└── progress.json     ← 处理状态记录

          ↓ python aligner_check.py（可选）

output_dir/
├── p1_0_ppg_plot.png
├── p1_0_ultrasound_plot.png
├── p1_0_ear_plot.png
└── ...
```

---

## 参考示例

`data/ywj/` 是一次完整跑下来留下的痕迹（生成的视频除外），可以直接对照参考：

```
data/ywj/
├── participants.json      ← 输入：被试 ID 映射表示例
├── trials.json            ← aligner.py 的最终输出，可看真实结构
├── pcm_durations.csv      ← Step 6 生成的超声时长表，可看格式
└── npz/
    └── progress.json      ← aligner_data.py 的处理状态记录
```

`ear/` 子文件夹里有带 EAR 标注的片段视频，是眼动提取时生成的副产物，可以用来肉眼验证眨眼识别是否正确，但不是最终训练数据的一部分。

---

## 常见问题

### Q：某个 trial 的 participant 字段是空的，被跳过了

`trials.json` 里对应条目的 `"participant"` 是空字符串，可能是第一个 trial 采集时 PsychoPy 还没记录被试名字。

**解决**：手动在 `trials.json` 里填上 `"participant": "xxx"`，然后再运行 `aligner_data.py`。

---

### Q：程序报"找不到 participant 'xxx' 的 id"

`participants.json` 里没有这个名字，检查拼写是否一致（区分大小写）。

---

### Q：眼动数据全是空数组

检查以下几点：
1. `video_path` 的文件名格式是否正确（`YYYY-MM-DD HH-MM-SS.mov`）
2. 面部录像的录制时间和 PPG 采集时间是否在同一天、同一时区
3. PPG 文件里是否有 Arduino `VIDEO_START` / `VIDEO_END` 标记（若两个都缺失，程序会询问是否跳过该 trial）

---

### Q：想重新跑某一步，不想从头开始

`aligner.py` 每一步都会询问"跳过? Y/N"，直接跳过已完成的步骤即可，程序会读取上次写入 `trials.json` 的结果继续。

---

### Q：`trials.json` 里 `error` 非空的 trial 还会被输出 npz 吗？

**会**（默认行为：程序会提示你，你可以选择强制处理或跳过）。`error` 只是记录问题，不自动阻止输出。有问题的信号会是空数组（shape 为 `(0,)`）。训练时可根据 `error` 字段决定是否过滤。

---

### Q：Step 6 的 PCM 时长计算结果不对

`get_pcm_duration()` 目前用的是默认参数（48000 Hz, 32-bit 单声道），如需修改请在 `aligner.py` 里调整对应参数。
