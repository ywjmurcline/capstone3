from pathlib import Path
import random
from typing import List, Dict
# from psychopy import prefs
# prefs.general['audioLib'] = ['pygame'] 
import subprocess
from psychopy import visual, core, event, gui, data, sound
from datetime import datetime
from dotenv import load_dotenv
import os
load_dotenv()



# ============ Debug SetUp ==============

FULLSCREEN = False # set to True during experiment, can set to false during debugging, otherwise, you might be stucked in a full screen display and cannot exsit, LOL
MAX_SECONDS = 3

# ========== DATA Path =========== #
SAVE_DATA_PATH = "/Users/lily/Documents/myApps/Capstone_Saveme/1_data_archive/426/2_emotions"


# ============ Experiment SetUp ==============

# 1. Collect Information #
exp_info = {"participant": "your_name", "session": "001", "time": datetime.now()}
dlg = gui.DlgFromDict(exp_info, title="Emotion Video Task")
if not dlg.OK:
    core.quit()

# initiate window
win = visual.Window(
    size=(1300, 700),
    fullscr=FULLSCREEN,
    units="pix",
    color="black",
    waitBlanking=True
)

# after window is defined, extract window height
WIN_WIDTH = win.size[0]
WIN_HEIGHT = win.size[1]
print("win: ", win.size, WIN_WIDTH, WIN_HEIGHT)

# Let PsychoPy's ExperimentHandler manages data files
exp_name = "emotion_video_task"
this_exp = data.ExperimentHandler(
    name=exp_name,
    extraInfo=exp_info,
    dataFileName=os.path.join(SAVE_DATA_PATH, "psychopy", f"{exp_name}_{exp_info['session']}_{exp_info['participant']}_{exp_info['time']}")
)

# ============ Arduino Writer ==============
USE_ARDUINO = True
ARDUINO_PATH = os.path.join(SAVE_DATA_PATH, "ppg")


import serial
import time
import csv
# connect to arduino
if USE_ARDUINO:

    SERIAL_PORT = "/dev/cu.usbmodem1101"

    ser = serial.Serial(SERIAL_PORT, 115200, timeout=0.001)
    time.sleep(2.0)  # many Arduino boards reset when serial opens

    ppgFile = open(f"{ARDUINO_PATH}/max30105_data_{exp_name}_{exp_info['session']}_{exp_info['time']}_{exp_info['participant']}.csv", 'w', newline='', encoding='utf-8')
    writer = csv.writer(ppgFile)
    writer.writerow(['pc_time_s', 'arduino_time_ms', 'redValue', 'ir_value', 'marker'])


# ========== Display Parameters =========== #
# This section contains some parameters related to PsychPy display

EXPERIMENT_INSTRUCTION_TEXT_HEIGHT = 70 * (WIN_HEIGHT / 2500)


# ========== Experiment Parameters =========== #
# This section contains some parameters related to the experiment design

WASHOUT_SEC = 2  # (seconds)     
VIDEO_CHUNK_SEC = 500 # set to none


# 1. Prepare Videos #
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shufflers.video_shufflers import demo_video_shufflers, trial_video, video_shufflers, tiny_trial_video, washout_path
from shufflers.video_shufflers import StimulationVideo

# ============ Experiment Emotion Discrete Tags ===========
emotions = {
    "好笑",
    "生气",
    "恶心",
    "激动",
    "害怕",
    "惊讶",
    "快乐",
    "悲伤",
    "中性"
}

# VIDEO_LIST = tiny_trial_video() 
VIDEO_LIST = video_shufflers("/Users/lily/Documents/capstone/Code/videos/DECAF/Videos-DECAF/Video Stimuli/decaf_videos.csv", 4, "/Users/lily/Documents/capstone/Code/temp_video_reserve/temp.json")

TRIAL_VIDEO_LIST = trial_video()



# ========== Utilities =========== #
# This section contains utilities functions, such as play video, collect feedback, etc.
# don't use MovieStim, there's tone of decoding issues


def play_video(video: StimulationVideo, max_seconds: int = 1000000000, chunk_seconds: int = VIDEO_CHUNK_SEC):
    """
        播放视频。
        - 每次连续播放最多 chunk_seconds 秒
        - 到时后自动暂停
        - 按空格继续播放同一视频
        - 直到视频自然结束
    """

    videoStartSent = False
    videoEndSent = False

    event.clearEvents(eventType='keyboard')

    movie = VlcMovieStim(
        win, 
        video.path, 
        size=scale_video(video.width, video.height), 
        loop=False
    )

    video_label = visual.TextStim(
        win,
        text=video.id,
        height=80,
        pos=(-600,  500),
        color="white",
        font="Arial"
    )
    
    win.flip()
    this_exp.addData("video_start_time", core.getTime())
    movie.play()

    timer = core.Clock()

    chunk_timer = core.Clock()

    while not movie.isFinished:
        keys = event.getKeys(keyList=["escape"])
        if "escape" in keys:
            if not videoEndSent:
                if USE_ARDUINO:
                    try:
                        ser.write(b'E')
                        ser.flush()
                        writer.writerow([core.getTime(), "", "", "", "VIDEO_END_PC"])
                    except:
                        pass
            this_exp.abort()
            win.close()
            core.quit()
    
        if(movie.status == 1) and not videoStartSent:
            print("started")
            if USE_ARDUINO:
                ser.write(b'S')
                ser.flush()
                writer.writerow([core.getTime(), "", "", "", "VIDEO_START_PC"])
            videoStartSent = True

        if USE_ARDUINO:
            while ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                # print(f"line: {line}")
                if not line:
                    continue

                parts = line.split(',')
                # print(parts)
                if len(parts) == 4:
                    # print(parts)
                    arduino_t, red_value, ir_value, marker = parts
                    writer.writerow([core.getTime(), arduino_t, red_value, ir_value, marker])


        movie.draw()
        video_label.draw()
        win.flip()
        # core.wait(0.001)
    
        # 自然播放结束
        if movie.isFinished or timer.getTime() > max_seconds:
            if not videoEndSent:
                print("finished")
                if USE_ARDUINO: 
                    ser.write(b'E')
                    ser.flush()
                    writer.writerow([core.getTime(), "", "", "", "VIDEO_END_PC"])
                videoEndSent = True
            break


        # 连续播放达到 40 秒 -> 暂停 -> 等待空格继续
        if chunk_timer.getTime() >= chunk_seconds:
            print(f"paused after {chunk_seconds} seconds")

            if USE_ARDUINO: 
                ser.write(b'P')
                ser.flush()
                writer.writerow([core.getTime(), "", "", "", "VIDEO_PAUSED_PC"])
            # 暂停视频
            movie.pause()

            # 等待被试 / 主试按空格继续
            press_space_to_continue(message = "该视频已播放 40 秒。\n\n请检查超声id和PPG传感器。\n按“空格”继续播放该视频。\n按“ESC”退出实验。")

            # 清掉旧按键，避免空格残留
            event.clearEvents()
            show_fixation(duration=2.0)
            event.clearEvents(eventType='keyboard')

            # 继续播放
            movie.play()
            if USE_ARDUINO: 
                ser.write(b'R')
                ser.flush()
                writer.writerow([core.getTime(), "", "", "", "VIDEO_RESUMED_PC"])
            # 重置这一段 40 秒计时器
            chunk_timer.reset()

    movie.stop()
    this_exp.addData("video_end_time", core.getTime())

def choose_seven_point(type="valence", top_message="", left_message="left", right_message="right", middle_message=""):
    event.clearEvents(eventType='keyboard')
    currentIndex = random.randint(0, 6)
    # image file names
    files = [
        'img1.png', 'img2.png', 'img3.png', 'img4.png',
        'img5.png', 'img6.png', 'img7.png'
    ]
    x_positions = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15] 

    images = [
        visual.ImageStim(win, image=f"images/{type}/" + f, pos=(x * WIN_WIDTH, -0.01 * WIN_WIDTH), size=(0.04 * WIN_WIDTH, 0.04 * WIN_WIDTH))
        for f, x in zip(files, x_positions)
    ]

    highlights = [
        visual.Rect(
            win,
            pos=(x * WIN_WIDTH, -0.01 * WIN_WIDTH),
            width=0.045 * WIN_WIDTH,
            height=0.045 * WIN_WIDTH,
            lineColor='yellow',
            fillColor=None,
            opacity=0
        )
        for x in x_positions
    ]

    instruction = visual.TextStim(
        win,
        text=top_message,
        pos=(0, 0.15 * WIN_HEIGHT),
        height=EXPERIMENT_INSTRUCTION_TEXT_HEIGHT,
        wrapWidth=WIN_WIDTH * 0.8,
        font="Hei"
    )

    annotation_left = visual.TextStim(
        win,
        text=left_message,
        pos=(-0.15 * WIN_WIDTH, 0.025* WIN_WIDTH),
        height= 0.8 * EXPERIMENT_INSTRUCTION_TEXT_HEIGHT,
        wrapWidth=WIN_WIDTH * 0.2,
        font="Hei"
    )

    annotation_middle = visual.TextStim(
        win,
        text=middle_message,
        pos=(0,  0.025* WIN_WIDTH),
        height= 0.8 * EXPERIMENT_INSTRUCTION_TEXT_HEIGHT,
        wrapWidth=WIN_WIDTH * 0.2,
        font="Hei"
    )

    annotation_right = visual.TextStim(
        win,
        text=right_message,
        pos=(0.15 * WIN_WIDTH, 0.025* WIN_WIDTH),
        height= 0.8 * EXPERIMENT_INSTRUCTION_TEXT_HEIGHT,
        wrapWidth=WIN_WIDTH * 0.2,
        font="Hei"
    )

    
    clock = core.Clock()

    chosen = False
    while not chosen:
        keys = event.getKeys(keyList=['left', 'right', 'space', 'escape'], timeStamped=clock)

        for key, rt in keys:
            if key == 'left':
                currentIndex = max(0, currentIndex - 1)
            elif key == 'right':
                currentIndex = min(6, currentIndex + 1)
            elif key == 'space':
                rating = currentIndex + 1
                print('Rating:', rating, 'RT:', rt)
                chosen = True
                # win.close()
                # core.quit()
            elif key == 'escape':
                win.close()
                core.quit()

        for i, hl in enumerate(highlights):
            hl.opacity = 1 if i == currentIndex else 0

        instruction.draw()
        annotation_left.draw()
        annotation_middle.draw()
        annotation_right.draw()
        for img in images:
            img.draw()
        for hl in highlights:
            hl.draw()
        win.flip()
    
    return (rating, rt)

def ask_seen_before_lr():
    event.clearEvents(eventType='keyboard')
    options = ["没看过", "看过"]
    currentIndex = random.randint(0, 1)
    clock = core.Clock()

    instruction = visual.TextStim(
        win,
        text="在本次实验之前，你是否看过该视频？\n\n用左右键选择，按空格确认",
        pos=(0, 0.15 * WIN_HEIGHT),
        height=EXPERIMENT_INSTRUCTION_TEXT_HEIGHT,
        wrapWidth=WIN_WIDTH * 0.8,
        font="Hei"
    )

    option_stims = []
    highlight_boxes = []

    x_positions = [-0.1, 0.1]

    for opt, x in zip(options, x_positions):
        stim = visual.TextStim(
            win,
            text=opt,
            pos=(x * WIN_WIDTH, -0.05 * WIN_HEIGHT),
            height=EXPERIMENT_INSTRUCTION_TEXT_HEIGHT,
            font="Hei"
        )
        option_stims.append(stim)

        box = visual.Rect(
            win,
            pos=(x * WIN_WIDTH, -0.05 * WIN_HEIGHT),
            width=0.15 * WIN_WIDTH,
            height=0.1 * WIN_HEIGHT,
            lineColor="yellow",
            fillColor=None,
            opacity=0
        )
        highlight_boxes.append(box)

    while True:
        keys = event.getKeys(keyList=['left', 'right', 'space', 'escape'], timeStamped=clock)

        for key, rt in keys:
            if key == 'left':
                currentIndex = max(0, currentIndex - 1)
            elif key == 'right':
                currentIndex = min(len(options) - 1, currentIndex + 1)
            elif key == 'space':
                event.clearEvents()
                return (currentIndex + 1, rt) # 1=没看过, 2=看过
            elif key == 'escape':
                win.close()
                core.quit()

        for i, box in enumerate(highlight_boxes):
            box.opacity = 1 if i == currentIndex else 0

        instruction.draw()
        for stim in option_stims:
            stim.draw()
        for box in highlight_boxes:
            box.draw()

        win.flip()

def ask_emotional_effect_lr():
    event.clearEvents(eventType='keyboard')
    options = [
        "仍然有明显影响",
        "有影响\n但比第一次弱",
        "几乎没有影响"
    ]

    currentIndex = random.randint(0, 2)
    clock = core.Clock()

    instruction = visual.TextStim(
        win,
        text="如果你之前看过该视频，它现在对你的情绪影响如何？\n\n用左右键选择，按空格确认\n(如果上一题写的是没有看过，随便选就好)",
        pos=(0, 0.15 * WIN_HEIGHT),
        height=EXPERIMENT_INSTRUCTION_TEXT_HEIGHT,
        wrapWidth=WIN_WIDTH * 0.8,
        font="Hei"
    )

    option_stims = []
    highlight_boxes = []

    x_positions = [-0.15, 0.0, 0.15]

    for opt, x in zip(options, x_positions):
        stim = visual.TextStim(
            win,
            text=opt,
            pos=(x * WIN_WIDTH, -0.05 * WIN_HEIGHT),
            height=0.8 * EXPERIMENT_INSTRUCTION_TEXT_HEIGHT,
            wrapWidth=0.15 * WIN_WIDTH,
            font="Hei"
        )
        option_stims.append(stim)

        box = visual.Rect(
            win,
            pos=(x * WIN_WIDTH, -0.05 * WIN_HEIGHT),
            width=0.17 * WIN_WIDTH,
            height=0.12 * WIN_HEIGHT,
            lineColor="yellow",
            fillColor=None,
            opacity=0
        )
        highlight_boxes.append(box)

    while True:
        keys = event.getKeys(keyList=['left', 'right', 'space', 'escape'], timeStamped=clock)

        for key, rt in keys:
            if key == 'left':
                currentIndex = max(0, currentIndex - 1)
            elif key == 'right':
                currentIndex = min(len(options) - 1, currentIndex + 1)
            elif key == 'space':
                event.clearEvents()
                return (currentIndex + 1, rt)  # 1/2/3
            elif key == 'escape':
                win.close()
                core.quit()

        for i, box in enumerate(highlight_boxes):
            box.opacity = 1 if i == currentIndex else 0

        instruction.draw()
        for stim in option_stims:
            stim.draw()
        for box in highlight_boxes:
            box.draw()

        win.flip()

def choose_emotions(emotions):
    event.clearEvents(eventType='keyboard')
    """
    Display a rating table where each emotion is rated from 0 to 5.
    Default value for every emotion is 0.
    User clicks a radio button to change a rating.
    User presses SPACE to confirm and proceed.

    Returns
    -------
    ratings : dict
        Example: {'happy': 2, 'sad': 0, 'calm': 5}
    """

    # ---------- settings ----------
    scale_values = [0, 1, 2, 3, 4, 5]
    left_margin_x = -0.20 * WIN_WIDTH          # emotion label column
    start_x = -0.1 * WIN_WIDTH                # first rating column (0)
    col_spacing = 0.04 * WIN_WIDTH             # horizontal distance between buttons
    header_y = 0.08 * WIN_HEIGHT               # header row y
    first_row_y = 0.04 * WIN_HEIGHT            # first emotion row y
    row_spacing = 0.03 * WIN_HEIGHT            # vertical distance between rows

    radio_radius = 10                          # outer circle radius in px
    inner_radius = 6                           # filled center radius in px

    # default ratings: all set to 0
    ratings = {emotion: 0 for emotion in emotions}

    # ---------- instruction ----------
    instruction = visual.TextStim(
        win,
        text="请选择哪种情绪更符合你观看视频时的感受\n\n用鼠标点击选择，按“空格”确认",
        pos=(0, 0.18 * WIN_HEIGHT),
        height=EXPERIMENT_INSTRUCTION_TEXT_HEIGHT * 0.8,
        wrapWidth=0.4 * WIN_WIDTH,
        color="white",
        font="Hei"
    )

    # ---------- header ----------
    header_left = visual.TextStim(
        win,
        text="情绪",
        pos=(left_margin_x, header_y),
        height=0.9 * EXPERIMENT_INSTRUCTION_TEXT_HEIGHT * 0.8,
        anchorHoriz='center',
        color="white",
        font="Hei"
    )

    header_scale = []
    for i, val in enumerate(scale_values):
        x = start_x + i * col_spacing
        label_text = str(val)
        if val == 0:
            label_text = "0\n(完全不符合)"
        elif val == 5:
            label_text = "5\n(非常符合)"
        else:
            label_text = str(val) + "\n"

        stim = visual.TextStim(
            win,
            text=label_text,
            pos=(x, header_y),
            height=0.7 * EXPERIMENT_INSTRUCTION_TEXT_HEIGHT  * 0.8,
            alignText='center',
            anchorHoriz='center',
            color="white",
            font="Hei"
        )
        header_scale.append(stim)

    # ---------- build table objects ----------
    emotion_labels = []
    radio_buttons = []   # each item: list of dicts for one row

    for row_idx, emotion in enumerate(emotions):
        y = first_row_y - row_idx * row_spacing

        # emotion label
        emotion_label = visual.TextStim(
            win,
            text=emotion,
            pos=(left_margin_x, y),
            height=0.9 * EXPERIMENT_INSTRUCTION_TEXT_HEIGHT * 0.8,
            anchorHoriz='center',
            color="white",
            font="Hei"
        )
        emotion_labels.append(emotion_label)

        row_buttons = []
        for col_idx, value in enumerate(scale_values):
            x = start_x + col_idx * col_spacing

            outer_circle = visual.Circle(
                win,
                radius=radio_radius,
                pos=(x, y),
                lineColor="white",
                fillColor=None,
                lineWidth=2
            )

            inner_circle = visual.Circle(
                win,
                radius=inner_radius,
                pos=(x, y),
                lineColor=None,
                fillColor="white",
                opacity=1 if value == 0 else 0   # default selected = 0
            )

            click_area = visual.Rect(
                win,
                pos=(x, y),
                width=radio_radius * 2.6,
                height=radio_radius * 2.6,
                lineColor=None,
                fillColor=None,
                opacity=0
            )

            row_buttons.append({
                "value": value,
                "outer": outer_circle,
                "inner": inner_circle,
                "hitbox": click_area
            })

        radio_buttons.append(row_buttons)

    # # ---------- submit hint ----------
    # footer = visual.TextStim(
    #     win,
    #     text="Press SPACE to continue",
    #     pos=(0, -0.42 * WIN_HEIGHT),
    #     height=0.8 * EXPERIMENT_INSTRUCTION_TEXT_HEIGHT,
    #     color="white"
    # )

    mouse = event.Mouse(win=win)
    clock = core.Clock()
    mouse_was_down = False
    confirmed = False

    while not confirmed:
        # ----- keyboard -----
        keys = event.getKeys(keyList=["space", "escape"], timeStamped=clock)
        for key, rt in keys:
            if key == "space":
                confirmed = True
            elif key == "escape":
                win.close()
                core.quit()

        # ----- mouse click -----
        left_down = mouse.getPressed()[0]
        if left_down and not mouse_was_down:
            for row_idx, emotion in enumerate(emotions):
                for button in radio_buttons[row_idx]:
                    if button["hitbox"].contains(mouse):
                        ratings[emotion] = button["value"]
                        break

        mouse_was_down = left_down

        # ----- update selected state -----
        for row_idx, emotion in enumerate(emotions):
            selected_value = ratings[emotion]
            for button in radio_buttons[row_idx]:
                button["inner"].opacity = 1 if button["value"] == selected_value else 0

        # ----- draw -----
        instruction.draw()
        header_left.draw()
        for stim in header_scale:
            stim.draw()

        for row_idx in range(len(emotions)):
            emotion_labels[row_idx].draw()
            for button in radio_buttons[row_idx]:
                button["outer"].draw()
                button["inner"].draw()

    #     footer.draw()
        win.flip()

    print("Emotion ratings:", ratings)
    return ratings

def one_video_loop(video, trial=True, session_idx="trial"):

    washout(WASHOUT_SEC)

    play_video(video, max_seconds=MAX_SECONDS)

    valence, v_rt = choose_seven_point(
        type="valence", 
        top_message='观看视频时，你的心情是正向的还是负向的？可以参考画面里的小人\n\n请用键盘上的左右键选择，按“空格”确认',
        left_message="非常负面", 
        right_message="非常正面", 
        middle_message="中性")
    
    arousal, a_rt = choose_seven_point(
        type="arousal", 
        top_message='观看视频时，你的情绪强烈程度是？可以参考画面里的小人\n\n请用键盘上的左右键选择，按“空格”确认',
        left_message="非常平淡", 
        right_message="非常激烈", 
        middle_message="")
    
    ratings = choose_emotions(emotions)

    familiarity, f_rt = ask_seen_before_lr()

    habituation, h_rt = ask_emotional_effect_lr()

    if not trial:
        this_exp.addData("session_index", session_idx)
        # this_exp.addData("trial_index", video_idx)
        this_exp.addData("video_id", video.id)
        this_exp.addData("video_file", video.path)

        this_exp.addData("valence", valence)
        this_exp.addData("valence_rt", v_rt)

        this_exp.addData("arousal", arousal)
        this_exp.addData("arousal_rt", a_rt)

        for emo, score in ratings.items():
            this_exp.addData(f"emotion_{emo}", score)

        # this_exp.addData("dominance", dominance)
        this_exp.addData("familiar", familiarity)
        this_exp.addData("familiar_rt", f_rt)

        this_exp.addData("habituation", habituation)
        this_exp.addData("habituation_rt", h_rt)

        this_exp.nextEntry()

    
    if USE_ARDUINO:
        while ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            # print(f"line: {line}")
            if not line:
                continue

            parts = line.split(',')
            # print(parts)
            if len(parts) == 4:
                # print(parts)
                arduino_t, red_value, ir_value, marker = parts
                writer.writerow([core.getTime(), arduino_t, red_value, ir_value, marker])


def trail(videolist: List[StimulationVideo]):
    press_space_to_continue("模拟环节：\n\n在这个环节中，你讲适应眼镜的佩戴和电脑操作。\n\n实验中途，如有任何不适，请按键盘左上角的escape键退出。\n\n按“空格”继续。")

    for video in videolist:
        one_video_loop(video, trial=False)

    press_space_to_continue("模拟环节结束\n\n按“空格”继续。")





# define message and fix cross
msg = visual.TextStim(
    win,
    text="", 
    height=EXPERIMENT_INSTRUCTION_TEXT_HEIGHT, 
    color="white", 
    wrapWidth=WIN_WIDTH * 0.8,
    font="Hei"   # or another Chinese font
    )

# Optional: show instructions
instruction_text = '''你将观看一系列视频切片。

并在每个视屏之后描述你对视频的感受。

请在观看过程中尽量沉浸在视屏中。

实验中途，如有任何不适，请按键盘左上角的escape键退出。

按“空格”进入模拟环节。'''

press_space_to_continue(instruction_text)




def session(session_idx, videolist: List[StimulationVideo]):
    for video_idx, video in enumerate(videolist):
        one_video_loop(video, trial=False, session_idx=session_idx)

    press_space_to_continue(message = f"第{session_idx+1}个session结束了，休息一会儿吧。\n\n中途可以摘下眼镜，但如果发现红灯不亮了请叫人。\n\n休息好了按'空格'继续。")
    





play_beep() 
show_fixation(duration=0.5) 

# trail(TRIAL_VIDEO_LIST)

for index, experiment_session in VIDEO_LIST.items():
    session(index, experiment_session)



if USE_ARDUINO:
    ppgFile.flush()
    ppgFile.close()
    ser.close()

press_space_to_continue(message = "试验结束，谢谢你的参与！！！！！！！")

win.close()
core.quit()
