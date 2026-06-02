import random
from typing import List
from psychopy import visual, core, event, gui, data, sound
from datetime import datetime
from dotenv import load_dotenv
import os
load_dotenv()

# TODO
USE_ARDUINO=True

FULLSCREEN=False

# TODO
BY_SESSION=True

# TODO
MAX_SECONDS = 5
# MAX_SECONDS = 1000000000


# TODO
OUTPUT_DATA_FOLDER="/Users/lily/Documents/myApps/Capstone_Saveme/1_data_archive/523/2_emotions"

# VIDEO_LIST = "/Users/lily/Documents/myApps/Capstone_Saveme/5_stimulation_materials/all_videos.csv"
VIDEO_LIST = "/Users/lily/Documents/myApps/Capstone_Saveme/5_stimulation_materials/all_videos_select_1.csv"

# TODO
VIDEO_JSON_DIR = "/Users/lily/Documents/myApps/Capstone_Saveme/1_data_archive/510/videos"

emotions_tags = {
    "Trial": True,
    "Amusing": False,
    "Anger": False,
    "Disgust": False,
    "Exciting": False,
    "Fear": False,
    # "Shock": False,
    # "Funny": False,
    "Sad": False,
    "Happy": False,
    "Neutral": False,
    # "Awe": False,
    # "Liking": False
    # "中性": True
}

emotions_cn = {
    "Amusing": '好笑',
    "Anger": '生气',
    "Disgust": '恶心',
    # "Exciting": False,
    "Fear": '恐惧',
    # "Shock": False,
    # "Funny": False,
    "Sad": '悲伤',
    "Happy": '快乐',
    "Neutral": '中性',
    # "Awe": False,
    # "Liking": False
    # "中性": True
}

# ===============================


class config():
    def __init__(self):
        self.SAVE_DATA_PATH = OUTPUT_DATA_FOLDER
        # Check if path exists, create if not
        if self.SAVE_DATA_PATH:
            if not os.path.exists(self.SAVE_DATA_PATH):
                os.makedirs(self.SAVE_DATA_PATH)
                print(f"Created directory: {self.SAVE_DATA_PATH}")
            else:
                # Path exists, check if empty
                if os.listdir(self.SAVE_DATA_PATH):  # Directory not empty
                    response = input(f"Directory '{self.SAVE_DATA_PATH}' is not empty. Override? (Y/N): ").upper()
                    if response == 'Y':
                        print("Proceeding with override...")
                        # Continue with your code
                    else:
                        print("Exiting or handling non-override case...")
                        # Handle cancellation (e.g., exit, raise exception, or change path)
                        exit(0)  # or raise Exception("User cancelled operation")
                else:
                    print("Directory exists and is empty. Proceeding...")
        else:
            print("ERROR: OUTPUT_DATA_FOLDER environment variable not set!")
            # Handle missing environment variable

        self.TEMP_VIDEO_PATH = os.path.join(self.SAVE_DATA_PATH, "temp_video.json")

        self.USE_ARDUINO = os.environ.get("USE_ARDUINO")

        self.ARDUINO_PATH = os.path.join(self.SAVE_DATA_PATH, "ppg")
        if not os.path.exists(self.ARDUINO_PATH):
            os.makedirs(self.ARDUINO_PATH)
            print(f"Created directory: {self.ARDUINO_PATH}")

        self.PSYCHOPY_PATH = os.path.join(self.SAVE_DATA_PATH, "psychopy")
        if not os.path.exists(self.PSYCHOPY_PATH):
            os.makedirs(self.PSYCHOPY_PATH)
            print(f"Created directory: {self.ARDUINO_PATH}")

        self.TEXT_HEIGHT_RATIO = float(os.environ.get("TEXT_HEIGHT_RATIO"))
        self.WRAP_RATIO = 0.8


exp_config = config()

# ============ Psychopy Experiment Init ==============

# 1. Collect Information #
exp_info = {"participant": "your_name", "session": "", "data_session": "", "time": datetime.now()}
dlg = gui.DlgFromDict(exp_info, title="Emotion Video Task")
if not dlg.OK:
    core.quit()

if BY_SESSION and not exp_info["data_session"]:
    ValueError("If in BY SESSION mode, please stipulate the data you would like to use with 'data session'")

print('exp_info["data_session"]', int(exp_info['data_session']))
# initiate window
win = visual.Window(
    # size=(1728, 1117),
    fullscr=FULLSCREEN,
    units="pix",
    color="black",
    waitBlanking=True
)

print(win.contentScaleFactor)

# after window is defined, extract window height
WIN_WIDTH = win.size[0]
WIN_HEIGHT = win.size[1]
print("win: ", win.size, WIN_WIDTH, WIN_HEIGHT)


# Let PsychoPy's ExperimentHandler manages data files
exp_name = "emotion_video_task"
this_exp = data.ExperimentHandler(
    name=exp_name,
    extraInfo=exp_info,
    dataFileName=os.path.join(exp_config.PSYCHOPY_PATH, f"psychopy_{exp_name}_{exp_info['session']}_{exp_info['participant']}_{exp_info['time']}")
)


# ============ Arduino Writer ==============

# connect to arduino
arduino_writer = None
if USE_ARDUINO:
    from arduino import ArduinoWriter
    ppgFileName = f"{exp_config.ARDUINO_PATH}/ppg_{exp_name}_{exp_info['session']}_{exp_info['participant']}_{exp_info['time']}.csv" 
    
    arduino_writer = ArduinoWriter(ppgFileName=ppgFileName, core=core, serial_port="/dev/cu.usbmodem2101")
    



from components import press_space_to_continue


# get videos
# 1. Prepare Videos #
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from video_shufflers import demo_video_shufflers, trial_video, video_shufflers, tiny_trial_video, get_washout_video
from video_shufflers import StimulationVideo


washout_video = get_washout_video()


if BY_SESSION:
    video_list = video_shufflers(
        input_path = "/Users/lily/Documents/myApps/Capstone_Saveme/5_stimulation_materials/all_videos_select_1.csv",
        active_tags =emotions_tags,
        multi_session = True,
        temp_json_dir_path = VIDEO_JSON_DIR,
        temp_json_path = "",
        num_sessions = None,
        session_num=int(exp_info['data_session']),
        shuffle = False
    )
else:
    video_list = video_shufflers(
        input_path = VIDEO_LIST,
        # num_sessions = 2,
        temp_json_path = exp_config.TEMP_VIDEO_PATH,
        active_tags = emotions_tags,
        shuffle=False
    )

print(video_list)

# start
press_space_to_continue(win, core, this_exp, event, exp_config, 
                        message= '''你将观看一系列视频切片。\n\n并在每个视屏之后描述你对视频的感受。\n\n请在观看过程中尽量沉浸在视屏中。\n\n实验中途，如有任何不适，\n请按键盘左上角的escape键退出。\n\n按“空格”进入模拟环节。''')

from utils import play_beep
from components import play_washout, play_video, collect_user_self_report
# play_beep() 

# press_space_to_continue(win, core, this_exp, event, exp_config, message="可以借机调整下眼镜\n接下来的黑白视频和有声视频过程中\n尽量不要动眼镜\n\n按空格继续......")
for session_idx, experiment_session_videos in video_list.items():

    for video_idx, video in enumerate(experiment_session_videos):

        press_space_to_continue(win, core, this_exp, event, exp_config, message="可以借机调整下眼镜\n接下来的黑白视频和有声视频过程中\n都尽量不要动眼镜\n\n同时按下超声按钮和空格继续......")
        if arduino_writer:
            arduino_writer.writerWrite([core.getTime(), "", "", "", "SPLIT_BY"])
        play_washout(win, core, this_exp, event, arduino_writer=arduino_writer, max_seconds=min(MAX_SECONDS,60))
        if  arduino_writer:
            arduino_writer.clearCache(core)

        press_space_to_continue(win, core, this_exp, event, exp_config, message="停下超声，按空格继续......")

        

        if video.background_content:
            bg_content = '\n'.join(video.background_content.split(' '))
            press_space_to_continue(win, core, this_exp, event, exp_config, message=f"接下来的影片的{video.background_type}是:\n\n{bg_content} \n\n 按空格继续......")

        if  arduino_writer:
            arduino_writer.clearCache(core)


        press_space_to_continue(win, core, this_exp, event, exp_config, message="同时按下超声按钮和空格继续......")
        play_video(win, core, this_exp, event, video, arduino_writer=arduino_writer, display_label = False, max_seconds=min(MAX_SECONDS, 1000000000))
        # if  arduino_writer:
        #     arduino_writer.clearCache(core)

        press_space_to_continue(win, core, this_exp, event, exp_config, message="停下超声，按空格继续......")

        collect_user_self_report(win, core, this_exp, event, session_idx, video, emotions_cn)
        if  arduino_writer:
            arduino_writer.clearCache(core)

        # press_space_to_continue(win, core, this_exp, event, exp_config, message="视频结束，按空格继续......")

        

    press_space_to_continue(win, core, this_exp, event, exp_config, message = f"第{session_idx+1}个session结束了，休息一会儿吧。\n\n中途可以摘下眼镜，但如果发现红灯不亮了请叫人。\n\n休息好了按'空格'继续。")


if USE_ARDUINO: arduino_writer.close()

if not BY_SESSION:
    press_space_to_continue(win, core, this_exp, event, exp_config, message = "试验结束，谢谢你的参与！！！！！！！")

win.close()
core.quit()