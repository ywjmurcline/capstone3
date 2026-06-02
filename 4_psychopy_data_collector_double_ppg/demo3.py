import random
from typing import List
from psychopy import visual, core, event, gui, data, sound
from datetime import datetime
from dotenv import load_dotenv
import os
import asyncio
from arduino import ArduinoWriter
import json

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from video_shufflers import demo_video_shufflers, trial_video, video_shufflers_one_per_session, tiny_trial_video, get_washout_video
from video_shufflers import StimulationVideo
load_dotenv()


emotions_cn = {
    "Amusing": '好笑',
    "Anger": '生气',
    "Disgust": '恶心',
    "Fear": '恐惧',
    "Sad": '悲伤',
    "Happy": '快乐',
    "Neutral": '中性',
}

class Config:
    def __init__(self, exp_name, test_video_play, exp_data_dir, fullscreen, video_csv, emo_tags_activation, use_arduino):
        self.exp_name = exp_name
        self.test_video_play = test_video_play
        self.exp_data_dir = exp_data_dir
        self.fullscreen = fullscreen
        self.video_csv = video_csv
        self.emo_tags_activation = emo_tags_activation
        self.use_arduino = use_arduino
        self.TEXT_HEIGHT_RATIO = 0.03
        self.WRAP_RATIO = 0.8
        return
    
class VideoManager:
    def __init__(self, path, video_list_csv, emo_tags):
        self.path = path
        self.video_list_csv = video_list_csv
        self.emo_tags = emo_tags

    def get_current_video_index(self):
        with open(self.current_video_txt, "r") as f:
            content = f.read()
            return content
        
    def move_to_next_video(self):
        with open(self.current_video_txt, "r") as f:
            content = f.read()
            current_idx = int(content.strip("\n"))
        with open(self.current_video_txt, "w") as f:
            f.write(str(current_idx+1))

    def set_current_video(self, current_idx):
        with open(self.current_video_txt, "w") as f:
            f.write(str(current_idx))

    def current_video_checker(self, video_id):
        with open(self.video_checker, "r", encoding="utf-8") as f:
            checker = json.load(f)
        return bool(checker[video_id])
    
    def get_session_video(self, session):
        return self.video_list[int(session)]

    def get_current_video(self):
        with open(self.current_video_txt, "r") as f:
            content = f.read()
            print(content)
            return self.video_list[int(content.strip("\n"))]

    def update_checker(self, entry, value):
        with open(self.video_checker, "r", encoding="utf-8") as f:
            checker = json.load(f)
        checker[entry] = bool(value)
        print(f"Updated video checker: video {entry} to {value}")
        with open(self.video_checker, "w", encoding="utf-8") as f:
            json.dump(checker, f, indent=4, ensure_ascii=False)


    @classmethod
    async def create(cls, path, video_list_csv, emo_tags):
        self = cls(path, video_list_csv, emo_tags)
        await self._async_init()
        return self
    
    async def _async_init(self):
        assert os.path.exists(self.path)
        self.current_video_txt = os.path.join(self.path, "temp.txt")
        if not os.path.exists(self.current_video_txt):
            with open(self.current_video_txt, "w") as f:
                f.write("0")

        self.video_list_json = os.path.join(self.path, "temp.json")

        # print(self.video_list_csv)
        # create_video_json
        self.video_list = video_shufflers_one_per_session(
            input_path = self.video_list_csv,
            active_tags = self.emo_tags,
            temp_json_path = self.video_list_json,
            shuffle = True,
        ) 

        self.total_videos = len(self.video_list)

        self.video_checker= os.path.join(self.path, "checker.json")
        if not os.path.exists(self.video_checker):
            checker = {}
            for _, video_items in self.video_list.items():
                if not isinstance(video_items, list):
                    continue

                for video in video_items:
                    video_name = video.id

                    if video_name is not None:
                        checker[str(video_name)] = False

            with open(self.video_checker, "w", encoding="utf-8") as f:
                json.dump(checker, f, indent=4, ensure_ascii=False)

class ExpManager:
    def __init__(self, config: Config):
        self.config = config
    
    @classmethod
    async def create(cls, config):
        self = cls(config)
        await self._async_init()
        return self
    
    def if_not_exist_create_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
            print(f"Created directory: {dir}")
        else:
            print(f"Directory {dir} already exists")

    def if_not_exist_create_file(self, file):
        if not os.path.exists(file):
            open(file, "w").close()

    def save_to_temp(self, text):
        with open(self.exp_temp_txt, "w") as f:
            f.write(text)

    def read_from_temp(self):
        with open(self.exp_temp_txt, "r") as f:
            content = f.read()
            return content.strip("\n")

    async def _async_init(self):
        self.fullscreen = bool(self.config.fullscreen)
        self.exp_data_dir = self.config.exp_data_dir

        # make sure exp_data_dir exist, this will be used to store all data for this experiment
        self.if_not_exist_create_dir(self.exp_data_dir)


        # biodata directory
        self.exp_bio_data_dir = os.path.join(self.exp_data_dir, "bio_data")
        self.if_not_exist_create_dir(self.exp_bio_data_dir)

        self.ppg_dir = os.path.join(self.exp_bio_data_dir, "ppg")
        self.if_not_exist_create_dir(self.ppg_dir)

        self.psychopy_dir = os.path.join(self.exp_bio_data_dir, "psychopy")
        self.if_not_exist_create_dir(self.psychopy_dir)


        # video manager
        self.video_manager_dir = os.path.join(self.exp_data_dir, "video")
        self.if_not_exist_create_dir(self.video_manager_dir)

        self.video_manager = await VideoManager.create(
            self.video_manager_dir,
            video_list_csv = self.config.video_csv,
            emo_tags = self.config.emo_tags_activation
        )
        
        # temporary data
        self.exp_temp_txt = os.path.join(self.exp_data_dir, "temp.txt")
        self.if_not_exist_create_file(self.exp_temp_txt)


        self.arduino = self.config.use_arduino

        self.exp_name = self.config.exp_name

        self.max_seconds = 5 if self.config.test_video_play else 1000000000



        # arduino
        # self.finger_ppgFileName = f"{exp_config.ARDUINO_PATH}/ppg_{exp_name}_{exp_info['session']}_{exp_info['participant']}_{exp_info['time']}.csv" 
    
        # arduino_writer = ArduinoWriter(ppgFileName=ppgFileName, core=core, serial_port="/dev/cu.usbmodem2101")
    

async def main():
    exp_config = Config(
        exp_name = "emotion_video_task",
        test_video_play =False,
        fullscreen=True,
        exp_data_dir="/Users/lily/Documents/myApps/Capstone_Saveme/1_data_archive/534", # TODO, 记得改成存数据的目录
        video_csv="/Users/lily/Documents/myApps/Capstone_Saveme/5_stimulation_materials/all_videos_select_1.csv", # TODO, 记得改成csv的地址
        emo_tags_activation = {
            "Amusing": True,
            "Anger": True,
            "Disgust":  True,
            "Exciting": True,
            "Fear":  True,
            "Sad": True,
            "Happy":  True,
            "Neutral":  True,
        },
        use_arduino = {
            "glass": {"use": True, "serial_port": "/dev/cu.usbmodem1101", "port": 74880},
            "finger": {"use": True, "serial_port": "/dev/cu.usbmodem2101", "port": 115200}
        }
    )

    exp_manager = await ExpManager.create(exp_config)


    your_name = exp_manager.read_from_temp()
    if not your_name:
        your_name = "enter your name"

    video_id = exp_manager.video_manager.get_current_video_index()

    exp_info = {"participant": your_name, "video_id": video_id, "total_videos": exp_manager.video_manager.total_videos, "time": datetime.now()}
    dlg = gui.DlgFromDict(
        exp_info, 
        title="Emotion Video Task",
        fixed=["total_videos", "time"]
        )
    if not dlg.OK:
        core.quit()

    exp_manager.save_to_temp(exp_info["participant"])
    exp_manager.video_manager.set_current_video(exp_info["video_id"])
    
    current_video_recorded = exp_manager.video_manager.current_video_checker(
        exp_manager.video_manager.get_session_video(exp_info["video_id"])[0].id
    )
    
    # safe guard, 不重复录制
    if current_video_recorded:
        i = input("You have already watched this video, press Y to override previous record: ")
        if not (i == "Y" or i == "y"):
            core.quit()

    # initiate window
    win = visual.Window(
        # size=(1728, 1117),
        fullscr=exp_manager.fullscreen,
        units="pix",
        color="black",
        waitBlanking=True
    )
    WIN_WIDTH = win.size[0]
    WIN_HEIGHT = win.size[1]
    print("win: ", win.size, WIN_WIDTH, WIN_HEIGHT)


    this_exp = data.ExperimentHandler(
        name=exp_manager.exp_name,
        extraInfo=exp_info,
        dataFileName=os.path.join(exp_manager.psychopy_dir, f"psychopy_{exp_manager.exp_name}_{exp_info['video_id']}_{exp_info['participant']}_{exp_info['time']}")
    )


    # ============ Arduino Writer ==============

    arduino_writer_glass = None
    arduino_writer_finger = None

    # connect to arduino
    if exp_manager.arduino["glass"]:
        from arduino import ArduinoWriter
        ppgFileNameFinger = f"{exp_manager.ppg_dir}/ppg_glass_{exp_manager.exp_name}_{exp_info['video_id']}_{exp_info['participant']}_{exp_info['time']}.csv" 
        arduino_writer_glass = ArduinoWriter(ppgFileName=ppgFileNameFinger, core=core, serial_port=exp_manager.arduino["glass"]["serial_port"], port=exp_manager.arduino["glass"]["port"], name="glass")
        
    if exp_manager.arduino["finger"]:
        from arduino import ArduinoWriter
        ppgFileNameGlass = f"{exp_manager.ppg_dir}/ppg_finger_{exp_manager.exp_name}_{exp_info['video_id']}_{exp_info['participant']}_{exp_info['time']}.csv" 
        arduino_writer_finger = ArduinoWriter(ppgFileName=ppgFileNameGlass, core=core, serial_port=exp_manager.arduino["finger"]["serial_port"], port=exp_manager.arduino["finger"]["port"], name="finger")
        
    arduino_writers = [arduino_writer_glass, arduino_writer_finger]
    
    from components import press_space_to_continue
    from components import play_washout, play_video, collect_user_self_report

    video = exp_manager.video_manager.get_session_video(exp_info["video_id"])[0]


    press_space_to_continue(win, core, this_exp, event, exp_config, message="可以借机调整下眼镜\n接下来的黑白视频和有声视频过程中\n都尽量不要动眼镜\n\n同时按下超声按钮和空格继续......")
    if arduino_writer_finger:
        arduino_writer_finger.writerWrite([core.getTime(), "", "", "", "SPLIT_BY"])
    if arduino_writer_glass:
        arduino_writer_glass.writerWrite([core.getTime(), "", "", "", "SPLIT_BY"])
    
    play_washout(win, core, this_exp, event, arduino_writer=arduino_writers, max_seconds=min(exp_manager.max_seconds, 60))
    if arduino_writer_finger: arduino_writer_finger.clearCache(core)
    if arduino_writer_glass: arduino_writer_glass.clearCache(core)

    press_space_to_continue(win, core, this_exp, event, exp_config, message="停下超声，按空格继续......")

    

    if video.background_content:
        bg_content = '\n'.join(video.background_content.split(' '))
        press_space_to_continue(win, core, this_exp, event, exp_config, message=f"接下来的影片的{video.background_type}是:\n\n{bg_content} \n\n 按空格继续......")

    if arduino_writer_finger: arduino_writer_finger.clearCache(core)
    if arduino_writer_glass: arduino_writer_glass.clearCache(core)


    press_space_to_continue(win, core, this_exp, event, exp_config, message="同时按下超声按钮和空格继续......")
    play_video(win, core, this_exp, event, video, arduino_writer=arduino_writers, display_label = False, max_seconds=min(exp_manager.max_seconds, 1000000000))
    # if  arduino_writer:
    #     arduino_writer.clearCache(core)

    press_space_to_continue(win, core, this_exp, event, exp_config, message="停下超声，按空格继续......")

    collect_user_self_report(win, core, this_exp, event, 0, video, emotions_cn)
    if arduino_writer_finger: arduino_writer_finger.clearCache(core)
    if arduino_writer_glass: arduino_writer_glass.clearCache(core)

    if arduino_writer_finger: arduino_writer_finger.close()
    if arduino_writer_glass: arduino_writer_glass.close()

    exp_manager.video_manager.update_checker(video.id, True)
    exp_manager.video_manager.move_to_next_video()

    win.close()
    core.quit()

if __name__ == "__main__":
    asyncio.run(main())