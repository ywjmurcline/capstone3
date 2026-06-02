import os
import sys
sys.path.append("/Users/lily/Documents/myApps/Capstone_Saveme/4_psychopy_data_collector_double_ppg")
from video_shufflers import get_washout_video
from utils import scale_video
from psychopy.visual.vlcmoviestim import VlcMovieStim
from psychopy.visual import TextStim, ImageStim, Rect, Circle
from dotenv import load_dotenv
from arduino import ArduinoWriter
import random
import os
load_dotenv()


def play_washout(win, core, exp, event, arduino_writer: list[ArduinoWriter] = None, max_seconds: int = 1000000000):
    '''
        play washout video for a maximum of {max_seconds} seconds
    '''

    videoStartSent = False
    videoEndSent = False

    event.clearEvents(eventType='keyboard')

    video = get_washout_video()

    movie = VlcMovieStim(
        win, 
        video.path, 
        size=scale_video(win, video.width, video.height), 
        loop=False, 
        noAudio=True)
    
    win.flip()
    movie.play()

    timer = core.Clock()

    while not movie.isFinished:
        if 'escape' in event.getKeys():
            if not videoEndSent:
                if arduino_writer[0]:
                    try:
                        arduino_writer[0].addTag(core, serTag = b'E', writerTag = "VIDEO_END_PC_WASHOUT")
                    except:
                        pass
                if arduino_writer[1]:
                    try:
                        arduino_writer[1].addTag(core, serTag = b'E', writerTag = "VIDEO_END_PC_WASHOUT")
                    except:
                        pass
                videoEndSent = True
            exp.abort()
            win.close()
            core.quit()
        
        if(movie.status == 1) and not videoStartSent:
            print("washout_started")
            if arduino_writer[0]:
                arduino_writer[0].addTag(core, serTag = b'S', writerTag = "VIDEO_START_PC_WASHOUT")
            if arduino_writer[1]:
                arduino_writer[1].addTag(core, serTag = b'S', writerTag = "VIDEO_START_PC_WASHOUT")
            videoStartSent = True

        if  arduino_writer[0]: arduino_writer[0].clearCache(core)
        if  arduino_writer[1]: arduino_writer[1].clearCache(core)
    
        movie.draw()
        win.flip()

        if movie.isFinished or timer.getTime() > max_seconds:
            if not videoEndSent:
                print("finished")
                if arduino_writer[0]: 
                    arduino_writer[0].addTag(core, serTag = b'E', writerTag = "VIDEO_END_PC_WASHOUT")
                if arduino_writer[1]: 
                    arduino_writer[1].addTag(core, serTag = b'E', writerTag = "VIDEO_END_PC_WASHOUT")
                videoEndSent = True
            break
        if  arduino_writer[0]: arduino_writer[0].clearCache(core)
        if  arduino_writer[1]: arduino_writer[1].clearCache(core)


    movie.stop()
    if  arduino_writer[0]: arduino_writer[0].clearCache(core)
    if  arduino_writer[1]: arduino_writer[1].clearCache(core)



def show_fixation(win, core, exp, event, duration=2.0):
    '''
        show fixation + for {duration} seconds
    '''

    event.clearEvents(eventType='keyboard')

    fixation = TextStim(
        win,
        text="+",
        height=80,
        color="white",
        font="Arial"
    )

    timer = core.Clock()
    while timer.getTime() < duration:
        keys = event.getKeys(keyList=["escape"])
        if "escape" in keys:
            exp.abort()
            win.close()
            core.quit()

        fixation.draw()
        win.flip()


def press_space_to_continue(win, core, exp, event, exp_config, message):
    event.clearEvents(eventType='keyboard')
    pause_msg = TextStim(
        win,
        text=message, 
        height=win.size[1] * exp_config.TEXT_HEIGHT_RATIO, 
        color="white", 
        wrapWidth=win.size[0] * exp_config.WRAP_RATIO,
        font="Hei"   # or another Chinese font
        )

    while True:
        pause_msg.draw()
        win.flip()

        keys = event.getKeys(keyList=["space", "escape"])
        if "space" in keys:
            event.clearEvents()
            return
        if "escape" in keys:
            exp.abort()
            win.close()
            core.quit()


def play_video(win, core, exp, event, video, arduino_writer: list[ArduinoWriter] = None, display_label = False, max_seconds: int = 1000000000):
    videoStartSent = False
    videoEndSent = False

    event.clearEvents(eventType='keyboard')

    movie = VlcMovieStim(
        win, 
        video.path, 
        size=scale_video(win, video.width, video.height), 
        loop=False
    )

    video_label = TextStim(
        win,
        text=video.id,
        height=80,
        pos=(-600,  500),
        color="white",
        font="Arial"
    )

    win.flip()
    exp.addData("video_start_time", core.getTime())
    movie.play()

    timer = core.Clock()

    while not movie.isFinished:
        keys = event.getKeys(keyList=["escape"])
        if "escape" in keys:
            if not videoEndSent:
                if arduino_writer[0]:
                    try:
                         arduino_writer[0].addTag(core, serTag = b'E', writerTag = "VIDEO_END_PC")
                    except:
                        pass
                if arduino_writer[1]:
                    try:
                         arduino_writer[1].addTag(core, serTag = b'E', writerTag = "VIDEO_END_PC")
                    except:
                        pass
            exp.abort()
            win.close()
            core.quit()
        
        if(movie.status == 1) and not videoStartSent:
            print("started")
            if  arduino_writer[0]:
                arduino_writer[0].addTag(core, serTag = b'S', writerTag = f"VIDEO_START_PC_{video.emo_tag}_{video.id}")
            if  arduino_writer[1]:
                arduino_writer[1].addTag(core, serTag = b'S', writerTag = f"VIDEO_START_PC_{video.emo_tag}_{video.id}")
            videoStartSent = True

        if  arduino_writer[0]: arduino_writer[0].clearCache(core)
        if  arduino_writer[1]: arduino_writer[1].clearCache(core)

        movie.draw()
        if display_label: video_label.draw()
        win.flip()

        if movie.isFinished or timer.getTime() > max_seconds:
            if not videoEndSent:
                print("finished")
                if arduino_writer[0]: 
                    arduino_writer[0].addTag(core, serTag = b'E', writerTag = "VIDEO_END_PC")
                if arduino_writer[1]: 
                    arduino_writer[1].addTag(core, serTag = b'E', writerTag = "VIDEO_END_PC")
                videoEndSent = True
            break
        if  arduino_writer[0]: arduino_writer[0].clearCache(core)
        if  arduino_writer[1]: arduino_writer[1].clearCache(core)

    movie.stop()
    exp.addData("video_end_time", core.getTime())
    if  arduino_writer[0]: arduino_writer[0].clearCache(core)
    if  arduino_writer[1]: arduino_writer[1].clearCache(core)


def choose_seven_point(win, core, exp, event, type="valence", top_message="", left_message="left", right_message="right", middle_message=""):
    event.clearEvents(eventType='keyboard')
    currentIndex = random.randint(0, 6)
    # image file names
    files = [
        'img1.png', 'img2.png', 'img3.png', 'img4.png',
        'img5.png', 'img6.png', 'img7.png'
    ]
    x_positions = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15] 

    images = [
        ImageStim(
            win, 
            image=f"./images/{type}/" + f, 
            pos=(x * win.size[0], -0.01 * win.size[0]), 
            size=(0.04 * win.size[0], 0.04 * win.size[0]))
        for f, x in zip(files, x_positions)
    ]

    highlights = [
        Rect(
            win,
            pos=(x * win.size[0], -0.01 * win.size[0]),
            width=0.045 * win.size[0],
            height=0.045 * win.size[0],
            lineColor='yellow',
            fillColor=None,
            opacity=0
        )
        for x in x_positions
    ]

    instruction = TextStim(
        win,
        text=top_message,
        pos=(0, 0.15 * win.size[1]),
        height=0.02 * win.size[1],
        wrapWidth=win.size[0] * 0.8,
        font="Hei"
    )

    annotation_left = TextStim(
        win,
        text=left_message,
        pos=(-0.15 * win.size[0], 0.025* win.size[0]),
        height= 0.02 * win.size[1],
        wrapWidth=win.size[0] * 0.2,
        font="Hei"
    )

    annotation_middle = TextStim(
        win,
        text=middle_message,
        pos=(0,  0.025* win.size[0]),
        height= 0.02 * win.size[1],
        wrapWidth=win.size[0] * 0.2,
        font="Hei"
    )


    annotation_right = TextStim(
        win,
        text=right_message,
        pos=(0.15 * win.size[0], 0.025* win.size[0]),
        height= 0.02 * win.size[1],
        wrapWidth=win.size[0] * 0.2,
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

def ask_seen_before_lr(win, core, exp, event):
    event.clearEvents(eventType='keyboard')
    options = ["没看过", "看过"]
    currentIndex = random.randint(0, 1)
    clock = core.Clock()

    instruction = TextStim(
        win,
        text="在本次实验之前，你是否看过该视频？\n\n用左右键选择，按空格确认",
        pos=(0, 0.15 * win.size[1]),
        height=0.02 * win.size[1],
        wrapWidth=win.size[0] * 0.8,
        font="Hei"
    )

    option_stims = []
    highlight_boxes = []

    x_positions = [-0.1, 0.1]

    for opt, x in zip(options, x_positions):
        stim = TextStim(
            win,
            text=opt,
            pos=(x * win.size[0], -0.05 *  win.size[1]),
            height=0.03 * win.size[1],
            font="Hei"
        )
        option_stims.append(stim)

        box = Rect(
            win,
            pos=(x * win.size[0], -0.05 * win.size[1]),
            width=0.15 * win.size[0],
            height=0.1 * win.size[1],
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

def ask_emotional_effect_lr(win, core, exp, event):
    event.clearEvents(eventType='keyboard')
    options = [
        "仍然有明显情绪",
        "有情绪\n但比第一次弱",
        "很平静"
    ]

    currentIndex = random.randint(0, 2)
    clock = core.Clock()

    instruction = TextStim(
        win,
        text="如果你之前看过该视频，它现在对你的情绪影响如何？\n\n用左右键选择，按空格确认\n(如果上一题写的是没有看过，随便选就好)",
        pos=(0, 0.15 * win.size[1]),
        height=0.02 * win.size[1],
        wrapWidth=win.size[0] * 0.8,
        font="Hei"
    )

    option_stims = []
    highlight_boxes = []

    x_positions = [-0.15, 0.0, 0.15]

    for opt, x in zip(options, x_positions):
        stim = TextStim(
            win,
            text=opt,
            pos=(x * win.size[0], -0.03 * win.size[0]),
            height=0.03 * win.size[1],
            wrapWidth=0.15 * win.size[0],
            font="Hei"
        )
        option_stims.append(stim)

        box = Rect(
            win,
            pos=(x * win.size[0], -0.05 * win.size[1]),
            width=0.17 * win.size[0],
            height=0.12 * win.size[1],
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

def choose_emotions(win, core, exp, event, emotions_cn):
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
    left_margin_x = -0.20 * win.size[0]          # emotion label column
    start_x = -0.1 * win.size[0]                # first rating column (0)
    col_spacing = 0.04 * win.size[0]             # horizontal distance between buttons
    header_y = 0.085 * win.size[1]              # header row y
    first_row_y = 0.04 * win.size[1]           # first emotion row y
    row_spacing = 0.035 * win.size[1]           # vertical distance between rows

    radio_radius = 10                          # outer circle radius in px
    inner_radius = 6                           # filled center radius in px

    # default ratings: all set to 0
    ratings = {emotion: 0 for emotion in emotions_cn.keys()}

    # ---------- instruction ----------
    instruction = TextStim(
        win,
        text="请选择哪种情绪更符合你观看视频时的感受\n\n用鼠标点击选择，按“空格”确认",
        pos=(0, 0.18 * win.size[1]),
        height=0.03 * win.size[1],
        wrapWidth=0.4 * win.size[0],
        color="white",
        font="Hei"
    )

    # ---------- header ----------
    header_left = TextStim(
        win,
        text="情绪",
        pos=(left_margin_x, header_y),
        height=0.9 * 0.03 * win.size[1],
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

        stim = TextStim(
            win,
            text=label_text,
            pos=(x, header_y),
            height=0.7 * 0.03 * win.size[1],
            alignText='center',
            anchorHoriz='center',
            color="white",
            font="Hei"
        )
        header_scale.append(stim)

    # ---------- build table objects ----------
    emotion_labels = []
    radio_buttons = []   # each item: list of dicts for one row

    for row_idx, emotion in enumerate(emotions_cn):
        y = first_row_y - row_idx * row_spacing

        # emotion label
        emotion_label = TextStim(
            win,
            text=emotions_cn[emotion],
            pos=(left_margin_x, y),
            height=0.9 * 0.03 * win.size[1],
            anchorHoriz='center',
            color="white",
            font="Hei"
        )
        emotion_labels.append(emotion_label)

        row_buttons = []
        for col_idx, value in enumerate(scale_values):
            x = start_x + col_idx * col_spacing

            outer_circle = Circle(
                win,
                radius=radio_radius,
                pos=(x, y),
                lineColor="white",
                fillColor=None,
                lineWidth=2
            )

            inner_circle = Circle(
                win,
                radius=inner_radius,
                pos=(x, y),
                lineColor=None,
                fillColor="white",
                opacity=1 if value == 0 else 0   # default selected = 0
            )

            click_area = Rect(
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
            for row_idx, emotion in enumerate(emotions_cn.keys()):
                for button in radio_buttons[row_idx]:
                    if button["hitbox"].contains(mouse):
                        ratings[emotion] = button["value"]
                        break

        mouse_was_down = left_down

        # ----- update selected state -----
        for row_idx, emotion in enumerate(emotions_cn.keys()):
            selected_value = ratings[emotion]
            for button in radio_buttons[row_idx]:
                button["inner"].opacity = 1 if button["value"] == selected_value else 0

        # ----- draw -----
        instruction.draw()
        header_left.draw()
        for stim in header_scale:
            stim.draw()

        for row_idx in range(len(emotions_cn.keys())):
            emotion_labels[row_idx].draw()
            for button in radio_buttons[row_idx]:
                button["outer"].draw()
                button["inner"].draw()

    #     footer.draw()
        win.flip()

    print("Emotion ratings:", ratings)
    return ratings



def collect_user_self_report(win, core, exp, event, session_idx, video, emotions_cn, trial = False):
    valence, v_rt = choose_seven_point(
        win, core, exp, event,
        type="valence", 
        top_message='观看视频时，你的心情是正向的还是负向的？可以参考画面里的小人\n\n请用键盘上的左右键选择，按“空格”确认',
        left_message="非常负面", 
        right_message="非常正面", 
        middle_message="中性")
    
    arousal, a_rt = choose_seven_point(
        win, core, exp, event,
        type="arousal", 
        top_message='观看视频时，你的情绪强烈程度是？可以参考画面里的小人\n\n请用键盘上的左右键选择，按“空格”确认',
        left_message="非常平淡", 
        right_message="非常激烈", 
        middle_message="")
    
    ratings = choose_emotions( win, core, exp, event, emotions_cn)

    familiarity, f_rt = ask_seen_before_lr( win, core, exp, event)

    habituation, h_rt = ask_emotional_effect_lr( win, core, exp, event)

    if not trial:
        exp.addData("session_index", session_idx)
        # exp.addData("trial_index", video_idx)
        exp.addData("video_id", video.id)
        exp.addData("video_file", video.path)

        exp.addData("valence", valence)
        exp.addData("valence_rt", v_rt)

        exp.addData("arousal", arousal)
        exp.addData("arousal_rt", a_rt)

        for emo, score in ratings.items():
            exp.addData(f"emotion_{emo}", score)

        # exp.addData("dominance", dominance)
        exp.addData("familiar", familiarity)
        exp.addData("familiar_rt", f_rt)

        exp.addData("habituation", habituation)
        exp.addData("habituation_rt", h_rt)

        exp.nextEntry()


