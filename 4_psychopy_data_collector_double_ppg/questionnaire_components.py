import random
from psychopy.visual import TextStim, Rect, ImageStim

def choose_seven_point(win, core, exp, event, win_size, type="valence", top_message="", left_message="left", right_message="right", middle_message=""):
    event.clearEvents(eventType='keyboard')
    currentIndex = random.randint(0, 6)
    # image file names
    files = [
        'img1.png', 'img2.png', 'img3.png', 'img4.png',
        'img5.png', 'img6.png', 'img7.png'
    ]
    x_positions = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15] 

    images = [
        ImageStim(win, image=f"images/{type}/" + f, pos=(x * win_size.width, -0.01 * win_size.width), size=(0.04 * win_size.width, 0.04 * win_size.width))
        for f, x in zip(files, x_positions)
    ]

    highlights = [
        Rect(
            win,
            pos=(x * win_size.width, -0.01 *  win_size.width),
            width=0.045 *  win_size.width,
            height=0.045 *  win_size.width,
            lineColor='yellow',
            fillColor=None,
            opacity=0
        )
        for x in x_positions
    ]

    instruction = TextStim(
        win,
        text=top_message,
        pos=(0, 0.15 * WIN_HEIGHT),
        height=EXPERIMENT_INSTRUCTION_TEXT_HEIGHT,
        wrapWidth= win_size.width * 0.8,
        font="Hei"
    )

    annotation_left = TextStim(
        win,
        text=left_message,
        pos=(-0.15 *  win_size.width, 0.025*  win_size.width),
        height= 0.8 * EXPERIMENT_INSTRUCTION_TEXT_HEIGHT,
        wrapWidth= win_size.width * 0.2,
        font="Hei"
    )

    annotation_middle = visual.TextStim(
        win,
        text=middle_message,
        pos=(0,  0.025*  win_size.width),
        height= 0.8 * EXPERIMENT_INSTRUCTION_TEXT_HEIGHT,
        wrapWidth= win_size.width * 0.2,
        font="Hei"
    )

    annotation_right = visual.TextStim(
        win,
        text=right_message,
        pos=(0.15 *  win_size.width, 0.025*  win_size.width),
        height= 0.8 * EXPERIMENT_INSTRUCTION_TEXT_HEIGHT,
        wrapWidth= win_size.width * 0.2,
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
        wrapWidth= win_size.width * 0.8,
        font="Hei"
    )

    option_stims = []
    highlight_boxes = []

    x_positions = [-0.1, 0.1]

    for opt, x in zip(options, x_positions):
        stim = visual.TextStim(
            win,
            text=opt,
            pos=(x *  win_size.width, -0.05 * WIN_HEIGHT),
            height=EXPERIMENT_INSTRUCTION_TEXT_HEIGHT,
            font="Hei"
        )
        option_stims.append(stim)

        box = visual.Rect(
            win,
            pos=(x *  win_size.width, -0.05 * WIN_HEIGHT),
            width=0.15 *  win_size.width,
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
        wrapWidth= win_size.width * 0.8,
        font="Hei"
    )

    option_stims = []
    highlight_boxes = []

    x_positions = [-0.15, 0.0, 0.15]

    for opt, x in zip(options, x_positions):
        stim = visual.TextStim(
            win,
            text=opt,
            pos=(x *  win_size.width, -0.05 * WIN_HEIGHT),
            height=0.8 * EXPERIMENT_INSTRUCTION_TEXT_HEIGHT,
            wrapWidth=0.15 *  win_size.width,
            font="Hei"
        )
        option_stims.append(stim)

        box = visual.Rect(
            win,
            pos=(x *  win_size.width, -0.05 * WIN_HEIGHT),
            width=0.17 *  win_size.width,
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
    left_margin_x = -0.20 *  win_size.width          # emotion label column
    start_x = -0.1 *  win_size.width                # first rating column (0)
    col_spacing = 0.04 *  win_size.width             # horizontal distance between buttons
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
        wrapWidth=0.4 *  win_size.width,
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
