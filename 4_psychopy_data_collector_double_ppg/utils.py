def scale_video(win, video_width, video_height):
    # Calculate aspect ratios
    screen_ratio = win.size[0] / win.size[1]
    video_ratio = video_width / video_height
    
    # If video is wider relative to screen (video_ratio > screen_ratio)
    # then scale to match screen width
    if video_ratio >= screen_ratio:
        scaled_width = win.size[0]
        scaled_height = int(scaled_width / video_ratio)
    # Otherwise scale to match screen height
    else:
        scaled_height = win.size[1]
        scaled_width = int(scaled_height * video_ratio)

    print(video_width, video_height)
    print(scaled_width, scaled_height)

    return (scaled_width / 2, scaled_height / 2)



import subprocess
def play_beep():
    """
    使用 macOS 系统声音播放提示音
    不依赖 PsychoPy sound / PTB
    """
    subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"])



