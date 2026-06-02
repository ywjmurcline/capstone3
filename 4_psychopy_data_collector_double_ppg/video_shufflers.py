# Return a dictionary file
# {
#   "session 1": [ videoObject, ... , videoObject ], 
#   "session 2": [ videoObject, ... , videoObject ], 
#   ...
#   "session m": [ videoObject, ... , videoObject ], 
# }

class StimulationVideo:
    id = None
    video_name = ""
    source_dataset = ""
    emo_tag = ""
    path = ""
    width = 0
    height = 0

    def __init__(self, local_path, width, height, id = None, video_name = "", source_dataset = "", emo_tag = "", emo_tag_cn = "", background_type= "", background_content = ""):
        self.id = id
        self.video_name = video_name
        self.source_dataset = source_dataset
        self.emo_tag = emo_tag
        self.emo_tag_cn = emo_tag_cn
        self.path = local_path
        self.width = width 
        self.height = height
        self.background_type = background_type
        self.background_content = background_content

import csv
import random
from collections import defaultdict
from typing import Dict, List, Any
import json


def csv_reader(csv_path: str) -> List[dict]:
    """
    Read the CSV and return a list of video dicts.
    The header row is not included because csv.DictReader consumes it as field names.

    Expected CSV columns:
    id, video_name, tag, absolute_path, frame_width, frame_height, url, source_dataset
    """
    videos = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_columns = {
            "id",
            "video_name",
            "tag",
            "tag_cn",
            "absolute_path",
            "frame_width",
            "frame_height",
            "duration_seconds",
            "background_type",
            "background_content",
            "source_dataset",
        }

        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

        for row in reader:
            videos.append({
                "id": row["id"],
                "video_name": row["video_name"],
                "tag": row["tag"],
                "tag_cn": row["tag_cn"],
                "absolute_path": row["absolute_path"],
                "frame_width": int(row["frame_width"]),
                "frame_height": int(row["frame_height"]),
                "duration": float(row["duration_seconds"]),
                "background_type": row["background_type"],
                "background_content": row["background_content"],
                "url": row["url"],
                "source_dataset": row["source_dataset"],
            })

    return videos

def _shuffle_session_with_boundary_constraint(
    session_videos: List[dict],
    previous_last_tag: str = None,
    max_attempts: int = 1000,
) -> List[dict]:
    """
    Shuffle one session so that:
    - if previous_last_tag is given, the first video in this session must not share that tag
    """
    if not session_videos:
        return []

    # Fast path
    if previous_last_tag is None:
        shuffled = session_videos[:]
        random.shuffle(shuffled)
        return shuffled

    # Try random shuffles first
    for _ in range(max_attempts):
        shuffled = session_videos[:]
        random.shuffle(shuffled)
        if shuffled[0]["tag"] != previous_last_tag:
            return shuffled

    # Deterministic fallback
    for i, video in enumerate(session_videos):
        if video["tag"] != previous_last_tag:
            remaining = session_videos[:i] + session_videos[i + 1:]
            random.shuffle(remaining)
            return [video] + remaining

    raise ValueError(
        "Could not satisfy boundary constraint: all videos in this session "
        "have the same tag as the previous session's last video."
    )

from pathlib import Path
def write_json(data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        session_to_videos = json.load(f)

    print(json.dumps(session_to_videos, indent=4, ensure_ascii=False))

    result = {}
    for session_num, session_videos in session_to_videos.items():
        session_num = int(session_num)
        result[session_num] = [
            StimulationVideo(
                id=video["id"],
                video_name=video["video_name"],
                source_dataset=video["source_dataset"],
                width=video["frame_width"],
                height=video["frame_height"],
                emo_tag=video["tag"],
                emo_tag_cn=video["tag_cn"],
                local_path=video["absolute_path"],
                background_type = video["background_type"],
                background_content = video["background_content"],
            )
            for video in session_videos
        ]

    return result

import os
# def video_shufflers(input_path: str, num_sessions, temp_json_path) -> Dict[Any, List["StimulationVideo"]]:
#     """
#     Steps:
#     1. Read videos from CSV
#     2. Group by tag
#     3. Assert each tag has exactly len(session_nums) videos
#     4. For each tag, randomly assign one video to each session
#     5. Shuffle videos inside each session, while ensuring that:
#        last tag of previous session != first tag of current session

#     Returns:
#         {
#             session_num: [StimulationVideo(...), StimulationVideo(...), ...],
#             ...
#         }
#     """

#     # If input is JSON → load and skip processing
#     if os.path.isfile(temp_json_path) and temp_json_path.suffix.lower() == ".json":
#         return load_json(temp_json_path)
    

#     videos = csv_reader(input_path)

#     # 1) Group by tag
#     videos_by_tag = defaultdict(list)
#     for video in videos:
#         videos_by_tag[video["tag"]].append(video)

#     if not videos_by_tag:
#         raise ValueError("No videos found in CSV.")

#     # 2) Assert each tag has same number of videos as number of sessions
#     for tag, tag_videos in videos_by_tag.items():
#         assert len(tag_videos) == num_sessions, (
#             f"Tag '{tag}' has {len(tag_videos)} videos, "
#             f"but expected {num_sessions} (one for each session)."
#         )

#     # 3) For each tag, randomly decide which video goes to which session
#     session_to_videos = {session_num: [] for session_num in range(num_sessions)}

#     for tag, tag_videos in videos_by_tag.items():
#         shuffled_tag_videos = tag_videos[:]
#         random.shuffle(shuffled_tag_videos)

#         for session_num in range(num_sessions):
#             session_to_videos[session_num].append(shuffled_tag_videos[session_num])

#     # print(json.dumps(session_to_videos, indent=4))
    


#     # 4) For each session, shuffle videos with boundary constraint
#     previous_last_tag = None
#     for session_num in range(num_sessions):
#         shuffled_session = _shuffle_session_with_boundary_constraint(
#             session_to_videos[session_num],
#             previous_last_tag=previous_last_tag,
#         )
#         session_to_videos[session_num] = shuffled_session
#         previous_last_tag = shuffled_session[-1]["tag"]

#     print(json.dumps(session_to_videos, indent=4))
#     write_json(session_to_videos, temp_json_path)

#     # 5) Convert dicts to StimulationVideo objects
#     result = {}
#     for session_num, session_videos in session_to_videos.items():
#         result[session_num] = [
#             StimulationVideo(
#                 id=video["id"],
#                 video_name=video["video_name"],
#                 source_dataset=video["source_dataset"],
#                 emo_tag=video["tag"],
#                 local_path=video["absolute_path"],
#                 width=video["frame_width"],
#                 height=video["frame_height"],
#             )
#             for video in session_videos
#         ]

#     return result

# video_shufflers("test_data.csv", 3)

import heapq


def video_shufflers(
    input_path: str,
    active_tags,
    multi_session = False,
    temp_json_dir_path = "",
    temp_json_path = "",
    num_sessions: int = None,
    shuffle = True,
    session_num = None
) -> Dict[Any, List["StimulationVideo"]]:
    """
    Steps:
    1. Read videos from CSV
    2. Group by tag
    3. Shuffle videos within each tag
    4. Distribute videos into sessions using a min-heap so session sizes stay
       roughly balanced, even when:
         - number of sessions != number of videos per tag
         - different tags have different numbers of videos
    5. Shuffle videos inside each session while ensuring that:
       last tag of previous session != first tag of current session
    6. Save JSON cache
    7. Convert dicts to StimulationVideo objects

    Returns:
        {
            session_num: [StimulationVideo(...), ...],
            ...
        }
    """

    if not shuffle and num_sessions is not None:
        raise ValueError(f"If not shuffle, the session number is the tag number")
    
    if num_sessions is not None and num_sessions <= 0:
        raise ValueError(f"num_sessions must be > 0, got {num_sessions}")

    temp_json_path = Path(temp_json_path)


    if session_num is not None and temp_json_dir_path is not None:
        print("session_num and temp_json_dir_path")
        json_path = os.path.join(temp_json_dir_path, f"{str(session_num)}.json")
        print(json_path)
        if os.path.isfile(json_path):
            return load_json(json_path)
        else:
            raise ValueError("No session found.")

    # If input is JSON cache -> load and skip processing
    if temp_json_path.is_file() and temp_json_path.suffix.lower() == ".json":
        return load_json(temp_json_path)
    


    videos = csv_reader(input_path)

    # 1) Group by tag
    videos_by_tag = defaultdict(list)
    for video in videos:
        tag = video.get("tag")
        if tag is None:
            raise ValueError(f"Video missing 'tag': {video}")
        if active_tags[tag]:
            videos_by_tag[tag].append(video)

    if not videos_by_tag:
        raise ValueError("No videos found in CSV.")

    # 2) Shuffle within each tag
    for tag_videos in videos_by_tag.values():
        random.shuffle(tag_videos)


    if not shuffle:
        session_to_videos: Dict[int, List[dict]] = {
            
        }

        for index, tag_videos in enumerate(videos_by_tag.values()):
            session_to_videos[index] = tag_videos


    else:
        # 3) Create session buckets
        session_to_videos: Dict[int, List[dict]] = {
            session_num: [] for session_num in range(num_sessions)
        }

        # 4) Min-heap of (current_session_size, random_tiebreaker, session_num)
        #
        # The random tiebreaker prevents deterministic bias when multiple sessions
        # have the same size.
        heap: List[tuple[int, float, int]] = [
            (0, random.random(), session_num) for session_num in range(num_sessions)
        ]
        heapq.heapify(heap)

        # Process tags in random order so one tag does not always dominate early
        tag_items = list(videos_by_tag.items())
        random.shuffle(tag_items)

        for tag, tag_videos in tag_items:
            for video in tag_videos:
                session_num = _pick_session_from_heap(
                    heap=heap,
                    session_to_videos=session_to_videos,
                    current_tag=tag,
                )
                session_to_videos[session_num].append(video)

        # 5) Shuffle each session with boundary constraint across sessions
        previous_last_tag = None
        for session_num in range(num_sessions):
            session_videos = session_to_videos[session_num]

            if not session_videos:
                continue

            shuffled_session = _shuffle_session_with_boundary_constraint(
                session_videos,
                previous_last_tag=previous_last_tag,
            )
            session_to_videos[session_num] = shuffled_session

            if shuffled_session:
                previous_last_tag = shuffled_session[-1]["tag"]

    # print(json.dumps(session_to_videos, indent=4))
    if multi_session:
        for session_key, videos in session_to_videos.items():
            session_json_path = os.path.join(temp_json_dir_path,f"{session_key}.json")
            write_json({session_key: videos}, session_json_path)
    else:
        write_json(session_to_videos, temp_json_path)


    # 6) Convert dicts to StimulationVideo objects
    result: Dict[int, List["StimulationVideo"]] = {}
    for session_num, session_videos in session_to_videos.items():
        result[session_num] = [
            StimulationVideo(
                id=video["id"],
                video_name=video["video_name"],
                source_dataset=video["source_dataset"],
                emo_tag=video["tag"],
                local_path=video["absolute_path"],
                width=video["frame_width"],
                height=video["frame_height"],
            )
            for video in session_videos
        ]

    return result


def _pick_session_from_heap(
    heap: List[tuple[int, float, int]],
    session_to_videos: Dict[int, List[dict]],
    current_tag: str,
    candidate_pool_size: int = 3,
) -> int:
    """
    Pick a session from the least-loaded sessions using a min-heap.

    Strategy:
    - Pop up to `candidate_pool_size` least-loaded sessions.
    - Prefer sessions that do not already end with the same tag, to reduce
      local tag clustering.
    - Randomly choose among the best candidates.
    - Reinsert all non-chosen sessions.
    - Reinsert chosen session with updated size.

    This keeps session sizes roughly balanced while preserving randomness.
    """
    popped: List[tuple[int, float, int]] = []
    k = min(candidate_pool_size, len(heap))

    for _ in range(k):
        popped.append(heapq.heappop(heap))

    # Prefer sessions whose last assigned tag is different from current_tag
    preferred = []
    fallback = []

    for size, tie, session_num in popped:
        session_videos = session_to_videos[session_num]
        if session_videos and session_videos[-1]["tag"] == current_tag:
            fallback.append((size, tie, session_num))
        else:
            preferred.append((size, tie, session_num))

    candidate_group = preferred if preferred else fallback

    # Randomly choose one candidate among the best few
    chosen = random.choice(candidate_group)
    chosen_size, _, chosen_session_num = chosen

    # Push back all non-chosen popped sessions unchanged
    for item in popped:
        if item != chosen:
            heapq.heappush(heap, item)

    # Push back chosen session with updated size
    new_size = chosen_size + 1
    heapq.heappush(heap, (new_size, random.random(), chosen_session_num))

    return chosen_session_num

def demo_video_shufflers():
    video_list = {
        "1": [
            StimulationVideo(
                id = 0, 
                video_name= "Bastard_Set_Of_Dreams", 
                source_dataset= "DEAP", 
                emo_tag= "", 
                # local_path= "/Users/lily/Documents/capstone/Code/videos/DEAP/videos/Bastard_Set_Of_Dreams.mp4",
                local_path= "/Users/lily/Documents/capstone/Code/code/util/video_trimmer/norway2.mp4",
                width = 320,
                height = 240
                ),
            StimulationVideo(
                id = 0, 
                video_name= "Blame It On The Boogie", 
                source_dataset= "DEAP", 
                emo_tag= "", 
                # local_path= "/Users/lily/Documents/capstone/Code/videos/DEAP/videos/Blame_It_On_The_Boogie.mp4",
                local_path= "/Users/lily/Documents/capstone/Code/code/util/video_trimmer/norway2.mp4",
                width = 472,
                height = 360
                ),
            ]
        }
    return video_list


def trial_video():
    video_list = [
            StimulationVideo(
                id = 0, 
                video_name= "Norway", 
                source_dataset= "Trial", 
                emo_tag= "", 
                local_path= "/Users/lily/Documents/capstone/Code/code/util/video_trimmer/norway_80s.mp4",
                width = 472,
                height = 360
                ),
            StimulationVideo(
                id = 0, 
                video_name= "Norway", 
                source_dataset= "Trial", 
                emo_tag= "", 
                local_path= "/Users/lily/Documents/capstone/Code/code/util/video_trimmer/norway_110s.mp4",
                width = 320,
                height = 240
                ),

            ]
    return video_list

def tiny_trial_video():
    video_list = [
            StimulationVideo(
                id = 0, 
                video_name= "Norway", 
                source_dataset= "Trial", 
                emo_tag= "", 
                local_path= "/Users/lily/Documents/capstone/Code/code/util/video_trimmer/norway_tiny.mp4",
                width = 472,
                height = 360
                ),
            StimulationVideo(
                id = 0, 
                video_name= "Norway", 
                source_dataset= "Trial", 
                emo_tag= "", 
                local_path= "/Users/lily/Documents/capstone/Code/code/util/video_trimmer/norway_tiny.mp4",
                width = 320,
                height = 240
                ),

            ]
    return video_list



def get_washout_video() -> StimulationVideo:
    return StimulationVideo(
            local_path="/Users/lily/Documents/myApps/Capstone_Saveme/5_stimulation_materials/Emognition/washout.mp4",
            height=720,
            width=1280
    )



# video_shufflers(
#     input_path = "/Users/lily/Documents/myApps/Capstone_Saveme/5_stimulation_materials/all_videos_select_1.csv",
#     active_tags = {
#         "Trial": True,
#         "Amusing": True,
#         "Anger": False,
#         "Disgust": True,
#         "Exciting": False,
#         "Fear": True,
#         # "Shock": False,
#         # "Funny": False,
#         "Sad": True,
#         "Happy": True,
#         "Neutral": True,
#         # "Awe": False,
#         # "Liking": False
#         # "中性": True
#     },
#     multi_session = True,
#     temp_json_dir_path = "/Users/lily/Documents/myApps/Capstone_Saveme/1_data_archive/509/videos",
#     temp_json_path = "",
#     num_sessions = None,
#     shuffle = False
# )

# print(
#     video_shufflers(
#     input_path = "/Users/lily/Documents/myApps/Capstone_Saveme/5_stimulation_materials/all_videos_select_1.csv",
#     active_tags = {
#         "Amusing": True,
#         "Anger": False,
#         "Disgust": True,
#         "Exciting": False,
#         "Fear": True,
#         # "Shock": False,
#         # "Funny": False,
#         "Sad": True,
#         "Happy": True,
#         "Neutral": True,
#         # "Awe": False,
#         # "Liking": False
#         # "中性": True
#     },
#     multi_session = True,
#     temp_json_dir_path = "/Users/lily/Documents/myApps/Capstone_Saveme/1_data_archive/509/videos",
#     temp_json_path = "",
#     num_sessions = None,
#     session_num=1,
#     shuffle = False
# )
# )




# # TODO
# OUTPUT_DATA_FOLDER="/Users/lily/Documents/myApps/Capstone_Saveme/1_data_archive/509/2_emotions"

# # VIDEO_LIST = "/Users/lily/Documents/myApps/Capstone_Saveme/5_stimulation_materials/all_videos.csv"
# VIDEO_LIST = "/Users/lily/Documents/myApps/Capstone_Saveme/5_stimulation_materials/all_videos_select_1.csv"

# # TODO
# VIDEO_JSON_DIR = "/Users/lily/Documents/myApps/Capstone_Saveme/1_data_archive/509/videos"


# video_list = video_shufflers(
#     input_path = "/Users/lily/Documents/myApps/Capstone_Saveme/5_stimulation_materials/all_videos_select_1.csv",
#     active_tags = active_tags,
#     multi_session = True,
#     temp_json_dir_path = VIDEO_JSON_DIR,
#     temp_json_path = "",
#     num_sessions = None,
#     session_num=0,
#     shuffle = False
# )




def video_shufflers_one_per_session(
    input_path: str, # csv
    active_tags, # json
    temp_json_path = "",
    shuffle = True,
    trial = True
# ) -> Dict[Any, List["StimulationVideo"]]:
):
    """
    Steps:
    1. Read videos from CSV
    2. Group by tag
    3. Shuffle videos within each tag
    4. Distribute videos into sessions using a min-heap so session sizes stay
       roughly balanced, even when:
         - number of sessions != number of videos per tag
         - different tags have different numbers of videos
    5. Shuffle videos inside each session while ensuring that:
       last tag of previous session != first tag of current session
    6. Save JSON cache
    7. Convert dicts to StimulationVideo objects

    Returns:
        {
            session_num: [StimulationVideo(...), ...],
            ...
        }
    """

    temp_json_path = Path(temp_json_path)

    # If input is JSON cache -> load and skip processing
    if temp_json_path.is_file() and temp_json_path.suffix.lower() == ".json":
        return load_json(temp_json_path)
    
    
    videos = csv_reader(input_path)
    # print(videos)

    # 1) Group by tag
    videos_by_tag = {}
    trial_videos = {}
    fear_videos={}
    total_videos = 0
    total_fear_videos = 0
    
    for video in videos:
        tag = video.get("tag")
        # print(tag)
        if tag is None:
            raise ValueError(f"Video missing 'tag': {video}")
        if tag == "Trial":
            if tag in trial_videos.keys(): trial_videos[tag].append(video)
            else: trial_videos[tag] = [video]
            total_videos += 1
        elif tag == "Fear":
            if tag in fear_videos.keys(): fear_videos[tag].append(video)
            else: fear_videos[tag] = [video]
            total_fear_videos += 1
            total_videos += 1
        elif active_tags[tag]:
            if tag in videos_by_tag.keys(): videos_by_tag[tag].append(video)
            else: videos_by_tag[tag] = [video]
            total_videos += 1

    # print(videos_by_tag)
    if not videos_by_tag:
        raise ValueError("No videos found in CSV meets the requirements.")


    # 2) Shuffle within each tag
    for key, tag_videos in videos_by_tag.items():
        # print(key, tag_videos)
        
        random.shuffle(tag_videos)
        # print(tag_videos)

    random.shuffle(fear_videos['Fear'])

    


    # 3) shuffle tags
    # print(videos_by_tag.keys())
    # Get items as a list of tuples
    items = list(videos_by_tag.items())

    # Shuffle the list in-place
    random.shuffle(items)

    # Create a new dictionary with the shuffled order
    videos_by_tag = dict(items)

    trial_and_tag_videos = trial_videos | videos_by_tag

    print(json.dumps(fear_videos, indent=4))
    # print(json.dumps(trial_and_tag_videos, indent=4))

    print(total_fear_videos, total_videos)
    fear_indexes = random.sample(range(2, total_videos-2, 3), total_fear_videos)
    fear_indexes.sort()
    print(fear_indexes)

    # print(trial_and_tag_videos)

    session_to_videos: Dict[int, List[dict]] = {}

    index = 0
    fear_indexes_pointer = 0

    for _, video_in_tag in trial_and_tag_videos.items():
        for video in video_in_tag:
            if ((fear_indexes_pointer) <= (total_fear_videos - 1)) and index == fear_indexes[fear_indexes_pointer]:
                session_to_videos[index] = [fear_videos["Fear"][fear_indexes_pointer]]
                fear_indexes_pointer += 1
                index += 1
            session_to_videos[index] = [video]
            index += 1

    # print("session_to_video=============", session_to_videos)
    # for index, tag_videos in enumerate(videos_by_tag.values()):

    #     session_to_videos[index] = tag_videos


    # print(videos_by_tag.keys())
    # random.shuffle(videos_by_tag.keys())

    # print(videos_by_tag)


    # if not shuffle:
    #     session_to_videos: Dict[int, List[dict]] = {
            
    #     }

    #     index = 0

    #     for index, tag_videos in enumerate(videos_by_tag.values()):

    #         session_to_videos[index] = tag_videos


    # else:
    #     # 3) Create session buckets
    #     session_to_videos: Dict[int, List[dict]] = {
    #         session_num: [] for session_num in range(num_sessions)
    #     }

    #     # 4) Min-heap of (current_session_size, random_tiebreaker, session_num)
    #     #
    #     # The random tiebreaker prevents deterministic bias when multiple sessions
    #     # have the same size.
    #     heap: List[tuple[int, float, int]] = [
    #         (0, random.random(), session_num) for session_num in range(num_sessions)
    #     ]
    #     heapq.heapify(heap)

    #     # Process tags in random order so one tag does not always dominate early
    #     tag_items = list(videos_by_tag.items())
    #     random.shuffle(tag_items)

    #     for tag, tag_videos in tag_items:
    #         for video in tag_videos:
    #             session_num = _pick_session_from_heap(
    #                 heap=heap,
    #                 session_to_videos=session_to_videos,
    #                 current_tag=tag,
    #             )
    #             session_to_videos[session_num].append(video)

    #     # 5) Shuffle each session with boundary constraint across sessions
    #     previous_last_tag = None
    #     for session_num in range(num_sessions):
    #         session_videos = session_to_videos[session_num]

    #         if not session_videos:
    #             continue

    #         shuffled_session = _shuffle_session_with_boundary_constraint(
    #             session_videos,
    #             previous_last_tag=previous_last_tag,
    #         )
    #         session_to_videos[session_num] = shuffled_session

    #         if shuffled_session:
    #             previous_last_tag = shuffled_session[-1]["tag"]

    # print(json.dumps(session_to_videos, indent=4))
    write_json(session_to_videos, temp_json_path)


    # 6) Convert dicts to StimulationVideo objects
    result: Dict[int, List["StimulationVideo"]] = {}
    for session_num, session_videos in session_to_videos.items():
        result[session_num] = [
            StimulationVideo(
                id=video["id"],
                video_name=video["video_name"],
                source_dataset=video["source_dataset"],
                emo_tag=video["tag"],
                local_path=video["absolute_path"],
                width=video["frame_width"],
                height=video["frame_height"],
            )
            for video in session_videos
        ]

    return result


active_tags = {
    "Amusing": True,
    "Anger": True,
    "Disgust": True,
    "Exciting": False,
    "Fear": True,
    # "Shock": False,
    # "Funny": False,
    "Sad": True,
    "Happy": True,
    "Neutral": True,
    
    # "Awe": False,
    # "Liking": False
    # "中性": True
}

video_shufflers_one_per_session(
    input_path = "/Users/lily/Documents/myApps/Capstone_Saveme/5_stimulation_materials/all_videos_select_1.csv",
    active_tags = active_tags,
    temp_json_path = "/Users/lily/Documents/myApps/Capstone_Saveme/4_psychopy_data_collector_double_ppg/temp.json",
    shuffle = True,
) 