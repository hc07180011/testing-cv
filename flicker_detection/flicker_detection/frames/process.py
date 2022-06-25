from cProfile import label
from re import L
import cv2
import os, json, shutil

PATH_TO_VIDEOS = "./frames/videos/testing_set"

class Label:
    def __init__(self, frame: int, padded: bool):
        self.frame = frame
        self.padded = padded

def load_frames(fname):
    mapping_file = open('./data/mapping.json')
    mapping_data = json.load(mapping_file)
    real_name = mapping_data[fname]
    mapping_file.close()

    label_file = open('./data/label.json')
    label_data = json.load(label_file)
    target_frames = label_data[real_name]
    mapping_file.close()

    return target_frames

def pad_labels(target_frames):
    if len(target_frames) == 0:
        return []

    result_frames = []
    last_frame = target_frames[0]
    for j in range(3, 0, -1):
        temp_label = Label(target_frames[0] - j, True)
        result_frames.append(temp_label)
    result_frames.append(Label(target_frames[0], False))

    for i, frame in enumerate(target_frames):
        if (i > 0):
            if (frame != last_frame + 1):
                # pads out three at the end
                for j in range(1, 4):
                    temp_label = Label(last_frame + j, True)
                    result_frames.append(temp_label)
                # assigns three before the next segment
                for j in range(3, 0, -1):
                    temp_label = Label(frame - j, True)
                    result_frames.append(temp_label)
            temp_label = Label(frame, False)
            result_frames.append(temp_label)
            last_frame = frame

    for j in range(1, 4):
        temp_label = Label(frame + j, True)
        result_frames.append(temp_label)
    
    return result_frames

def process(n_video: str):
    print("Processing file 0{}.mp4".format(n_video))
    target_frames = load_frames('0{}.mp4'.format(n_video))
    padded_labels = pad_labels(target_frames)

    vidcap = cv2.VideoCapture(PATH_TO_VIDEOS + '/0{}.mp4'.format(n_video))

    amount_frames = vidcap.get(7)
    print("Frames in the video: {}".format(amount_frames))

    success, frame = vidcap.read()
    count = 0
    next_label = 0
    padded_labels_length = len(padded_labels)
    
    result_folder = './frames/result/0{}/'.format(n_video)
    if os.path.isdir(result_folder):
        shutil.rmtree(result_folder)
    os.mkdir(result_folder)
    while (success and next_label < padded_labels_length):
        if (count == padded_labels[next_label].frame):
            padded = ""
            if padded_labels[next_label].padded:
                padded = "_P"
            f_name = "{}{}{}.jpg".format(result_folder, count, padded)
            cv2.imwrite(f_name, frame)
            print("Saved frame {}{}".format(count, padded))
            next_label += 1

        # if count in target_frames:
        success, frame = vidcap.read()
        count += 1
    
    vidcap.release()
    cv2.destroyAllWindows()