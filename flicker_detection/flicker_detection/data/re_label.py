import os
import json
import cv2
import numpy as np


def save_flicker_img(vid_path: str, init_sec):
    cap = cv2.VideoCapture(vid_path)
    h, w, total, fps = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(
        cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    vid_arr = np.zeros((int(total)+1, int(h), int(w), 3))
    success, frame = True, 0
    while success:
        success, vid_arr[frame] = cap.read()
        if frame >= (init_sec * fps) and frame <= (init_sec + 2) * fps:
            cv2.imwrite(
                f"flicker_images/{vid_path[-8:-4]}_frame_{frame}.jpg", vid_arr[frame])
        frame += int(success)
    cap.release()
    return h, w, total


def label_aug():
    label = json.load(open("label.json", "r"))
    mapping = json.load(open("mapping.json", "r"))
    # print(mapping.keys())
    vids = dict(map(lambda s: (s[:4], s), mapping.keys()))
    print(vids)
    for aug_vid in os.listdir("data/augmented/"):
        if aug_vid[:4] in vids:
            mapping[aug_vid] = mapping[vids[aug_vid[:4]]]
    json.dump(mapping, open("mapping_test.json", "w"))


if __name__ == "__main__":
    save_flicker_img("flicker-detection/0096.mp4", 5)
