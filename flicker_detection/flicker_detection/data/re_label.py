import os
import json
import cv2
import numpy as np


def save_flicker_img(vid_path: str, init_sec, flicker_frames: list = None):
    cap = cv2.VideoCapture(vid_path)
    h, w, total, fps = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(
        cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    # vid_arr = np.zeros((int(total)+1, int(h), int(w), 3))
    success, frame = True, 0
    while success:
        success, img = cap.read()

        if flicker_frames and frame in flicker_frames:
            cv2.imwrite(
                f"flicker_images/{vid_path[-8:-4]}_frame_{frame}.jpg", img)
        elif flicker_frames is None and frame >= (init_sec * fps) and frame <= (init_sec + 2) * fps:
            cv2.imwrite(
                f"flicker_images/{vid_path[-8:-4]}_frame_{frame}.jpg", img)

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


def read_proto_string():
    with open('new_label.textproto', 'r') as f:
        lines = f.readlines()
    dic = {}
    vid_name = None
    for line in lines:
        if 'video' in line:
            vid_name = str(line[10:-2])
            dic[vid_name] = None
        elif 'frame' in line and 'flicker' not in line:
            if ',' in line:
                dic[vid_name] = list(
                    map(int, line[10:-2].replace(' ', '').split(',')))
            elif line[10:-2] != '':
                dic[vid_name] = [int(line[10:-2])]
    with open('new_label.json', 'w') as out:
        json.dump(dic, out)
    return dic


def writeimg_new_labels():
    raw_labels = json.load(open('new_label.json', 'r'))
    mapping = json.load(open('mapping.json', "r"))
    inv_map = {v: k for k, v in mapping.items()}
    for vid in raw_labels:
        if inv_map[vid] in mapping.keys():
            save_flicker_img(
                f'flicker-detection/{inv_map[vid]}', 1, raw_labels[vid])


def manual_label(vid_path: str,):
    cap = cv2.VideoCapture(vid_path)
    success, frame = cap.read()
    h, w, total, fps = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(
        cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    count = 0
    p_frame = frame
    while success:
        frame = cv2.resize(frame, (int(w//3), int(h//3)))
        cv2.imshow(f'frame_{count}', frame)
        cv2.imshow(f'Previous_frame_{count-1}', p_frame)
        k = cv2.waitKey(5000)
        if k == ord('q'):
            cv2.destroyAllWindows()
            break
        if k == ord('s'):
            cv2.imwrite(
                f"flicker_images/{vid_path[-8:-4]}_frame_{count}.jpg", frame)
        elif k == ord('n'):
            cv2.waitKey(0)
        p_frame = frame
        success, frame = cap.read()
    cap.release()


if __name__ == "__main__":
    # save_flicker_img("flicker-detection/0096.mp4", 5)
    # read_proto_string()
    # writeimg_new_labels()
    manual_label('flicker-detection/0052.mp4')
