import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt


def save_flicker_img(vid_path: str, init_sec, flicker_frames: list = None, raw_name=None):
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    success, frame = True, 0
    while success:
        success, img = cap.read()

        if flicker_frames and frame in flicker_frames:
            cv2.imwrite(
                f"flicker_images/{vid_path[-8:-4] if raw_name is None else raw_name}_frame_{frame}.jpg", img)
        elif flicker_frames is None and frame >= (init_sec * fps) and frame <= (init_sec + 2) * fps:
            cv2.imwrite(
                f"flicker_images/{vid_path[-8:-4] if raw_name is None else raw_name}_frame_{frame}.jpg", img)

        frame += int(success)
    cap.release()


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
            if vid_name not in dic.keys():
                dic[vid_name] = None
        if 'frame' in line and 'flicker' not in line:
            if not dic[vid_name] and ',' in line:
                dic[vid_name] = list(
                    map(int, line[10:-2].replace(' ', '').split(',')))
            elif not dic[vid_name] and line[10:-2] != '':
                dic[vid_name] = [int(line[10:-2])]
            elif dic[vid_name] and ',' in line:
                dic[vid_name].extend(list(
                    map(int, line[10:-2].replace(' ', '').split(','))))
            elif dic[vid_name] and line[10:-2] != '':
                dic[vid_name].append(int(line[10:-2]))

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
                f'flicker-detection/{inv_map[vid]}', 1, raw_labels[vid], raw_name=vid)
        print(vid)


def show_images(images: list[np.ndarray], save: bool = False, raw_vid: str = None) -> None:
    n: int = len(images)
    f = plt.figure(figsize=(10, 6))
    for i in range(n):
        # Debug, plot figure
        f.add_subplot(1, n, i + 1)
        plt.imshow(images[i])
    if save:
        plt.savefig(f"{raw_vid}.jpg")
    plt.show(block=True)


def merge(d1, d2, merge):
    result = dict(d1)
    for k, v in d2.iteritems():
        if k in result:
            result[k] = merge(result[k], v)
        else:
            result[k] = v
    return result


def manual_label(vid_path: str,):
    cap = cv2.VideoCapture(vid_path)
    h, w, total, fps = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(
        cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    vid_arr = [0]*(int(total)+1)
    frame_count, success, labels = 0, True, {}
    while success:
        success, vid_arr[frame_count] = cap.read()
        frame_count += int(success)
    cap.release()

    labels[vid_path] = []
    cur_frame = 2
    while cur_frame < total:

        show_images(
            [vid_arr[cur_frame-2], vid_arr[cur_frame-1], vid_arr[cur_frame]])
        if str(input("Save set?[y/n]\n")) == 'y':
            labels[vid_path].extend([cur_frame-2, cur_frame-1, cur_frame])
            show_images(
                [vid_arr[cur_frame-2], vid_arr[cur_frame-1], vid_arr[cur_frame]], save=True, raw_vid=vid_path)

        if str(input("break?[q]\n")) == 'q':
            break
        cur_frame = int(
            input(f"Choose frame idx: frames from 0 - {int(total)}\n"))

    with open("manual_labels.json", 'r') as infile:
        existing = json.load(infile)

    with open("manual_labels.json", "w") as out:
        labels = merge(existing, labels, lambda x, y: (x, y))
        json.dump(labels, out)
    return h, w, total


if __name__ == "__main__":
    # save_flicker_img("flicker-detection/0096.mp4", 5)
    # read_proto_string()
    # writeimg_new_labels()
    manual_label('flicker-detection/0136.mp4')
