import os
import json
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def save_flicker_img(vid_path: str, init_sec, flicker_frames: list = None, raw_name=None) -> None:
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    window = [0]*3
    success, frame = True, 0
    while success:
        success, img = cap.read()
        window[frame % 3] = img

        if flicker_frames and (frame in flicker_frames):
            show_images([window[(frame - 2) % 3], window[(frame - 1) % 3], window[frame % 3]],
                        save=True,
                        filename=f"flicker_images/{vid_path[-8:-4] if raw_name is None else raw_name}_frame_{frame}.jpg")
        elif flicker_frames is None and frame >= (init_sec * fps) and frame <= (init_sec + 1) * fps:
            cv2.imwrite(
                f"flicker_images/{vid_path[-8:-4] if raw_name is None else raw_name}_frame_{frame}.jpg", img)

        frame += int(success)
    cap.release()


def writeimg_new_labels():
    matplotlib.use('Agg')
    plt.ioff()
    raw_labels = json.load(open('new_label.json', 'r'))
    mapping = json.load(open('mapping.json', "r"))
    inv_map = {v: k for k, v in mapping.items()}
    for vid in raw_labels:
        if inv_map[vid] in mapping.keys():
            save_flicker_img(
                f'flicker-detection/{inv_map[vid]}', 1, raw_labels[vid], raw_name=vid)
        plt.close('all')
        print(vid)


def show_images(images: list[np.ndarray], save=False, filename=None) -> None:
    f = plt.figure(figsize=(10, 6))
    for idx, img in enumerate(images):
        f.add_subplot(1, len(images), idx + 1)
        plt.imshow(img)
    plt.savefig(filename) if save else plt.show(block=True)


def read_proto_string(infile: str = '0824.textproto', outfile: str = 'new_label.json') -> dict:
    with open(infile, 'r') as f:
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

    json.dump(dic, open(outfile, "w"))
    return dic


def multiclass_labels(
    infile: str = '0824.textproto',
    outfile: str = 'multi_label.json'
) -> dict:
    import ast
    with open(infile, 'r') as f:
        lines = f.readlines()
    labels = {}
    for i in range(0, len(lines), 5):
        flicker = "".join(lines[i:i+5])\
            .replace('\n', '')\
            .replace(" ", "")\
                    .split(":")
        if len(flicker) < 4:
            continue

        vid_name = flicker[-2][1:-6]
        flicker_idxs = ast.literal_eval(flicker[-1].replace("}", ""))

        labels[f'{flicker_idxs[0]}_{vid_name}'] = 1
        labels[f'{flicker_idxs[-1]}_{vid_name}'] = 1
        if len(flicker_idxs) > 2:
            labels[f'{flicker_idxs[0]}_{vid_name}'] = 2
            labels[f'{flicker_idxs[-1]}_{vid_name}'] = 3
            labels.update({
                f"{i}_{vid_name}": 4
                for i in flicker_idxs[1:-1]
            })

    json.dump(labels, open(outfile, "w"))
    return labels


def histogram(
    labels: dict,
    save_path: str
) -> None:
    res = Counter(labels.values())
    print(res)
    plt.bar(['isolated flicker', 'start flicker', 'end flicker',
            'between flicker'], list(res.values()))
    plt.xlabel("Class 1-4")
    plt.ylabel("Instance")
    plt.title("Multiclass Histogram")
    plt.savefig(save_path)
    plt.show()


def add_normal_vid_labels(
    new_labels_json: str = "new_label.json",
    videos_path: str = 'flicker-detection',
) -> None:
    new_labels = json.load(open(new_labels_json, "r"))
    for f in os.listdir(videos_path):
        if f.split(".mp4")[0] not in new_labels:
            new_labels[f.split(".mp4")[0]] = []
    json.dump(new_labels, open(new_labels_json, "w"))


def videos_mapping(
    videos_path: str = 'flicker-detection',
    mapping_path: str = "mapping.json"
) -> None:
    mapping = {
        f'{str(idx).zfill(4)}': file.split(".mp4")[0].replace(" ", "")
        for idx, file in enumerate(os.listdir(videos_path))
        if file[-4:] == '.mp4'
    }
    json.dump(mapping, open(mapping_path, "w"))


def label_aug(
    mapping_path: str = "mapping.json",
    aug_dir: str = "augmented"
) -> None:
    mapping = json.load(open(mapping_path, "r"))
    map_reverse = {
        encode: num
        for num, encode in mapping.items()
    }
    for aug_vid in os.listdir(aug_dir):
        name = aug_vid.split("_aug")
        mapping["{}{}".format(map_reverse[name[0]],
                              name[1].replace(".mp4", ""))] = mapping[map_reverse[name[0]]]
    json.dump(mapping, open(mapping_path, "w"))


if __name__ == "__main__":
    """
    sudomen

    ffmpeg -i video.mp4 -vf select='between(n\,x\,y)' -vsync 0 -start_number x frames%d.png
    for ex if the label frame is 475, i will run:
    ffmpeg -i in.mp4 -vf select='between(n\,460\,490)' -vsync 0 -start_number 475 frames%d.png
    correct:
    ffmpeg -i in.mp4 -vf select='between(n\,460\,490)' -vsync 0 -start_number 460 frames%d.png

    fyi the command to get the frame pts
    ffprobe -i test_01.mp4 -show_frames | grep pkt_pts_time

    Convert variable frame rate to standard fps
    for file in flicker-detection/*;
        do ffmpeg -y -i $file -c copy -f h264 "h264_vids/${file:18:4}.h264";
    done
    for file in h264_vids/*;
        do ffmpeg -y -r 30 -i $file -c copy "standard_fps_vid/${file:10:4}.mp4";
    done

    for file in augmented/*;
       do ffmpeg -y -i $file -c copy -f h264 "h264_vids/${file:10:6}.h264";
    done

    for file in h264_vids/*;
        do ffmpeg -y -r 30 -i $file -c copy "standard_fps_vid/${file:10:6}.mp4";
    done
    Check frame count ffmpeg
    ffmpeg -i "path to file" -f null /dev/null

    ?????? opencv problem?
    https://github.com/opencv/opencv/issues/17257

    # ffprobe -i test_01.mp4 -show_frames -select_streams v:0 -print_format flat | grep pkt_pts_time=
    ffmpeg
    """
    # save_flicker_img("flicker-detection/0145.mp4", 13)
    # writeimg_new_labels()
    # manual_label('flicker-detection/0136.mp4')
    # for vid in os.listdir('standard_fps_vid/'):
    #     check_fps(os.path.join('standard_fps_vid/',vid))
    # check_fps('flicker-detection/0001.mp4')
    # read_proto_string()
    # add_normal_vid_labels()
    # videos_mapping()
    # label_aug()

    multi_labels = multiclass_labels()
    histogram(multi_labels, "../plots/label_classes.jpg")
