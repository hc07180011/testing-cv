import os
import moviepy.editor as mp

if __name__ == "__main__":
    videos_root = os.path.join(os.getcwd(), '../data/flicker-detection')
    for vid in os.listdir(os.getcwd()):
        print(vid)
        if not vid.endswith('.gif'):
            continue
        video = mp.VideoFileClip(vid)
        video.write_videofile(videos_root+f'/{vid[:-3]}mp4')
