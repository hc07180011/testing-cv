#!/bin/bash
for f in *.mp4; do
    ffmpeg -i $f -vf scale=180:360 -preset slow -crf 18 -vsync passthrough reduced_$f;
done;

# count frames
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -print_format csv reduced_17271FQCB00002_video_6.mp4