#!/bin/bash

# some videos seem to refuse to reduce its resolution
for f in *.mp4; do
    ffmpeg -i $f -vf scale=180:360 -preset slow -crf 18 -vsync passthrough reduced_$f;
done;

# count frames
for f in *.mp4; do
    echo $f &&
    ffprobe \
    -v error \
    -select_streams v:0 -count_frames \
    -show_entries stream=nb_read_frames \
    -print_format csv $f 
done > ../log.txt;

# count file in directory
ls -f | wc -l

# measure memory size
du -sh .

# get free GB
free -g -h -t