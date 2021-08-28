#!/bin/bash
. host.config
now=$(date +"%Y-%m-%d %H:%M:%S")
# shellcheck disable=SC2002
filename=$(cat /dev/urandom | tr -cd 'a-f0-9' | head -c 32)
ffmpeg -f v4l2 -t 1800 -video_size 640x360 -i /dev/video0 -metadata "creation_time=$now" "$filename.mp4"
# shellcheck disable=SC2154
scp -i ~/.ssh/id_pipeline "$filename.mp4" $user@$host:~/captures/
rm "$filename.mp4"
