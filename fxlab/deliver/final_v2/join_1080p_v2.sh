#!/bin/sh
# 把三段 1080p 无损拼接（不重新编码）。用法：sh join_1080p.sh   （三个文件放在当前目录）
set -e
printf "file 'final_1080p_v2_part1.mp4'\nfile 'final_1080p_v2_part2.mp4'\nfile 'final_1080p_v2_part3.mp4'\n" > join_list.txt
ffmpeg -y -f concat -safe 0 -i join_list.txt -c copy -movflags +faststart final_1080p_v2_full.mp4
echo "已生成 final_1080p_v2_full.mp4"
