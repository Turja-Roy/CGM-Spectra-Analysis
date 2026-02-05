#!/bin/bash

group_1=("p1_n2" "p1_n1" "p1_0" "p1_1" "p1_2")
group_2=("p2_n2" "p2_n1" "p2_0" "p2_1" "p2_2")
group_7=("p7_n2" "p7_n1" "p7_0" "p7_1" "p7_2")
group_8=("p8_n2" "p8_n1" "p8_0" "p8_1" "p8_2")
group_9=("p9_n2" "p9_n1" "p9_0" "p9_1" "p9_2")

groups=("group_1" "group_2" "group_7" "group_8" "group_9")

for group_name in "${groups[@]}"; do
    declare -n group=$group_name
    tmux new-window -n "$group_name"
    for sim in "${group[@]}"; do
        tmux split-window -h "source .venv/bin/activate && python downloader.py --set 1P --sim $sim --snapshot 80; read -p \"Press Enter to continue...\""
        tmux select-layout tiled
    done
done
