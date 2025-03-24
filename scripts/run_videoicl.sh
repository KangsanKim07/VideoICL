#! /bin/bash

SPLIT_NUM=1
python qwen2vl.py --annot-dir data/UCF-Crimes/simrank --data-dir data/UCF-Crimes/videos --split-num $SPLIT_NUM