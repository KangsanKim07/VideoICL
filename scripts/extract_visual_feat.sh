#! /bin/bash

MODEL_PATH=$1

cd InternVideo/InternVideo2/multi_modality

# pip install torch peft open_clip_torch
# pip install -r requirements.txt
# git clone https://github.com/Dao-AILab/flash-attention.git
# cd flash-attention/csrc/layer_norm && pip install .
# cd ../../..

python extract_videofeats.py --data-path ../../../data/UCF-Crimes --model-path $MODEL_PATH --annot-path ../../../data/UCF-Crimes/Action_Regnition_splits