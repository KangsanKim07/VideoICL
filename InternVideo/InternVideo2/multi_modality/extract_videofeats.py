import os
import cv2
import argparse
import pickle
from tqdm import tqdm
from demo.config import Config, eval_dict_leaf
from demo.utils import _frame_from_video, setup_internvideo2, frames2tensor


def load_model(model_path):
    config = Config.from_file('demo/internvideo2_stage2_config.py')
    config = eval_dict_leaf(config)
    config['pretrained_path'] = model_path
    config['model']['vision_encoder']['pretrained'] = model_path
    device = 'cuda'
    intern_model = setup_internvideo2(config)
    intern_model.to(device)
    return intern_model, config, device


def get_video_list(annot_path):
    test_vids = []
    for i in range(1,5):
        with open(f'{annot_path}/test_00{i}.txt', 'r') as f:
            test_vids += f.readlines()
    test_vids = [x.replace(' \n', '') for x in test_vids]
    train_vids = []
    for i in range(1,5):
        with open(f'{annot_path}/train_00{i}.txt', 'r') as f:
            train_vids += f.readlines()
    train_vids = [x.replace(' \n', '') for x in train_vids]
    all_videos = list(set(train_vids + test_vids))
    return all_videos


def extract_feat(args):
    intern_model, config, device = load_model(args.model_path)
    all_videos = get_video_list(args.annot_path)
    video_feats = []
    for vid_path in tqdm(all_videos):
        video_path = os.path.join(args.data_path, 'videos', vid_path)
        video = cv2.VideoCapture(video_path)
        frames = [x for x in _frame_from_video(video)]
        fn = config.get('num_frames', 4)
        size_t = config.get('size_t', 224)
        frames_tensor = frames2tensor(frames, fnum=fn, target_size=(size_t, size_t), device=device)
        video_feature = intern_model.get_vid_feat(frames_tensor).cpu().numpy()
        video_feats.append({'video_name':vid_path.split('/')[-1], 'feature':video_feature})

    with open(f'{args.data_path}/vid_feat.pkl', 'wb') as f:
        pickle.dump(video_feats, f)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to the data root directory containing video files. e.g. 'data/UCF-Crimes'")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model checkpoint. e.g. InternVideo2-stage2_1b-224p-f4.pt")
    parser.add_argument("--annot-path", type=str, required=True, help="Path to the annotation directory. e.g. 'data/UCF-Crimes/Action_Regnition_splits'")
    args = parser.parse_args()
    extract_feat(args)
