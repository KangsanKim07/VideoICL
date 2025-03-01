import os
import pickle
import json
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity


def get_rank(feat_path, output_dir, split_num):
    test_vids = []
    with open(f'data/UCF-Crimes/Action_Regnition_splits/test_00{split_num}.txt', 'r') as f:
        test_vids += f.readlines()
    test_vids = [x.replace(' \n', '') for x in test_vids]
    train_vids = []
    with open(f'data/UCF-Crimes/Action_Regnition_splits/train_00{split_num}.txt', 'r') as f:
        train_vids += f.readlines()
    train_vids = [x.replace(' \n', '') for x in train_vids]
    all_videos = train_vids + test_vids

    with open(feat_path, 'rb') as f:
        data = pickle.load(f)
        features = {d['video_name']: d['feature'] for d in data}

    train_vid_feat, test_vid_feat = [], []
    train_vid_ids, test_vid_ids = [], []

    for video_name in train_vids:
        video_name = video_name.split('/')[-1]
        feat = features[video_name].squeeze()
        train_vid_feat.append(feat)
        train_vid_ids.append(video_name)

    for video_name in test_vids:
        video_name = video_name.split('/')[-1]
        feat = features[video_name].squeeze()
        test_vid_feat.append(feat)
        test_vid_ids.append(video_name)

    similarity_matrix = cosine_similarity(test_vid_feat, train_vid_feat)
    top_k = 100
    top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, :top_k]
    top_k_similarity_values = np.sort(-similarity_matrix, axis=1)[:, :top_k]
    result = []
    for test_id, topk_indices, similarity_vals in zip(test_vid_ids, top_k_indices, top_k_similarity_values):
        topk_filenames = [train_vid_ids[idx] for idx in topk_indices]
        topk_similarities = [-similarity_vals[idx] for idx in range(len(topk_indices))]
        topk_similarities = [x.item() for x in topk_similarities]
        elem = {
            "test_sample": test_id,
            "train_examples": topk_filenames,
            "similarity": topk_similarities
        }
        result.append(elem)

    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/video_top100_sim_subset{split_num}.json', 'w') as f:
        json.dump(result, f, indent=4)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat-path', type=str, required=True, help='Path to the video features')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to the output directory')
    args = parser.parse_args()
    for split_num in range(1, 5):
        get_rank(feat_path=args.feat_path, output_dir=args.output_dir, split_num=split_num)