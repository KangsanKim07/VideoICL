import torch
import json
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F
import argparse
import os
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


def put_examples(examples, data_dir):
    messages = []
    for ex in examples:
        folder = ex.split('_')[0].split('0')[0]
        if folder == "Normal":
            folder = "Normal_Videos_event"
        video_path = f"{data_dir}/{folder}/{ex}"

        ex_answer = ex.split('_')[0].split('0')[0]
        if ex_answer == 'Normal':
            ex_answer = "Normal Event"
        elif ex_answer == 'RoadAccidents':
            ex_answer = 'Road Accident'
        
        messages += [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "max_pixels": 224 * 224,
                            "max_frames": 16,
                            "min_frames": 1,
                            "fps": 1,
                        },
                        {
                            "type": "text",
                            "text": "Classify the following video into one of the following categories: Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, Normal Event, Road Accident, Robbery, Shooting, Shoplifting, Stealing, or Vandalism. Just answer the name of the category."
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": ex_answer
                        }
                    ]
                }
            ]
    
    return messages


def get_data(annot_dir, data_dir, split_num):
    with open(os.path.join(annot_dir, f'video_top100_sim_subset{split_num}.json'), 'r') as f:
        annot = json.load(f)
        sim_rank = {}
        for i in annot:
            sim_rank[i['test_sample']] = i['train_examples'] 
    
    with open(os.path.join("data/UCF-Crimes/Action_Regnition_splits", f'test_00{split_num}.txt'), 'r') as f:
        test_vids = f.readlines()
        test_vids = [x.replace(' \n', '') for x in test_vids]
    return sim_rank, test_vids


def load_model_and_processor():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto", do_sample=False
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    return model, processor


def prepare_messages(data_dir, candidates, n_shot, video_path, question_sample, max_frames_num):
    examples = candidates[:n_shot]
    messages = put_examples(examples, data_dir)
    messages += [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 384 * 384,
                    "max_frames": max_frames_num,
                    "min_frames": 1,
                    "fps": 1,
                },
                {
                    "type": "text",
                    "text": question_sample
                }
            ]
        }
    ]
    return messages


def process_inference(model, processor, messages):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    cont = model.generate(**inputs, max_new_tokens=20, output_scores=True, return_dict_in_generate=True)
    generated_ids = cont.sequences
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    scores = cont.scores
    scores = torch.concat(list(scores), dim=0)
    probs = F.softmax(scores, dim=-1)
    prob = probs.max(dim=-1).values.min().item()
    return output_text, prob


def evaluate_output(output_text, answer, acc, max_confi, prob):
    if prob > 0.5:
        max_confi = prob
    else:
        if prob > max_confi:
            max_confi = prob

    if answer.lower() in output_text.lower():
        acc.append(1)
    else:
        acc.append(0)

    return acc, max_confi


def inference(annot_dir, data_dir, split_num):
    model, processor = load_model_and_processor()
    sim_rank, test_vids = get_data(annot_dir, data_dir, split_num)
    n_shot = 2
    n_iter = 4
    max_frames_num = 32
    question_sample = "Classify the following video into one of the following categories: Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, Normal Event, Road Accident, Robbery, Shooting, Shoplifting, Stealing, or Vandalism. Just answer the name of the category."
    acc = []

    for vid in tqdm(test_vids):
        video_path = os.path.join(data_dir, vid)
        candidates = sim_rank[vid.split('/')[-1]]
        max_text_outputs = None
        max_confi = 0

        for num in range(n_iter):
            messages = prepare_messages(data_dir, candidates, n_shot, video_path, question_sample, max_frames_num)
            output_text, prob = process_inference(model, processor, messages)

            if prob > 0.5:
                max_text_outputs = output_text
                max_confi = prob
                break
            else:
                if prob > max_confi:
                    max_confi = prob
                    max_text_outputs = output_text

        answer = vid.split('/')[0]
        if answer == 'RoadAccidents':
            answer = 'Road Accident'
        elif answer == 'Normal_Videos_event':
            answer = 'Normal Event'

        acc, max_confi = evaluate_output(max_text_outputs, answer, acc, max_confi, prob)
    
    print("Acc:", round(sum(acc) / len(acc), 5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot-dir', type=str, required=True, help='Path to the annotation directory')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--split-num', type=int, default=1, help='Split number')
    args = parser.parse_args()
    inference(args.annot_dir, args.data_dir, args.split_num)