import jsonlines
import json
import os
import re
import random
random.seed(1234)
import sys
import numpy as np
from tqdm import tqdm

data_path = sys.argv[1]
prior_data_path = sys.argv[2]
save_path = sys.argv[3]
iteration = sys.argv[4]

def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    if response.count("</answer>") >1:
        return 0.0
    else:
        return 1.0 if format_match else 0.0

def contains_chinese(s):
    return bool(re.search(r'[\u4e00-\u9fff\u3400-\u4dbf\uac00-\ud7af'
                          r'\u3040-\u309f\u30a0-\u30ff]', s))

def filter_output(data, key, filter_thred):
    rejection_sampling, filter_data = [], []
    filter_results= []
    for item in data:
        rejection_sampling.append(item["output_text"])
        filter_data.append(item[key])

    if len(filter_data) >1 :
        score_mean, score_std = np.array(filter_data).mean(), np.array(filter_data).std()
        for res, score in zip(data, filter_data):
            if (0.0 if score_std == 0 else (score-score_mean)/score_std)>filter_thred:
                filter_results.append(res)
    else:
        filter_results = data
    
    return filter_results

def save_data(process_data, rejected_data, key, save_path):
    for filter_thred in [-0.5]:
        new_data = []
        for id, data in process_data.items():
            if len(rejected_data[data[0]["idx"]]["rejected_data"]) >0:
                filter_results = filter_output(data, key, filter_thred)

                for item in filter_results:
                    if "<image>" not in item["output_text"] and "</image>" not in item["output_text"]:
                        new_data.append({
                            "images": item["images"],
                            "system": SYSTEM_PROMPT,
                            "conversations": [
                                    {
                                        "from": "human",
                                        "value": item["question"]
                                    }
                                ],
                                "chosen": {
                                    "from": "gpt",
                                    "value": item["output_text"]
                                },
                                "rejected": {
                                    "from": "gpt",
                                    "value": random.choice(rejected_data[data[0]["idx"]]["rejected_data"])
                                },
                            "answer": item["answer"],
                            })
        print(f"{key} -- {len(new_data)}")
        with open(f"{save_path}/our_dpo_{key}_iter{iteration}.json", "w", encoding="utf-8") as save_f:
            json.dump(new_data, save_f, ensure_ascii=False, indent=4)

SYSTEM_PROMPT = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user a concise final answer in a short word or phrase. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"

rejected_data={}
with jsonlines.open(prior_data_path) as r:
    for id, item in tqdm(enumerate(r)):
        prompt = item["question"]
        images= item["images"]
        answer= item["answer"]

        for output in item["parsed_outputs"]:
            if not output["is_correct"] and format_reward(output["output_text"]) == 1 and not contains_chinese(output["output_text"]) and "<image>" not in output["output_text"] and "</image>" not in output["output_text"]:
                if id not in rejected_data:
                    rejected_data[id] = {"images": images, "question": prompt, "rejected_data":[output["output_text"]]}
                else:
                    rejected_data[id]["rejected_data"].append(output["output_text"])

process_data={}
with jsonlines.open(data_path) as r:
    for item in tqdm(r):
        if item["idx"] in rejected_data.keys():
            assert item["images"] == rejected_data[item["idx"]]["images"], f"idx = {item['idx']}"
            assert item["question"] == rejected_data[item["idx"]]["question"], f"idx = {item['idx']}"
            if format_reward(item["output_text"]) == 1:
                if item["idx"] not in process_data:
                    process_data[item["idx"]] = [item]
                else:
                    process_data[item["idx"]].append(item)

key = "vas_score"
save_data(process_data, rejected_data, key, save_path)
