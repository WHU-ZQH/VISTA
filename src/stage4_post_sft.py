import jsonlines
import json
import os
import re
import random
random.seed(1234)
import sys
import numpy as np

data_path = sys.argv[1]
save_path = sys.argv[2]
iteration = sys.argv[3]

def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    if response.count("</answer>") >1:
        return 0.0
    else:
        return 1.0 if format_match else 0.0

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

def save_data(process_data, key, save_path):
    for filter_thred in [-0.5]:
        new_data = []
        for id, data in process_data.items():
            filter_results = filter_output(data, key, filter_thred)

            for item in filter_results:
                if "<image>" not in item["output_text"] and "</image>" not in item["output_text"]:
                    new_data.append({
                            "images": item["images"],
                            "messages": [
                                {
                                    "content": SYSTEM_PROMPT,
                                    "role": "system"
                                },
                                {
                                    "content": item["question"],
                                    "role": "user"
                                },
                                {
                                    "content": item["output_text"],
                                    "role": "assistant"
                                }
                            ],
                            "answer": item["answer"],
                            f"{key}": item[key]
                    })
        print(f"{key} -- {len(new_data)}")
        with open(f"{save_path}/our_sft_{key}_iter{iteration}.json", "w", encoding="utf-8") as save_f:
            json.dump(new_data, save_f, ensure_ascii=False, indent=4)

SYSTEM_PROMPT = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user a concise final answer in a short word or phrase. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"

process_data={}
with jsonlines.open(data_path) as r:
    for item in r:
        if format_reward(item["output_text"]) == 1:
            if item["idx"] not in process_data:
                process_data[item["idx"]] = [item]
            else:
                process_data[item["idx"]].append(item)

key = "vas_score"
save_data(process_data, key, save_path)
