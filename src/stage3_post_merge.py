import jsonlines
import json
import os
import re
import random
random.seed(1234)
import sys
from tqdm import tqdm

guide_data_path = sys.argv[1]
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

process_data={}
with jsonlines.open(prior_data_path) as r:
    for id, item in tqdm(enumerate(r)):
        prompt = item["question"]
        images= item["images"]
        answer= item["answer"]

        for output in item["parsed_outputs"]:
            if output["is_correct"] and format_reward(output["output_text"]) == 1 and not contains_chinese(output["output_text"]) and "<image>" not in output["output_text"] and "</image>" not in output["output_text"]:
                if id not in process_data:
                    process_data[id] = {"images": images, "question": prompt, "answer": answer, "collect_outputs": [output["output_text"]], "guided_outputs": []}
                else:
                    process_data[id]["collect_outputs"].append(output["output_text"])

with jsonlines.open(guide_data_path) as r:
    for item in r:
        if item["idx"] not in process_data:
            process_data[item["idx"]] = {"images": item["images"], "question": item["question"], "answer": item["answer"], "collect_outputs": [], "guided_outputs": []}
        assert item["images"] == process_data[item["idx"]]["images"], f'{item["idx"]}'
        assert item["question"] == process_data[item["idx"]]["question"]
        for output in item["parsed_outputs"]:
            if output["is_correct"] and format_reward(output["output_text"]) == 1 and not contains_chinese(output["output_text"]) and "<image>" not in output["output_text"] and "</image>" not in output["output_text"]:
                process_data[item["idx"]]["guided_outputs"].append(output["output_text"])

SYSTEM_PROMPT = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user a concise final answer in a short word or phrase. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"

global_idx=0
global_idx2=0

new_data=[]
sft_data = []

process_data = dict(sorted(process_data.items(), key=lambda kv: int(kv[0])))

for id, item in process_data.items():
    prompt = item["question"]
    images= item["images"]
    answer= item["answer"]
    item["num_collect"] = len(item["collect_outputs"])+len(item["guided_outputs"])
    if item["num_collect"] <=10:
        for text in item["collect_outputs"]:
            new_data.append({"idx": id, "global_idx": global_idx, "system": SYSTEM_PROMPT, "images": images, "problem": prompt, "answer": answer, "output_text": text})
            global_idx += 1
        for text in item["guided_outputs"]:
            new_data.append({"idx": id, "global_idx": global_idx, "system": SYSTEM_PROMPT, "images": images, "problem": prompt, "answer": answer, "output_text": text})
            global_idx += 1
    else:
        for text in item["collect_outputs"]:
            new_data.append({"idx": id, "global_idx": global_idx, "system": SYSTEM_PROMPT, "images": images, "problem": prompt, "answer": answer, "output_text": text})
            global_idx += 1
        for text in random.sample(item["guided_outputs"], 10-len(item["collect_outputs"])):
            new_data.append({"idx": id, "global_idx": global_idx, "system": SYSTEM_PROMPT, "images": images, "problem": prompt, "answer": answer, "output_text": text})
            global_idx += 1

print(len(new_data))
with open(f"{save_path}/compute_score_input.jsonl", "w", encoding="utf-8") as f:
    for result in new_data:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

for item in new_data:
    prompt = item["problem"]
    images= item["images"]
    answer= item["answer"]
    sft_data.append({
        "images": images,
        "messages": [
            {
                "content": SYSTEM_PROMPT,
                "role": "system"
            },
            {
                "content": prompt,
                "role": "user"
            },
            {
                "content": item["output_text"],
                "role": "assistant"
            }
        ],
        "answer": answer,
    })

print(len(sft_data))
with open(f"{save_path}/sft_data_guide_iter{iteration}.json", "w", encoding="utf-8") as save_f:
    json.dump(sft_data, save_f, ensure_ascii=False, indent=4)
