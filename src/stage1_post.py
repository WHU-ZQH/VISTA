import jsonlines
import json
import os
import re
import random
random.seed(1234)
import sys
from mathruler.grader import grade_answer

data_path = sys.argv[1]
save_path = sys.argv[2]

sft_data = []

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

def accuracy_reward(response: str, ground_truth: str) -> float:
    try:
        content_match = re.search(r"<answer>(.*?)</answer>", response)
        given_answer = content_match.group(1).strip() if content_match else response.strip()
        if grade_answer(given_answer, ground_truth.strip()):
            return 1.0

    except Exception:
        pass

    return 0.0

SYSTEM_PROMPT = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user a concise final answer in a short word or phrase. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"

with jsonlines.open(data_path) as r:
    for item in r:
        prompt = item["question"]
        images= item["images"]
        answer= item["answer"]

        right_outputs = []

        for output in item["parsed_outputs"]:
            if output["is_correct"]:
                if format_reward(output["output_text"]) == 1 and accuracy_reward(output["output_text"], answer) == 1 and not contains_chinese(output["output_text"]) and "<image>" not in output["output_text"] and "</image>" not in output["output_text"] and len(output["output_text"])>300:
                    right_outputs.append(output["output_text"])

        if right_outputs!=[]:
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
                            "content": random.choice(right_outputs),
                            "role": "assistant"
                        }
                    ],
                    "answer": answer,
                })

print(len(sft_data))
with open(save_path, "w", encoding="utf-8") as save_f:
    json.dump(sft_data, save_f, ensure_ascii=False, indent=4)
