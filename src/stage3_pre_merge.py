import jsonlines
import json
import os
import re
import random
random.seed(1234)
import sys
import re
from tqdm import tqdm

data_path = sys.argv[1]
save_path = sys.argv[2]

new_data=[]
global_idx=0

def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0

def contains_chinese(s):
    return bool(re.search(r'[\u4e00-\u9fff\u3400-\u4dbf\uac00-\ud7af'
                          r'\u3040-\u309f\u30a0-\u30ff]', s))

with jsonlines.open(data_path) as r:
    for id, item in tqdm(enumerate(r)):
        prompt = item["question"]
        images= item["images"]
        answer= item["answer"]

        for output in item["parsed_outputs"]:
            if not output["is_correct"] and not contains_chinese(output["output_text"]) and format_reward(output["output_text"]) == 1:
                think_output = output['output_text'].split("</think>")[0]
                new_data.append({"idx": id, "global_idx": global_idx, "images": images, "problem": prompt, "output_text": think_output, "answer": answer})
                global_idx += 1

print(len(new_data))
with open(save_path, "w", encoding="utf-8") as f:
    for result in new_data:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
