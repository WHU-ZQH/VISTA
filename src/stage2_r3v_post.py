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

new_data=[]

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

pos_solution, neg_solution = {}, {}
with jsonlines.open(data_path) as r:
    for id, item in enumerate(r):
        pos_solution[id] = []
        neg_solution[id] = []
        answer= item["answer"]
        for output in item["parsed_outputs"]:
            if output["is_correct"] and format_reward(output["output_text"]) == 1 and accuracy_reward(output["output_text"], answer) == 1 and not contains_chinese(output["output_text"]) and "<image>" not in output["output_text"] and "</image>" not in output["output_text"]:
                pos_solution[id].append(output["output_text"])
            if not output["is_correct"] and format_reward(output["output_text"]) == 1 and not contains_chinese(output["output_text"]) and "<image>" not in output["output_text"] and "</image>" not in output["output_text"]:
                neg_solution[id].append(output["output_text"])


with jsonlines.open(data_path) as r:
    for id, item in enumerate(r):
        prompt = item["question"] if "question" in item else item["problem"]
        images= item["images"]
        answer= item["answer"]

        for output in item["parsed_outputs"]:
            if output["is_correct"] and format_reward(output["output_text"]) == 1 and accuracy_reward(output["output_text"], answer) == 1 and not contains_chinese(output["output_text"]) and "<image>" not in output["output_text"] and "</image>" not in output["output_text"]:
                new_data.append({
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
                                "content": output["output_text"],
                                "role": "assistant"
                            }
                        ],
                        "answer": answer,
                    })
                
        if len(neg_solution[id]) !=0 and len(pos_solution[id]) !=0:
            for solution in pos_solution[id]:
                self_refine_prompt=f"{prompt}\n\nModel Prediction:\n{random.choice(neg_solution[id])}\n\nRefine the above model prediction, and output the correct solution."
                new_data.append({
                        "images": images,
                        "messages": [
                            {
                                "content": SYSTEM_PROMPT,
                                "role": "system"
                            },
                            {
                                "content": self_refine_prompt,
                                "role": "user"
                            },
                            {
                                "content": solution,
                                "role": "assistant"
                            }
                        ],
                        "answer": answer,
                    })

                neg_context=""
                for i, neg_sol in enumerate(neg_solution[id][:2]):
                    neg_context += f"Model Prediction{i+1}:\n{neg_sol}\n\n"
                self_select_prompt = f"{prompt}\n\n{neg_context}Model Prediction{i+2}:\n{solution}\n\nWhich prediction is collect? Give the final answer for the beginning question by selecting the correct option."

                new_data.append({
                        "images": images,
                        "messages": [
                            {
                                "content": "You are a helpful AI assistant.",
                                "role": "system"
                            },
                            {
                                "content": self_select_prompt,
                                "role": "user"
                            },
                            {
                                "content": answer,
                                "role": "assistant"
                            }
                        ],
                        "answer": answer,
                    })

print(len(new_data))
with open(save_path, "w", encoding="utf-8") as save_f:
    json.dump(new_data, save_f, ensure_ascii=False, indent=4)
