import dotenv

dotenv.load_dotenv(override=True)

import json
import multiprocessing as mp
import os
import re
import shutil
import traceback
import types
from math import ceil
from pathlib import Path
import torch
import click
import pandas as pd
from datasets import load_dataset
from PIL import Image
from tqdm import trange
from transformers import AutoProcessor, AutoModelForImageTextToText
# from internvl_chat import InternVLChatConfig
from vllm import LLM, SamplingParams
from vllm.utils import get_open_port
import numpy as np
import torch.nn.functional as F
import tqdm
Image.MAX_IMAGE_PIXELS = 100_000_000

def main(**kwargs):
    try:
        _main(**kwargs)
    except Exception as e:
        global_dp_rank = -1
        if "global_dp_rank" in kwargs:
            global_dp_rank = kwargs["global_dp_rank"]

        print(f"Rank [{global_dp_rank}]:  Exception occurred: {e}")
        traceback.print_exc()

        exit(1)

def _main(
    *,
    dp_size,
    local_dp_rank,
    global_dp_rank,
    gpu_rank,
    dp_master_ip,
    dp_master_port,
    tp_size,
    args,
    barrier,
):

    # engine processes.

    gpu_ids = range(local_dp_rank * tp_size, (local_dp_rank + 1) * tp_size)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else None
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        devices[gpu_id + gpu_rank] if devices else str(gpu_id + gpu_rank)
        for gpu_id in gpu_ids
    )
    print(f"Rank [{global_dp_rank}]: Using GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # Sample prompts.
    # ---------- Load dataset ----------
    dataset_name = args.dataset_name
    subset = args.subset
    split = args.split
    dataset_size = args.dataset_size
    num_proc = args.num_proc

    # ds = load_dataset(dataset_name, subset)[split]
    ds = load_dataset("json", data_files=dataset_name)[split]
    # ds = ds.shuffle(seed=42) 
    if dataset_size:
        ds = ds.select(range(dataset_size))
    
    # test dataloading
    model = args.model
    trust_remote_code = args.trust_remote_code

    min_pixels = 128*128
    max_pixels = 512*512
    processor = AutoProcessor.from_pretrained(model, min_pixels=min_pixels, max_pixels=max_pixels, trust_remote_code=trust_remote_code)

    build_prompt(ds[0], processor, args)

    promts_per_rank = ceil(len(ds) / dp_size)
    start = global_dp_rank * promts_per_rank
    end = min(start + promts_per_rank, len(ds))
    ds = ds.select(range(start, end))

    output_dir = Path(args.output_dir)
    output_dir = output_dir / "shards"
    out_file = output_dir / f"dp_{global_dp_rank}.jsonl"
    if out_file.exists() and not args.overwrite:
        dataset_index_set = set()
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    result = json.loads(line.strip())
                    dataset_index_set.add(result["global_idx"])
        original_len_ds = len(ds)
        # filtered already processed dataset
        ds = ds.filter(
            lambda row: row["global_idx"] not in dataset_index_set,
            num_proc=num_proc,
            keep_in_memory=True,
        )
        new_len_ds = len(ds)
        print(
            f"Rank [{global_dp_rank}]: Filtered dataset from {original_len_ds} to {new_len_ds} records."
        )


    if len(ds) == 0:
        print(f"Rank [{global_dp_rank}]: have no data; exiting.")
        barrier.wait()
        return

    model = args.model
    # config = InternVLChatConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
    # config.max_dynamic_patch = 4  

    llm = AutoModelForImageTextToText.from_pretrained(
        model,
        dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
        attn_implementation="eager",
        device_map="auto"
    )

    if "qwen" in args.model.lower():
        img_end_token_id = 151653 
    elif "intern" in args.model.lower():
        img_end_token_id = 151666 
    else:
        assert 0, f"Error!"

    llm.zero_grad()
    with torch.no_grad():
        batch_size = args.batch_size
        for start_idx in trange(
            0,
            len(ds),
            batch_size,
            unit_scale=batch_size,
            desc=f"[Global DP Rank {global_dp_rank}] Processing dataset",
        ):
            end_idx = min(len(ds), start_idx + batch_size)

            ds_chunk = ds.select(range(start_idx, end_idx))

            text_list = []
            image_list = []
            # reasoning_steps_ids_list = []

            for row in ds_chunk:
                item = build_prompt(row, processor, args) 
                text_list.append(item["prompt"])
                # reasoning_steps_ids_list.append(item["reasoning_steps_ids"])
                image_list.append(item["multi_modal_data"]["image"])

            inputs = processor(
                text=text_list,
                images=image_list, 
                padding=True, 
                truncation=True,
                return_tensors="pt"
            )
            inputs = inputs.to("cuda")
            input_ids = inputs["input_ids"]

            system_prompt_ids, vision_ids, instruction_ids, response_ids = _location_ids(input_ids=input_ids, img_end_token_id=img_end_token_id)

            try:
                output = llm(**inputs, output_attentions=True, output_hidden_states=True)
            except:
                print(input_ids.shape)
                print(image_list[0][0].size)
                print(input_ids[0][:200])
                assert 0

            attentions = output.attentions
            mean_visual_attention_allocations = _attention_compute(attentions, system_prompt_ids, vision_ids, instruction_ids, response_ids)
            
            results = []

            for idx, row in enumerate(ds_chunk):

                results.append(
                    {
                        "idx": row["idx"],
                        "global_idx": row["global_idx"],
                        "images": row["images"],
                        "question": row["problem"],
                        "answer": row["answer"],
                        "output_text": row["output_text"],
                        "vas_score": mean_visual_attention_allocations[idx],
                    }
                )

            torch.cuda.empty_cache()

            output_dir = Path(args.output_dir)
            output_dir = output_dir / "shards"
            output_dir.mkdir(parents=True, exist_ok=True)

            out_file = output_dir / f"dp_{global_dp_rank}.jsonl"
            print(f"Saving results to '{out_file}'...")
            with open(out_file, "a", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"\nSaved {len(results)} records to '{out_file}'.")

        output_dir = Path(args.output_dir)
        output_dir = output_dir / "shards"
        output_dir.mkdir(parents=True, exist_ok=True)

        out_file = output_dir / f"dp_{global_dp_rank}.jsonl"
        if not out_file.exists():
            if barrier is not None:
                barrier.wait()
            print(f"Rank [{global_dp_rank}]: No output file found. Exiting.")
            return

        # NOTE(xk) Wait for all processes to finish before exiting.
        # Otherwise, the main process (using pytorch dist) may exit before all processes finish writing.
        if barrier is not None:
            barrier.wait()

def _location_ids(input_ids, img_end_token_id):
    system_prompt_ids, vision_ids, instruction_ids, response_ids = [], [], [], []
    for i in range(input_ids.shape[0]):
        system_prompt_ids.append(torch.tensor([(input_ids[i]==151644).nonzero()[0][0], (input_ids[i]==151645).nonzero()[0][0]+1]))
        try:
            vision_ids.append(torch.tensor([(input_ids[i]==151644).nonzero()[1][0], (input_ids[i]==img_end_token_id).nonzero()[0][0]+1]))
        except:
            print(input_ids[i])
            assert 0
        instruction_ids.append(torch.tensor([(input_ids[i]==img_end_token_id).nonzero()[0][0]+1, (input_ids[i]==151645).nonzero()[1][0]+1]))
        if (input_ids[i]==151645).nonzero().shape[0] ==2:
            response_ids.append(torch.tensor([(input_ids[i]==151644).nonzero()[2][0]+1, -1]))
        else:
            response_ids.append(torch.tensor([(input_ids[i]==151644).nonzero()[2][0]+1, (input_ids[i]==151645).nonzero()[2][0]+1]))
    
    system_prompt_ids = torch.stack(system_prompt_ids, dim=0)
    vision_ids = torch.stack(vision_ids, dim=0)
    instruction_ids = torch.stack(instruction_ids, dim=0)
    response_ids = torch.stack(response_ids, dim=0)
    return system_prompt_ids, vision_ids, instruction_ids, response_ids


def _attention_compute(attentions, system_prompt_ids, vision_ids, instruction_ids, response_ids):    
    all_attention_allocation_mean=[]
    for i in range(attentions[0].shape[0]):
        props_sys, props_vis, props_text = 0, 0, 0
        attention=attentions[int(len(attentions)/2)]
        attention = torch.mean(attention, dim=1)
        atten = attention[i].cpu().detach().float().numpy()
        props_sys += atten[response_ids[i][0]:response_ids[i][1],system_prompt_ids[i][0]:system_prompt_ids[i][1]].sum()
        props_vis += atten[response_ids[i][0]:response_ids[i][1],vision_ids[i][0]:vision_ids[i][1]].sum()
        props_text += atten[response_ids[i][0]:response_ids[i][1],instruction_ids[i][0]:instruction_ids[i][1]].sum()
        all_attention_allocation_mean.append((props_vis/(props_sys+props_vis+props_text)).tolist())

    return all_attention_allocation_mean

def extract_answer(text: str) -> str:
    """Extract the model’s final outputs."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.S)
    answer = m.group(1).strip() if m else text.strip()
    p = re.search(r"\((.*?)\)", answer, re.S)
    return p.group(1).replace(" ", "") if p else answer


def grade_answer(prediction, answer, answer_label=None):
    if answer_label is not None:
        if prediction.strip().lower() == f"{answer_label}. {answer}".strip().lower():
            return True
        elif prediction.strip().lower() == answer_label.strip().lower():
            return True

    if prediction.strip().lower().split(" ")[0] == answer.strip().lower():
        return True

    return False


def build_prompt(row, processor, args):
    messages = build_messages(row, processor, args)
    if getattr(args, "debug", False):
        print(f"Prompt: {messages}...")
    return messages

SYSTEM_PROMPT = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user a concise final answer in a short word or phrase. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"


def build_messages(row, processor, args):
    system = SYSTEM_PROMPT
    prompt = row["problem"].replace("<image>", "")
    output_text = row["output_text"]
    images = row["images"]

    content=[]
    new_images=[]
    for image in images:
        new_images.append(image)
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    
    message=[
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": content,
                },
                {
                    "role": "assistant",
                    "content": output_text,
                }
            ]
    prompt = processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=False
    )

    open_images = []
    max_edge = 512
    for image in new_images:
        img = Image.open(image).convert("RGB")
        if max(img.size) > max_edge: 
            img.thumbnail((max_edge, max_edge))  
        open_images.append(img)

    images_input = {
        "multi_modal_data": {"image": open_images},
    }

    return {
        "prompt": prompt,
        **images_input,
    }

def merge_output(output_dir):
    output_dir = Path(output_dir)

    output_shard_dir = output_dir / "shards"
    if not output_shard_dir.exists():
        raise FileNotFoundError(
            f"Output directory '{output_shard_dir}' does not exist."
        )

    # out_file_list = output_shard_dir.glob("*.jsonl")
    out_file_list = sorted(output_shard_dir.glob("*.jsonl"), key=lambda p: p.name)
    results = []
    for out_file_ in tqdm.tqdm(out_file_list):
        with open(out_file_) as f:
            for line in f:
                if line.strip():
                    result = json.loads(line.strip())
                    results.append(result)

    out_file = output_dir / "eval_results.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(results)} records to '{out_file}'.")

@click.command()
@click.option(
    "--model",
    type=str,
    default="Qwen/Qwen2.5-0.5B-Instruct",
    help="Model name or path",
    show_default=True,
)
@click.option(
    "--dp_size", type=int, default=1, help="Data parallel size", show_default=True
)
@click.option(
    "--tp_size", type=int, default=1, help="Tensor parallel size", show_default=True
)
@click.option(
    "--node_size", type=int, default=1, help="Total number of nodes", show_default=True
)
@click.option(
    "--node_rank",
    type=int,
    default=0,
    help="Rank of the current node",
    show_default=True,
)
@click.option(
    "--gpu_rank",
    type=int,
    default=0,
    help="Rank of the start gpu",
    show_default=True,
)
@click.option(
    "--master_addr",
    type=str,
    default="",
    help="Master node IP address",
    show_default=True,
)
@click.option(
    "--master_port", type=int, default=0, help="Master node port", show_default=True
)
@click.option("--enforce_eager", is_flag=True, help="Enforce eager mode execution.")
@click.option("--trust_remote_code", is_flag=True, help="Trust remote code.")
@click.option("--max_model_len", type=int, default=16384, help="Max model length.")
@click.option(
    "--gpu_memory_utilization",
    type=float,
    default=0.6,
    help="GPU memory utilization fraction.",
)
@click.option("--dtype", type=str, default="bfloat16", help="Model dtype.")
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility.")
# sampling
@click.option(
    "--temperature",
    type=float,
    default=0.0,
    help="Sampling temperature",
    show_default=True,
)
@click.option(
    "--top_p", type=float, default=1.0, help="Top-p sampling", show_default=True
)
@click.option(
    "--max_tokens",
    type=int,
    default=2048,
    help="Max tokens to generate",
    show_default=True,
)
@click.option(
    "--n", type=int, default=1, help="Number of samples to generate", show_default=True
)
# chat template
@click.option("--model_prompt_type", type=str, default=None)
@click.option("--instruction_prompt", type=str, default=None)
# dataset
@click.option("--dataset_name", default="SLAKE")
@click.option("--subset", default="MM")
@click.option("--split", default="test")
@click.option(
    "--num_proc", type=int, default=16, help="Number of processes for dataset loading."
)
@click.option("--dataset_size", type=int, default=None, help="Debug subset size.")
# inference
@click.option("--batch_size", default=256, type=int)
# output
@click.option("--output_dir", default="outputs", type=str)
@click.option("--overwrite", is_flag=True, help="Overwrite output directory.")
# debug
@click.option("--debug", is_flag=True)
# misc
@click.option("--ignore_image", is_flag=True, help="Ignore image inputs.")

def multiprocess(**kwargs):
    args = types.SimpleNamespace(**kwargs)

    output_dir = Path(args.output_dir)
    print(f"Output directory: {output_dir}, checking...")
    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            print(f"Output directory '{output_dir}' already exists. Overwriting.")
            shutil.rmtree(output_dir)
        else:
            print(f"try to resume from existing output directory '{output_dir}'.")
    output_dir.mkdir(parents=True, exist_ok=True)

    # save args
    args_file = output_dir / "args.json"
    with open(args_file, "w", encoding="utf-8") as f:
        json.dump(
            vars(args),
            f,
            indent=2,
            ensure_ascii=False,
        )

    dp_size = args.dp_size
    tp_size = args.tp_size
    node_size = args.node_size
    node_rank = args.node_rank
    gpu_rank =  args.gpu_rank

    if node_size == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port = get_open_port()
    else:
        dp_master_ip = args.master_addr
        dp_master_port = args.master_port
        print(
            f"Although set those variables, we do not use them. Using master address: {dp_master_ip}, port: {dp_master_port}"
        )

    assert dp_size % node_size == 0, "dp_size should be divisible by node_size"
    dp_per_node = dp_size // node_size

    if args.debug is True:
        print("In debug mode")
        main(
            dp_size=1,
            local_dp_rank=0,
            global_dp_rank=0,
            gpu_rank=gpu_rank,
            dp_master_ip=dp_master_ip,
            dp_master_port=dp_master_port,
            tp_size=1,
            args=args,
            barrier=None,
        )
        exit()

    from multiprocessing import Barrier, Process

    procs = []
    num_process = len(range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node))
    barrier = Barrier(num_process)
    for local_dp_rank, global_dp_rank in enumerate(
        range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node)
    ):
        proc = Process(
            target=main,
            kwargs=dict(
                dp_size=dp_size,
                local_dp_rank=local_dp_rank,
                global_dp_rank=global_dp_rank,
                gpu_rank=gpu_rank,
                dp_master_ip=dp_master_ip,
                dp_master_port=dp_master_port,
                tp_size=tp_size,
                args=args,
                barrier=barrier,
            ),
        )
        proc.start()
        procs.append(proc)

    exit_code = 0
    for proc in procs:
        # proc.join(timeout=300)
        proc.join()
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} that didn't stop within 5 minutes.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode
    if exit_code == 0:
        merge_output(args.output_dir)

    exit(exit_code)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    multiprocess()
