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

import click
import pandas as pd
from datasets import load_dataset
from merge_results import compute_results_acc, merge_output
from PIL import Image
from tqdm import trange
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.utils import get_open_port
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
    ds = ds.shuffle(seed=42) 
    if dataset_size:
        ds = ds.select(range(dataset_size))
    
    # test dataloading
    model = args.model
    trust_remote_code = args.trust_remote_code

    min_pixels = 128*128
    max_pixels = 512*512
    processor = AutoProcessor.from_pretrained(model, min_pixels=min_pixels, max_pixels=max_pixels, trust_remote_code=trust_remote_code)

    build_prompt(ds[0], processor, args)

    # with DP, each rank should process different prompts.
    # usually all the DP ranks process a full dataset,
    # and each rank processes a different part of the dataset.
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
                    dataset_index_set.add(result["dp_index"])
        original_len_ds = len(ds)
        # filtered already processed dataset
        ds = ds.filter(
            lambda row: row["dp_index"] not in dataset_index_set,
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

    temperature = args.temperature
    top_p = args.top_p
    max_tokens = args.max_tokens
    n = args.n
    sampling_params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    # Create an LLM.
    model = args.model
    enforce_eager = args.enforce_eager
    gpu_memory_utilization = args.gpu_memory_utilization
    max_model_len = args.max_model_len
    dtype = args.dtype
    seed = args.seed
    llm = LLM(
        model=model,
        tensor_parallel_size=tp_size,
        enforce_eager=enforce_eager,
        # enable_expert_parallel=True,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=trust_remote_code,
        seed=seed,
    )

    # Print the outputs.
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
        prompts = [build_prompt(row, processor, args) for row in ds_chunk]
        outputs = llm.generate(prompts, sampling_params=sampling_params)

        results = []

        for idx, (row, output) in enumerate(zip(ds_chunk, outputs)):

            dp_index = start_idx + idx
            row_prompt = prompts[idx]["prompt"]

            # answer
            answer_label = None
            answer = row["answer"]

            # predictions
            parsed_outputs = []

            for rollout_output in output.outputs:
                output_text = rollout_output.text.strip()

                pred_letter = extract_answer(output_text)
                is_correct = grade_answer(pred_letter, answer, answer_label)

                parsed_outputs.append(
                    {
                        "output_text": output_text,
                        "pred_letter": pred_letter,
                        "is_correct": is_correct,
                    }
                )

            # stats
            num_rollouts = len(parsed_outputs)
            num_correct = sum(1 for o in parsed_outputs if o["is_correct"])

            results.append(
                {
                    "dp_index": dp_index,
                    "images": row["images"],
                    "question": row["problem"] if "problem" in row else row["question"],
                    "answer": answer,
                    "prompts": row_prompt,
                    "parsed_outputs": parsed_outputs,
                    "num_rollouts": num_rollouts,
                    "num_correct": num_correct,
                }
            )

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

    if barrier is not None:
        barrier.wait()


def extract_answer(text: str) -> str:
    """Extract the model’s final outputs."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.S)
    if m:
        # answer = m.group(1).strip() if m else text.strip()
        answer = m.group(1).strip()
        p = re.search(r"\((.*?)\)", answer, re.S)
        return p.group(1).replace(" ", "") if p else answer
    else:
        return "None"


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


def build_messages(row, processor, args):
    system = row["system"]
    prompt = row["problem"] if "problem" in row else row["question"]
    prompt = prompt.replace("<image>", "")
    images = row["images"]

    content=[]
    new_images=[]
    for image in images:
        new_images.append(image)
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt.replace("<image>", "")})
    
    message=[
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": content,
                }
            ]
    prompt = processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )

    # open_images = [Image.open(image).convert("RGB") for image in new_images]
    open_images = []
    max_edge = 512
    for image in new_images:
        img = Image.open(image).convert("RGB")
        if max(img.size) > max_edge:               # 判断最长边
            img.thumbnail((max_edge, max_edge))    # 等比缩小，就地修改
        open_images.append(img)

    images_input = {
        "multi_modal_data": {"image": open_images},
    }

    return {
        "prompt": prompt,
        **images_input,
    }


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
