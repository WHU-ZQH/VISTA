# VISTA

**Learn to Think: Improving Multimodal Reasoning through Vision-Aware Self-Improvement Training**

*ICML 2026*

Qihuang Zhong, Liang Ding, Wenjie Xuan, Juhua Liu, Bo Du, and Dacheng Tao.

[ICML 2026](https://icml.cc/virtual/2026/poster/66575) · 
[arXiv](https://arxiv.org/abs/2605.11931) · [Data format](docs/DATA.md) ·
[Reproducibility notes](docs/REPRODUCIBILITY.md) · [中文说明](README_zh.md)

VISTA improves multimodal reasoning through rejection sampling, prefix-guided
reasoning refinement, and vision-aware selection of self-generated training data.
This repository contains the research implementation, a portable SFT/DPO runner,
evaluation code and the archived EasyR1 GRPO configuration.

## Installation

Use Linux with NVIDIA GPUs and Python 3.10 or 3.11. Generation and attention
scoring require GPU memory appropriate for the selected model. The experiments
used A800 GPUs. CPU tests do not establish GPU runtime compatibility.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install a compatible CUDA/PyTorch/vLLM combination. `requirements.txt` gives
compatibility ranges, **not a verified environment lock**. Attention scoring
requires a Transformers backend that returns attention tensors.

Install [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) in a separate
environment, including its DeepSpeed dependencies. Supply its source checkout
and Python interpreter below. The source snapshot used LLaMA-Factory requirements
with Transformers 4.49–4.57.1 and TRL 0.8.6–0.9.6; do not combine that training
environment blindly with a newer vLLM installation. Record the exact versions
and framework commit IDs used for your run.

## Data

Text annotations are bundled under `dataset/`, with questions, answers, system
prompts, task identifiers and relative image filenames. They cover SLAKE,
VQA-RAD, Geometry3K, ScienceQA, ChartQA, and three additional evaluation sets.
Images and model weights are supplied separately. See [docs/DATA.md](docs/DATA.md)
for split sizes and the image directory layout.

After obtaining the images, resolve their paths for your machine:

```bash
python scripts/prepare_data.py --image-root /datasets/vista-images \
  --output-dir prepared_inputs --check-images
```

This writes separate JSONL files with absolute image paths and leaves bundled
annotations unchanged. Omit `--check-images` to prepare paths before images are
available; actual multimodal training still requires the images.

For seed generation, also set `VISTA_FEWSHOT_IMAGE_ROOT` to the directory
containing the three demonstration images `fig1.jpg`, `fig2.jpg`, `fig3.jpg`.
Their text prompts are included in the code; the images are supplied separately.

## Training

Preview the full pipeline first; `--dry-run` needs only Python and writes nothing:

```bash
python run.py \
  --base-model /models/Qwen2.5-VL-3B-Instruct \
  --train-data prepared_inputs/slake/slake_train.jsonl \
  --llamafactory /opt/LLaMA-Factory \
  --train-python /opt/lf-env/bin/python \
  --output outputs/slake-vista-sft \
  --gpus 0,1,2,3,4,5,6,7 --mode sft --iterations 3 --dry-run
```

Remove `--dry-run` to execute. Use `--mode dpo` for preference optimization and a
different output directory. The runner generates seed SFT data, trains the seed
model, and repeats sampling → prefix localization → guided sampling → VAS scoring
→ filtering → training. Every stage stops if its command fails. Training dataset
registrations are generated automatically; no manual changes to LLaMA-Factory
are required. Existing nonempty run directories are refused.

Both iterative SFT and DPO initialize training from the base model. Each iteration
uses the previous model to generate/select data. Defaults: K=10, J=3, threshold
−0.5, learning rate 1e−5, three epochs, SFT global batch 32 and DPO global batch 16,
beta=0.1, auxiliary SFT weight=0.5. The vision encoder and projector are frozen.
For the paper's ScienceQA/ChartQA and additional-backbone studies use
`--iterations 1`; the main SLAKE/VQA-RAD/Geometry3K experiments use three.
`--template` must match the selected model (default `qwen2_vl`).

Generated files are grouped under `prepared_data/`, `models/`, `seed_sampling/`
and `iteration_N/`. There is no automatic restart from an interrupted pipeline;
inspect the saved `run.json` and intermediate files before starting a fresh run.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python src/evaluation.py \
  --model outputs/slake-vista-sft/models/vista_sft_3 \
  --dataset_name prepared_inputs/slake/slake_test.jsonl --split train \
  --output_dir outputs/slake-eval --dp_size 1 --tp_size 1 --gpu_rank 0 \
  --temperature 0 --n 1 --max_tokens 2048 --trust_remote_code
```

`--split train` is intentional: the local JSON loader creates a `train` split
even when the input file is a test set. Never use the test file during training.
Predictions and aggregate accuracy are written in the evaluation directory.

## Additional experiments

`stage2_post.py` and `stage2_r3v_post.py` retain baseline post-processing code.
The old machine-specific baseline launchers have been superseded; the supported
end-to-end runner covers VISTA SFT/DPO. `scripts/config_vanilla.yaml` and
`scripts/grpo.sh` provide the optional EasyR1 experiment; install EasyR1 separately.
See the reproducibility notes before comparing those runs to paper tables.

## Tests

```bash
pip install -r requirements-cpu.txt
python -m unittest discover -s tests -v
```

The tests exercise synthetic score filtering, per-example labels and runner
planning. They do not load a model or reproduce the paper's GPU experiments.

## Citation and license

See [citation.bib](citation.bib). Code is licensed under [Apache-2.0](LICENSE).
Third-party notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
