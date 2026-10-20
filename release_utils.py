"""Small, dependency-free launcher helpers. No GPU imports during dry runs."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


class Runner:
    def __init__(self, args, root):
        self.args, self.root = args, Path(root)
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus)
        self.out = Path(args.output).expanduser().resolve()
        self.registry = {}

    def start(self):
        if self.args.dry_run:
            return
        if self.out.exists() and any(self.out.iterdir()):
            raise FileExistsError(f"Choose a fresh output directory: {self.out}")
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / 'run.json').write_text(json.dumps(vars(self.args), indent=2), encoding='utf-8')

    def run(self, command, cwd=None, env=None):
        command = [str(x) for x in command]
        print(shlex.join(command), flush=True)
        if not self.args.dry_run:
            subprocess.run(command, cwd=cwd or self.root, env=env or self.env, check=True)

    def script(self, name, *args):
        self.run([self.args.python, self.root / 'src' / name, *args])

    def register(self, name, file, columns, formatting='alpaca', ranking=False, tags=None):
        entry = {'file_name': str(Path(file).resolve()), 'columns': columns, 'formatting': formatting}
        if ranking:
            entry['ranking'] = True
        if tags:
            entry['tags'] = tags
        self.registry[name] = entry
        if not self.args.dry_run:
            data = json.loads(Path(file).read_text(encoding='utf-8'))
            if not data:
                raise ValueError(f'No training examples in {file}; inspect the preceding stage.')
            target = self.out / 'prepared_data'
            target.mkdir(exist_ok=True)
            (target / 'dataset_info.json').write_text(json.dumps(self.registry, indent=2), encoding='utf-8')

    def train(self, model, dataset, output, stage, lr, multimodal=False):
        a = self.args
        n = len(a.gpus.split(','))
        batch = 32 if stage == 'sft' else 16
        if batch % n:
            raise ValueError(f'{n} GPUs do not divide global batch {batch}.')
        repo = Path(a.llamafactory).expanduser().resolve()
        command = [a.train_python or a.python, '-m', 'torch.distributed.run',
                   '--nproc_per_node', n, '--master_port', a.port, repo / 'src/train.py',
                   '--deepspeed', repo / 'examples/deepspeed/ds_z3_config.json',
                   '--stage', stage, '--template', a.template, '--model_name_or_path', model,
                   '--do_train', '--dataset', dataset, '--dataset_dir', self.out / 'prepared_data',
                   '--finetuning_type', 'full', '--output_dir', output, '--cutoff_len', 2048,
                   '--warmup_ratio', 0.1, '--per_device_train_batch_size', 1,
                   '--gradient_accumulation_steps', batch // n, '--lr_scheduler_type', 'cosine',
                   '--logging_steps', 10, '--save_strategy', 'no', '--learning_rate', lr,
                   '--num_train_epochs', 3, '--bf16', '--report_to', 'none', '--seed', 42]
        if stage == 'dpo':
            command += ['--pref_ftx', 0.5, '--pref_beta', 0.1]
        if multimodal:
            command += ['--freeze_vision_tower', 'true', '--freeze_multi_modal_projector', 'true',
                        '--freeze_language_model', 'false', '--image_max_pixels', 262144]
        self.run(command, cwd=repo)


def common_arguments(parser):
    parser.add_argument('--base-model', required=True, help='Local model path or Hugging Face model ID')
    parser.add_argument('--train-data', required=True)
    parser.add_argument('--output', required=True, help='Fresh run directory; existing nonempty directories are refused')
    parser.add_argument('--llamafactory', required=True, help='LLaMA-Factory source checkout')
    parser.add_argument('--mode', choices=['sft', 'dpo'], default='sft')
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--python', default=sys.executable, help='Generation/scoring Python interpreter')
    parser.add_argument('--train-python', help='Separate LLaMA-Factory interpreter')
    parser.add_argument('--port', type=int, default=29501)
    parser.add_argument('--dry-run', action='store_true', help='Print commands without loading models or writing outputs')


def validate_arguments(args):
    for name in ('train_data', 'seed_data', 'preference_seed', 'llamafactory'):
        value = getattr(args, name, None)
        if value:
            setattr(args, name, str(Path(value).expanduser().resolve()))
    if Path(args.base_model).expanduser().exists():
        args.base_model = str(Path(args.base_model).expanduser().resolve())
    gpu_ids = args.gpus.split(',')
    if not gpu_ids or any(not x.isdigit() for x in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError('--gpus must contain distinct comma-separated numeric GPU IDs')
    if args.iterations < 1:
        raise ValueError('--iterations must be positive')
    if 16 % len(gpu_ids):
        raise ValueError('Use 1, 2, 4, 8, or 16 GPUs so the global batch is preserved')
    if not args.dry_run:
        if not Path(args.train_data).is_file():
            raise FileNotFoundError(args.train_data)
        if not (Path(args.llamafactory) / 'src/train.py').is_file():
            raise FileNotFoundError('LLaMA-Factory src/train.py not found')
