"""VISTA sampling, refinement, scoring and SFT/DPO orchestration."""
import argparse
from pathlib import Path
from release_utils import Runner, common_arguments, validate_arguments

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    common_arguments(p)
    p.add_argument('--template', default='qwen2_vl')
    p.add_argument('--lr', type=float, default=1e-5)
    a = p.parse_args()
    validate_arguments(a)
    r = Runner(a, ROOT)
    r.start()
    prepared = r.out / 'prepared_data'
    if not a.dry_run:
        prepared.mkdir(exist_ok=True)

    def generate(script, model, data, output, sample=None):
        if not a.dry_run and Path(data).is_file() and not Path(data).read_text(encoding='utf-8').strip():
            output.mkdir(parents=True, exist_ok=True)
            result = output / 'eval_results.jsonl'
            result.write_text('', encoding='utf-8')
            print(f'No candidates for {script}; wrote an empty intermediate result.')
            return result
        options = ['--model', model, '--dataset_name', data, '--split', 'train',
                   '--output_dir', output, '--dp_size', len(a.gpus.split(',')),
                   '--tp_size', 1, '--gpu_rank', 0, '--batch_size', 8, '--trust_remote_code']
        if sample is not None:
            options += ['--n', sample, '--temperature', 1.0, '--max_model_len', 32768,
                        '--max_tokens', 2048, '--gpu_memory_utilization', 0.6, '--enforce_eager']
        r.script(script, *options)
        return output / 'eval_results.jsonl'

    tags = dict(role_tag='role', content_tag='content', user_tag='user', assistant_tag='assistant', system_tag='system')
    def sft_register(name, file):
        r.register(name, file, {'messages': 'messages', 'images': 'images'}, 'sharegpt', tags=tags)

    seed_results = generate('stage1_rejection_sampling_self.py', a.base_model, a.train_data, r.out/'seed_sampling', 10)
    seed = prepared / 'sft_seed.json'
    r.script('stage1_post.py', seed_results, seed)
    sft_register('seed', seed)
    current = r.out / 'models' / 'seed'
    r.train(a.base_model, 'seed', current, 'sft', a.lr, True)
    for iteration in range(1, a.iterations + 1):
        work = r.out / f'iteration_{iteration}'
        prior = generate('stage2_rejection_sampling.py', current, a.train_data, work/'sampling', 10)
        locate_input = work/'sampling'/'locate_input.jsonl'
        r.script('stage3_pre_merge.py', prior, locate_input)
        prefixes = generate('stage3_locate_prefix.py', current, locate_input, work/'prefixes')
        guided = generate('stage3_prefix_sampling.py', current, prefixes, work/'guided', 3)
        r.script('stage3_post_merge.py', guided, prior, work/'guided', iteration)
        scores = generate('stage4_compute_vas_score.py', current, work/'guided'/'compute_score_input.jsonl', work/'scores')
        dataset = f'vista_{a.mode}_{iteration}'
        if a.mode == 'sft':
            r.script('stage4_post_sft.py', scores, prepared, iteration)
            sft_register(dataset, prepared/f'our_sft_vas_score_iter{iteration}.json')
        else:
            r.script('stage4_post_dpo.py', scores, prior, prepared, iteration)
            r.register(dataset, prepared/f'our_dpo_vas_score_iter{iteration}.json',
                       {'messages': 'conversations', 'chosen': 'chosen', 'rejected': 'rejected',
                        'system': 'system', 'images': 'images'}, 'sharegpt', ranking=True)
        current = r.out/'models'/dataset
        # The VISTA paper initializes each SFT/DPO iteration from the base model.
        r.train(a.base_model, dataset, current, a.mode, a.lr, True)


if __name__ == '__main__':
    main()
