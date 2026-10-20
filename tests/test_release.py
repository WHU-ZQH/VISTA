import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def script(self, name, *args):
        subprocess.run([sys.executable, str(ROOT/'src'/name), *map(str, args)], check=True, capture_output=True, text=True)

    def test_constant_scores_and_individual_dpo_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            scores, prior = [], []
            for index, answer in enumerate(['blue', 'red']):
                question = f'Question {index}'
                for _ in range(2):
                    scores.append(dict(idx=index, question=question, images=['synthetic.png'], answer=answer,
                                       output_text=f'<think>Reason.</think><answer>{answer}</answer>', vas_score=0.5))
                prior.append(dict(question=question, images=['synthetic.png'], answer=answer,
                                  parsed_outputs=[dict(is_correct=False, output_text='<think>Wrong.</think><answer>green</answer>')]))
            for filename, records in [('scores.jsonl', scores), ('prior.jsonl', prior)]:
                (d/filename).write_text(''.join(json.dumps(row)+'\n' for row in records), encoding='utf-8')
            self.script('stage4_post_sft.py', d/'scores.jsonl', d, 1)
            self.assertEqual(len(json.loads((d/'our_sft_vas_score_iter1.json').read_text())), 4)
            self.script('stage4_post_dpo.py', d/'scores.jsonl', d/'prior.jsonl', d, 1)
            result = json.loads((d/'our_dpo_vas_score_iter1.json').read_text())
            self.assertEqual([row['answer'] for row in result], ['blue', 'blue', 'red', 'red'])

    def test_training_plans_are_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in ['sft', 'dpo']:
                out = Path(directory)/mode
                result = subprocess.run([sys.executable, str(ROOT/'run.py'), '--base-model', 'base-model',
                    '--train-data', 'absent.jsonl', '--llamafactory', 'absent-framework', '--output', str(out),
                    '--mode', mode, '--iterations', '1', '--dry-run'], capture_output=True, text=True, check=True)
                self.assertFalse(out.exists())
                self.assertIn('stage3_prefix_sampling.py', result.stdout)
                self.assertIn('--model_name_or_path base-model', result.stdout)
                self.assertIn('--gradient_accumulation_steps '+('4' if mode=='sft' else '2'), result.stdout)


if __name__ == '__main__':
    unittest.main()
