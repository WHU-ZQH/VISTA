# Bundled text data and image setup

The `dataset/` directory includes 13 JSONL files. No image bytes are bundled.
Each record preserves its question, answer, system prompt and image filename;
`task` is supplied for every record. `dataset/manifest.json` records source and
packaged checksums, counts and any field preparation.

| Dataset directory | Training records | Test records |
|---|---:|---:|
| slake | 1,681 | 355 |
| vqa_rad | 940 | 251 |
| geometry3k | 2,101 | 601 |
| scienceqa | 2,635 | 813 |
| chartqa | 3,233 | 1,026 |
| ood_test_set/mathvista_test.jsonl | - | 1,000 |
| ood_test_set/illusionBench_test.jsonl | - | 5,139 |
| ood_test_set/path_vqa_test.jsonl | - | 3,362 |

Use test files only for evaluation. Benchmark content retains its source dataset
terms; the repository's Apache-2.0 license applies to code.

## Image directories

Place corresponding images under an external root with these relative paths:

```text
IMAGE_ROOT/
  slake_images/
  vqa_rad_images/
  geometry3k_images/
  scienceqa/images/
  chartqa/images/
  MathVista_images/
  pathvqa_images/
  illusionBench/IllusionDataset/
```

Match the exact filenames recorded in `images`. Some filenames are those of
the project's exported images, so copying an upstream dataset without matching
those names may require a mapping step.

Seed generation also uses three few-shot demonstrations whose text is included
in `src/stage1_rejection_sampling_self.py`. Provide their corresponding
`fig1.jpg`, `fig2.jpg`, and `fig3.jpg` externally and set
`VISTA_FEWSHOT_IMAGE_ROOT` to that directory. These demonstration images are
not included in the text-only package.

```bash
python scripts/prepare_data.py --image-root /datasets/vista-images \
  --output-dir prepared_inputs --check-images
```

The output keeps the same split directories and uses absolute image paths.
The script validates all requested image paths before writing files and refuses
to overwrite existing output files. Without images, annotations can be inspected
and paths prepared by omitting `--check-images`; multimodal runs need images.

## Record format

Input is one JSON object per line. Required fields:

| Field | Type | Meaning |
|---|---|---|
| `problem` | string | User question, using the project's image placeholder convention |
| `images` | list of strings | Image paths accessible on the GPU host |
| `answer` | string | Ground-truth answer used for verification |
| `task` | string | Task identifier, e.g. `slake`, `vqa_rad`, `geometry3k` |

Synthetic illustration (not a benchmark example):

```json
{"problem":"<image>Question: What color is the square?","images":["/datasets/synthetic/square.png"],"answer":"blue","task":"slake"}
```

The image in this illustration is not included. Sampling kernels supply the
reasoning-format system prompt. Responses are expected in
`<think>...</think><answer>...</answer>` format. The original post-processors
apply additional format/language filters; these are retained.

Training files produced by the pipeline use ShareGPT `messages` for SFT and
`conversations`, `chosen`, `rejected`, `system`, `images` for DPO. The runner
writes a matching `prepared_data/dataset_info.json` automatically.

Obtain images from the original dataset distributions or your image backup,
then match the bundled filenames and run the preparation command above.
