# VISTA 代码说明

本目录是论文 [Learn to Think: Improving Multimodal Reasoning through Vision-Aware Self-Improvement Training](https://arxiv.org/abs/2605.11931) （ICML 2026）的独立代码整理版。
论文链接：[ICML 2026](https://icml.cc/virtual/2026/poster/66575) · [OpenReview](https://openreview.net/forum?id=2UJrwCrqbw) · [arXiv](https://arxiv.org/abs/2605.11931)。
完整安装、训练与评测命令见 [英文 README](README.md)。

- `run.py`：可配置路径、GPU 数量和训练模式的入口；加 `--dry-run` 只预览命令。
- `src/`：论文方法的采样、评分、数据筛选和评测实现。
- `dataset/`：5 个任务的训练/测试文本，以及 3 个额外评测集，共 13 份 JSONL。
- `docs/DATA.md`：数据条数、字段和图像目录说明。
- `scripts/prepare_data.py`：将标注中的图像文件名转换为本机绝对路径。
- `docs/REPRODUCIBILITY.md`：论文配置、修复内容与尚未验证的部分。
- `tests/`：无需 GPU 的回归测试。
- `LICENSE`：Apache-2.0；第三方版权声明保留。

代码包包含问题、答案、提示词、任务名和图像路径，不包含图像、模型权重和历史训练输出。
补齐对应图像后，运行 `scripts/prepare_data.py --image-root 图像目录 --output-dir prepared_inputs --check-images`。
种子生成还需提供 `fig1.jpg`、`fig2.jpg`、`fig3.jpg`，并用 `VISTA_FEWSHOT_IMAGE_ROOT` 指定其目录。
本次只验证代码语法、合成数据处理及运行流程预览，没有运行 GPU 训练或完整复现实验。
训练、采样环境需要按 CUDA 与框架版本分别安装；依赖范围不是已验证的版本锁定文件。
