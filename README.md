# Assignment 1：BPE 与 Transformer

本人实现的 BPE 和 decoder-only Transformer，包含 OWT 正式训练、W&B 源码记录、六小时累计计时、BF16 与 torch.compile 实验支持。

最佳正式结果：**完整 OWT 验证 PPL 27.9381**，累计耗时约5小时48分钟。

- [最终成绩与提交链接](docs/submission.md)
- [代码目录说明](docs/code-layout.md)
- [服务器运行指南](docs/server-training.md)
- [实验与故障恢复记录](docs/experiments/2026-10-20.md)
- [Stanford leaderboard 调研](docs/reference/leaderboard-review.md)
- [最佳超参数配置](configs/owt_best.json)
- [原始 Stanford 作业说明](cs336_assignment1_basics.pdf)

## 目录

```text
assignment1-basics/
├── transformer_main.py       # 正式 Transformer 实验入口
├── record_bpe.py             # 正式 BPE 实验入口
├── wandb_record.py           # W&B 源码和评分记录
├── run_server.sh             # check / bpe / train / best
├── w-b.sh                    # 个人实验默认配置（不含API Key）
├── cs336_basics/             # 手写算法、训练循环和生成
├── scripts/                  # 检查、基准、搜索和配置启动工具
├── configs/                  # 可复用超参数
├── docs/                     # 操作说明、最终成绩与实验历史
├── tests/                    # 测试与官方适配
└── test.sh                   # pytest快捷入口
```

## 在课程服务器使用

使用已安装的公共环境：

```bash
source /lmfs2026/env/a1/bin/activate
source ./w-b.sh
bash run_server.sh check
```

公共数据实际路径为 `/ai/lmfs2026/data/assignment1-data`。可用 `ASSIGNMENT_DATA_DIR`、`SERVER_PYTHON` 覆盖数据目录和Python路径。

运行BPE：

```bash
bash run_server.sh bpe --num-threads 8
```

查看最佳配置生成的命令（不会启动训练）：

```bash
bash run_server.sh best \
  --tokenizer-prefix /ai/lmfs2026/sunyuquan/assignment1-basics/outputs/507bb0852a58/bpe/tokenizer \
  --print-command
```

正式运行时移除 `--print-command`，并按实际GPU分配设置 `CUDA_VISIBLE_DEVICES`。启动器会创建新实验，重新编码数据并从程序入口累计计时；已有正式成绩保持不变。额外训练参数可在命令末尾覆盖，例如 `--run-name`。

自定义训练通过 `bash run_server.sh train --tokenizer-prefix ...` 或 `python transformer_main.py --help` 查看选项。

## 辅助工具

在项目根目录运行：

```bash
python -m scripts.check_server --help
python -m scripts.benchmark_encoding --help
python -m scripts.benchmark_runtime --help
python -m scripts.server_search --help
```

## 本地检查

沿用已有Python环境即可：

```bash
PYTHON=.venv/bin/python bash test.sh -q \
  tests/test_runtime_options.py tests/test_controller_recovery.py \
  tests/test_server_training.py tests/test_model.py tests/test_optimizer.py \
  tests/test_nn_utils.py tests/test_data.py tests/test_serialization.py
```

运行全部测试可执行 `.venv/bin/python -m pytest`。依赖由 `pyproject.toml` 与 `uv.lock` 记录，课程公共环境保持只读。
