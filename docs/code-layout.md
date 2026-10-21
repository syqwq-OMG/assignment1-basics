# 代码目录与入口

## 日常入口

| 文件 | 用途 |
| --- | --- |
| `transformer_main.py` | 正式 OWT Transformer 实验：源码快照、编码、训练、最终完整验证和累计计时 |
| `record_bpe.py` | BPE 训练、保存、完整验证与 W&B 评分记录 |
| `wandb_record.py` | W&B 初始化和源码审计 |
| `run_server.sh` | `check`、`bpe`、`train`、`best` 统一入口 |
| `w-b.sh` | 非敏感 W&B 默认配置；API Key 由个人登录配置管理 |
| `configs/owt_best.json` | 最佳正式实验的模型和训练超参数 |
| `test.sh` | 在当前 Python 环境运行 pytest，透传参数 |

## 实现

- `cs336_basics/BPE/`：本人 BPE、预分词及原有调试笔记。
- `cs336_basics/Trans4mer/`：手写 Transformer、RoPE、RMSNorm、SwiGLU、优化器、损失和 checkpoint。
- `cs336_basics/encoding.py`：按安全文档边界进行有序、限流并行编码。
- `cs336_basics/main.py`：模型训练循环、精度/编译选项、完整验证和推理入口。
- `cs336_basics/decode.py`：生成时的采样逻辑。
- `cs336_basics/doc/`：原有 Typst 作业笔记。
- `tests/`：官方适配与本项目的运行、恢复和数值检查。

`optim1zer.py` 是训练和官方适配实际导入的优化器文件；`operat0r.py` 是实际导入的基础运算文件。保留原有名称以维持导入兼容。`optimizer.py` 是另一份原有学习草稿，尚有未提交改动，未用于正式训练。

## 辅助脚本

在项目根目录使用模块形式执行：

```bash
python -m scripts.check_server --help
python -m scripts.benchmark_encoding --help
python -m scripts.benchmark_runtime --help
python -m scripts.server_search --help
python -m scripts.train_best --help
```

`server_search` 是10月20日实验调度器，固定GPU#3并包含当次候选和恢复流程。用于新分配时应先调整其runtime设备配置、日期标签和截止时间。

`scripts/make_submission.sh` 保留 Stanford 原始压缩打包方式。本课程正式提交内容见 [提交说明](submission.md)。

## 文档和产物

- `docs/server-training.md`：服务器操作和计时说明。
- `docs/from-scratch.md`：新服务器安装、原始数据下载、从零生成vocab/merges与训练模型的完整流程。
- `docs/submission.md`：最终成绩、提交链接和模型位置。
- `docs/experiments/2026-10-20.md`：训练和故障恢复历史；以开头的最终结果为准。
- `docs/reference/leaderboard-review.md`：Stanford公开配置调研。
- `outputs/`、`wandb/`、`ignored/`：本地生成文件或资料，仍由 `.gitignore` 排除。

已完成实验的服务器目录和W&B源码快照保持原样。整理后的代码已并入 `main`；正式成绩对应的源码以各run快照和 `train/precision-compile` 分支历史为准。
