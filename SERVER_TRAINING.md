# 服务器运行说明

按 [课程 Assignment 1 说明](https://dasellmg.yuque.com/org-wiki-dasellmg-vsl5wg/lmfs2026/rtubf0h818hevi7h)（2026-10-15 更新）准备。
正式实验使用公共 OWT 数据、本人 BPE 和两个独立 W&B Run。公共环境与原始数据只读。

## 1. 拉取代码并检查环境

在自己的服务器项目目录操作；首次使用可先 `git clone https://github.com/syqwq-OMG/assignment1-basics.git`。

```bash
source /env/activate_a1.sh
cd /ai/lmfs2026/$USER/assignment1-basics
git pull --ff-only
source ./w-b.sh
wandb login
bash run_server.sh check --wandb
```

`wandb login` 首次登录时执行即可，在交互提示中输入自己的 Key，不要写进源码或提交到 Git。
`w-b.sh` 默认 Entity 为 `syqwq-east-china-normal-university`，项目为 `lmfs-assignment1`，学号为 `10234900421`。
可在 source 前 export 同名变量覆盖。`check --wandb` 会创建一个小型 environment-test Run，确认账户和项目可用。
若 Entity 错误，按课程说明先用 W&B 默认 Entity 测试，然后设置实际可用值。

运行器固定使用 `/env/a1/bin/python`，不运行 `uv sync`、`pip install`，不改动公共环境。
仅本地调试可通过 `SERVER_PYTHON=/绝对路径/python` 覆盖。

## 2. BPE 正式实验

```bash
bash run_server.sh bpe
```

默认读取 `/data/assignment1-data/owt_train.txt`，词表上限 32000，特殊 token 为 `<|endoftext|>`。
可传 `--num-threads N`；不传则沿用已有 BPE 实现的自动策略。
在完整 `/data/assignment1-data/owt_valid.txt` 上，以刚加载的分词器计时 encode、decode 和逐字相等检查。
读文件、加载分词器、生成表格及上传均在该计时区间之外。

W&B 提供完整源码、前 200 个 token 表、`bpe/valid_seconds` 和 `bpe/roundtrip_passed`。
终端会打印分词器路径前缀，形如 `outputs/BPE_RUN_ID/bpe/tokenizer`。
同目录保存三个 tokenizer JSON 和 `result.json`，必须保留，并在任务二使用这一份分词器。
不要拿小数据调试得到的分词器用于正式评分。

## 3. Transformer 正式实验

建议在 `tmux` 会话内运行，以便断开 SSH 后继续。先执行 `tmux new -s assignment1`，再激活环境、进入项目目录并运行：

```bash
bash run_server.sh train \
  --tokenizer-prefix outputs/BPE_RUN_ID/bpe/tokenizer \
  --batch-size 32 \
  --context-length 256 \
  --d-model 512 \
  --num-layers 4 \
  --num-heads 16 \
  --d-ff 1344 \
  --max-iters 10000
```

把 `BPE_RUN_ID` 换成上一步真实 ID。这些是现有模型默认参数，未声称是 A100 上的最优配置；正式开始前先做小数据调试确认。
`Ctrl-b` 再按 `d` 可离开 tmux，`tmux attach -t assignment1` 恢复。

入口 `transformer_main.py` 从导入 PyTorch 前开始计时。连接 W&B、源码快照、原文编码、初始化、训练、保存最后模型和完整验证均计入预算。
原文编码调用仓库自己的 BPE，在完整特殊文档标记处切分，保留空白；没有该标记时保留整个文本编码，不任意截断。
不额外插入 BOS/EOS。训练/验证 token 文件保存在本次 Run 的 `outputs/RUN_ID/` 下。

编码默认使用最多 8 个独立 CPU 进程，每个进程保留 65536 项预分词缓存；不使用 Python 线程执行 BPE。
任务按完整 `<|endoftext|>` 边界切分，结果按原文顺序写入，在途任务数最多为进程数的两倍。
没有安全文档标记时自动保留完整文本单任务编码，避免改变分词结果。
每约 10 秒显示编码百分比、吞吐和 ETA。可按分配到的 CPU 资源调整：

```bash
bash run_server.sh train \
  --tokenizer-prefix outputs/BPE_RUN_ID/bpe/tokenizer \
  --encoding-workers 8 \
  --encoding-cache-size 65536 \
  --encoding-chunk-mib 1
```

所有参数记录到 W&B config；编码仍计入累计六小时预算。
已运行的旧进程不会自动采用新实现，不要在其运行目录中覆盖源码。
若需切换实现，应先保留旧 Run 的实际耗时和记录，再启动独立的新 Run，不得重置关联实验累计预算。

可在独立目录运行小样本 CPU 基准，不启动正式训练、不创建 W&B Run，也不修改公共数据：

```bash
python benchmark_encoding.py \
  --tokenizer-prefix /绝对路径/本人BPE/bpe/tokenizer \
  --sample-mib 32 --workers 1 4 8 --output encoding-benchmark.json
```

基准使用本人已训练的分词器，对比原 2048 项缓存的单进程与 65536 项缓存的并行编码；
每个配置都必须产生相同 token 数和相同输出 SHA256。

每 100 次更新记录一次区间训练 PPL；最后不足 100 步的区间也记录。
周期验证是训练期间的随机抽样诊断；最终评分另行遍历完整验证 token 序列，按总 NLL / 总目标 token 数计算 PPL。
验证按不重叠的 context-length 块重置上下文，覆盖每个 next-token 目标一次（共 N−1 个），包含最后不足一块的部分，无 padding。

默认累计预算 21600 秒，预留最后 1800 秒用于保存和完整验证。
**预留时间不是完成时限保证**：需结合服务器实际验证速度调整 `--finalize-reserve-seconds`；大数据编码、单步或验证耗时可能超出预留，超预算会标记失败。
`--max-iters` 达到上限也会提前结束；没有必要为了用满预算继续运行。

本地保存 `outputs/RUN_ID/last.pt` 和 `outputs/RUN_ID/result.json`。
W&B Summary 包含 `val/ppl`、`val/token_count`、`train/final_step`、起止 UTC、`time/total_seconds`、`time/difference_from_6h_seconds`、`time/within_6h`。
模型 checkpoint 只保存在本地，不上传中间 checkpoint 或参数。启动时会打印 Run ID、URL 和输出目录。

## 4. 续训

沿用同一个分词器和模型结构；`--max-iters` 表示累计更新次数目标。

```bash
bash run_server.sh train \
  --tokenizer-prefix outputs/BPE_RUN_ID/bpe/tokenizer \
  --resume outputs/PREVIOUS_RUN_ID/last.pt \
  --previous-timing outputs/PREVIOUS_RUN_ID/result.json \
  --max-iters 20000
```

非默认模型参数须重新传入相同值。入口会核对 tokenizer SHA256、模型结构和 checkpoint 路径，读取上一 Run 的累计耗时并记录关联 Run。
累计六小时已用完时拒绝续训；每次续训重新加载和编码数据所用时间也计入。
优化器和步数恢复；当前 checkpoint 格式不保存 RNG 状态，所以不保证与不中断训练逐位一致。
正常结束或 Python 异常会保存计时结果；强杀、机器断电无法保证完整记录，不能用最后一条训练日志代替真实累计耗时。

## 5. 离线与提交

网络不可用时，在运行前设置：

```bash
export WANDB_MODE=offline
```

恢复网络后使用终端打印的实际离线目录同步：

```bash
wandb sync wandb/offline-run-实际目录
```

确认远端 Run 的源码、结果表、最终评分和训练曲线可查看后，再按课程表单提交两个正式 Run 链接；续训附关联 Run。
调试使用的 TinyStories / 合成数据结果不作为正式评分。

## 本地验证记录

相关模型、优化器、数据、序列化及新增服务器配套测试可运行：

```bash
python -m pytest tests/test_server_training.py tests/test_model.py tests/test_optimizer.py \
  tests/test_nn_utils.py tests/test_data.py tests/test_serialization.py
```

本机 CPU 小数据验证不代表 A100 上完整 OWT 的吞吐、显存和验证时长；这些要在分配的服务器上确认。
