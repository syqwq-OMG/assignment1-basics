# 新服务器从零部署：下载 OWT → 训练 BPE → 训练 Transformer

本流程从原始文本开始，在新服务器重新生成 vocab、merges 和模型权重。无需复制旧服务器的分词器、token缓存或checkpoint。

以下命令在Linux、Bash环境执行。使用已验证的配置直接训练一组模型，不重复整轮参数搜索。

## 1. 准备服务器和拉取代码

基础条件：

- NVIDIA GPU和可用的驱动；默认配置使用BF16与torch.compile，原实验使用A100 80GB。
- `git`、`curl`、`gzip`、C/C++编译器；`torch.compile`需要编译工具链。
- 为原始数据、Python环境、每次实验生成的token文件和checkpoint预留空间，建议至少50GB可用磁盘。
- CPU和内存也用于完整OWT的BPE训练；编码进程数可按实际资源调整。

Ubuntu/Debian若缺少基础工具，可安装：

```bash
sudo apt-get update
sudo apt-get install -y git curl gzip build-essential tmux
```

已有工具或没有sudo时，直接使用服务器管理员提供的环境。先确认GPU可见：

```bash
nvidia-smi
```

建议在tmux中执行后续长任务，SSH断开后仍可继续：

```bash
tmux new -s assignment1
```

在tmux终端中拉取main：

```bash
git clone --branch main https://github.com/syqwq-OMG/assignment1-basics.git
cd assignment1-basics
mkdir -p logs data
```

## 2. 创建自己的Python环境

以下步骤仅用于新服务器上的个人环境；如果使用课程只读公共环境，直接激活公共环境并跳到第3步。

安装uv，创建Python 3.13环境：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv python install 3.13
uv venv --python 3.13 .venv
source .venv/bin/activate
```

先安装CUDA 12.8版PyTorch，再安装项目依赖：

```bash
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .

python -c 'import torch; print("torch:", torch.__version__); print("CUDA build:", torch.version.cuda); print("CUDA available:", torch.cuda.is_available())'
```

应能看到CUDA可用。原实验环境为Python 3.13、PyTorch 2.11.0+cu128、W&B 0.30.0。新服务器驱动须支持所安装的CUDA版PyTorch；`nvidia-smi`中的CUDA Version是驱动支持能力，与`torch.version.cuda`不一定完全相同。

安装方式参考：[uv官方说明](https://docs.astral.sh/uv/getting-started/installation/)、[PyTorch官方版本安装说明](https://pytorch.org/get-started/previous-versions/)。

## 3. 下载并解压原始OWT数据

下面的数据URL与课程作业说明一致，使用完整OWT训练集和验证集。若新服务器已经有这两份原始文本，可以直接使用原路径，并在第4步设置数据目录。

```bash
curl -fL --retry 3 \
  https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz \
  -o data/owt_train.txt.gz

curl -fL --retry 3 \
  https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz \
  -o data/owt_valid.txt.gz

gzip -d data/owt_train.txt.gz data/owt_valid.txt.gz
ls -lh data/owt_train.txt data/owt_valid.txt
```

原实验使用的解压文本大小为：训练集11920511059字节、验证集289998753字节。可快速检查：

```bash
wc -c data/owt_train.txt data/owt_valid.txt
```

文件大小检查用于发现下载或解压异常，不是内容哈希校验。不要对正式数据另行清洗、截断或改换行。

## 4. 设置GPU、数据路径和W&B

以下命令均在项目根目录执行。GPU编号改成新服务器分配给你的编号：

```bash
export CUDA_VISIBLE_DEVICES=0
export SERVER_PYTHON="$(command -v python)"
export ASSIGNMENT_DATA_DIR="$PWD/data"

export STUDENT_ID="10234900421"
export WANDB_ENTITY="syqwq-east-china-normal-university"
export WANDB_PROJECT="lmfs-assignment1"
export WANDB_MODE="online"
source ./w-b.sh

wandb login
```

`WANDB_ENTITY`和登录账号应为你有权限的账号或团队；API Key只在登录提示中输入，不写进代码或Git。

如果使用其他数据目录，修改`ASSIGNMENT_DATA_DIR`即可。不要沿用旧服务器上的`/data`别名或固定Python路径。

检查环境、数据和在线记录：

```bash
bash run_server.sh check --wandb
```

成功时会打印`Preflight passed.`；`--wandb`会创建一个独立的环境检查run。

## 5. 从零训练BPE，生成vocab和merges

```bash
set -o pipefail
bash run_server.sh bpe --num-threads 8 2>&1 | tee logs/bpe.log
```

该命令会：

1. 使用完整`owt_train.txt`训练本项目的BPE，词表上限32000，特殊token默认`<|endoftext|>`。
2. 生成`tokenizer.vocab.json`、`tokenizer.merges.json`和`tokenizer.special_tokens.json`。
3. 重新加载分词器，在完整`owt_valid.txt`上执行encode、decode及原文比较。
4. 将源码、tokenizer产物、前200个token和完整验证耗时记录到W&B。

文件保存在：

```text
outputs/本次BPE_RUN_ID/bpe/tokenizer.vocab.json
outputs/本次BPE_RUN_ID/bpe/tokenizer.merges.json
outputs/本次BPE_RUN_ID/bpe/tokenizer.special_tokens.json
outputs/本次BPE_RUN_ID/bpe/result.json
```

完成后终端会打印`Transformer --tokenizer-prefix ...`。从本次日志提取这个新路径：

```bash
export TOKENIZER_PREFIX="$(sed -n 's/^Transformer --tokenizer-prefix //p' logs/bpe.log | tail -n 1)"
test -n "$TOKENIZER_PREFIX"
ls -lh "$TOKENIZER_PREFIX".*.json
cat "$(dirname "$TOKENIZER_PREFIX")/result.json"
```

确认`bpe/roundtrip_passed`为`true`后再开始Transformer。若BPE命令失败，先检查`logs/bpe.log`，不要选用旧run产物继续。

BPE训练和验证是CPU工作，完整OWT可能需要较长时间；它不计入Transformer单次六小时预算。

## 6. 从零训练Transformer

先预览命令，确认数据目录和刚生成的tokenizer路径：

```bash
bash run_server.sh best \
  --tokenizer-prefix "$TOKENIZER_PREFIX" \
  --print-command
```

然后启动正式训练：

```bash
bash run_server.sh best \
  --tokenizer-prefix "$TOKENIZER_PREFIX" \
  --run-name "${STUDENT_ID}-owt-from-scratch" \
  2>&1 | tee logs/transformer.log
```

默认参数来自`configs/owt_best.json`：8层、宽度768、FFN2048、12个头、上下文512、batch64，使用BF16和torch.compile。学习率从0.0006衰减至0.00006。

训练入口会使用新BPE重新编码完整训练/验证文本，随机初始化模型，记录每100次更新的训练PPL，保存最后checkpoint并进行完整验证。数据编码、编译、训练、保存和完整验证均计入21600秒预算；默认预留900秒用于最后保存和验证。

这是一组独立的新实验，不加载旧模型，也不需要启动历史参数搜索调度器。不同服务器的速度不同，六小时内完成的更新步数和最终PPL也会不同。

显存不足时可以从较小batch开始，例如：

```bash
bash run_server.sh best \
  --tokenizer-prefix "$TOKENIZER_PREFIX" \
  --batch-size 16 \
  --run-name "${STUDENT_ID}-owt-from-scratch-b16" \
  2>&1 | tee logs/transformer-b16.log
```

若GPU不支持BF16，可以额外传`--precision fp32`。GPU分配允许的总时长仍需自行遵守；BPE耗时加Transformer预算可能超过六小时。

## 7. 查看结果和保存产物

训练结束后终端会打印本次Transformer run的URL与输出目录。对应文件是：

```text
outputs/本次TRANSFORMER_RUN_ID/last.pt
outputs/本次TRANSFORMER_RUN_ID/result.json
outputs/本次TRANSFORMER_RUN_ID/source/
outputs/本次TRANSFORMER_RUN_ID/source_dump.txt
```

W&B中检查：

- Charts的`train/ppl`：训练曲线。
- Summary的`val/ppl`：最后保存模型的完整验证PPL。
- `time/total_seconds`、`time/within_6h`：累计耗时及预算检查。
- `source_changed`：运行期间源码是否有变化，正常应为空列表。

本课程提交BPE与Transformer两个正式run链接，详见[提交说明](submission.md)。保留新生成的三份tokenizer文件、checkpoint和result.json供复查。

## 8. SSH断开或重新进入会话

tmux中按`Ctrl-b`再按`d`退出显示，任务仍会继续。重新连接服务器后：

```bash
tmux attach -t assignment1
```

若是新开了普通终端，需要重新激活环境、source `w-b.sh`，并设置GPU、数据目录和`SERVER_PYTHON`。`TOKENIZER_PREFIX`应指向这次BPE的产物；可以重新从`logs/bpe.log`提取。

完整训练会在程序内执行时间检查。若进程异常退出且需要续训，应使用上次`last.pt`和`result.json`继承累计耗时，方法见[服务器指南](server-training.md#4-续训)。
